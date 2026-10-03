pub mod pick;
pub mod predict;

use std::collections::HashMap;

use azalea_block::BlockState;
use azalea_core::{
    delta::LpVec3,
    direction::Direction,
    game_type::GameMode,
    hit_result::{BlockHitResult, HitResult},
    position::{BlockPos, Vec3},
    tick::GameTick,
};
use azalea_entity::{
    Attributes, Dead, EntityKindComponent, LocalEntity, LookDirection, PlayerAbilities, Position,
    attributes::{
        creative_block_interaction_range_modifier, creative_entity_interaction_range_modifier,
    },
    clamp_look_direction,
    dimensions::EntityDimensions,
    indexing::EntityIdIndex,
    inventory::Inventory,
    metadata::FallFlying,
};
use azalea_inventory::{ItemStack, ItemStackData, components};
use azalea_physics::{
    PhysicsSystems, collision::entity_collisions::update_last_bounding_box,
    local_player::PhysicsState,
};
use azalea_protocol::packets::game::{
    ServerboundInteract, ServerboundUseItem, s_interact::InteractionHand,
    s_swing::ServerboundSwing, s_use_item_on::ServerboundUseItemOn,
};
use azalea_world::{World, WorldName, Worlds};
use bevy_app::{App, Plugin, Update};
use bevy_ecs::prelude::*;
use tracing::warn;

use super::mining::Mining;
use crate::{
    attack::handle_attack_event,
    interact::pick::{HitResultComponent, update_hit_result_component},
    inventory::InventorySystems,
    local_player::{Hunger, LocalGameMode, PermissionLevel},
    movement::MoveEventsSystems,
    packet::game::SendGamePacketEvent,
    respawn::perform_respawn,
};

/// A plugin that allows clients to interact with blocks in the world.
pub struct InteractPlugin;
impl Plugin for InteractPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<StartUseItemEvent>()
            .add_systems(
                Update,
                (
                    update_attributes_for_gamemode,
                    handle_start_use_item_event,
                    update_hit_result_component
                        .after(clamp_look_direction)
                        .after(update_last_bounding_box),
                )
                    .after(InventorySystems)
                    .after(MoveEventsSystems)
                    .after(perform_respawn)
                    .after(handle_attack_event)
                    .chain(),
            )
            .add_systems(
                GameTick,
                handle_start_use_item_queued.before(PhysicsSystems),
            )
            .add_observer(handle_entity_interact)
            .add_observer(handle_swing_arm_trigger);
    }
}

/// A component that contains information about our local block state
/// predictions.
#[derive(Clone, Component, Debug, Default)]
pub struct BlockStatePredictionHandler {
    /// The total number of changes that this client has made to blocks.
    seq: u32,
    server_state: HashMap<BlockPos, ServerVerifiedState>,
}
#[derive(Clone, Debug)]
struct ServerVerifiedState {
    seq: u32,
    block_state: BlockState,
    /// Used for teleporting the player back if we're colliding with the block
    /// that got placed back.
    #[allow(unused)]
    player_pos: Vec3,
}

impl BlockStatePredictionHandler {
    /// Get the next sequence number that we're going to use and increment the
    /// value.
    pub fn start_predicting(&mut self) -> u32 {
        self.seq += 1;
        self.seq
    }

    /// Return whether `pos` still has a locally predicted block state awaiting
    /// the server's acknowledgement.
    ///
    /// Callers that report action outcomes should not treat the current world
    /// value as server-confirmed while this is true: an acknowledgement may
    /// still keep it or roll it back.
    pub fn is_prediction_pending(&self, pos: BlockPos) -> bool {
        self.server_state.contains_key(&pos)
    }

    /// Should be called right before the client updates a block with its
    /// prediction.
    ///
    /// This is used to make sure that we can rollback to this state if the
    /// server acknowledges the sequence number (with
    /// [`ClientboundBlockChangedAck`]) without having sent a block update.
    ///
    /// [`ClientboundBlockChangedAck`]: azalea_protocol::packets::game::ClientboundBlockChangedAck
    pub fn retain_known_server_state(
        &mut self,
        pos: BlockPos,
        old_state: BlockState,
        player_pos: Vec3,
    ) {
        self.server_state
            .entry(pos)
            .and_modify(|s| s.seq = self.seq)
            .or_insert(ServerVerifiedState {
                seq: self.seq,
                block_state: old_state,
                player_pos,
            });
    }

    /// Save this update as the correct server state so when the server sends a
    /// [`ClientboundBlockChangedAck`] we don't roll back this new update.
    ///
    /// This should be used when we receive a block update from the server.
    ///
    /// [`ClientboundBlockChangedAck`]: azalea_protocol::packets::game::ClientboundBlockChangedAck
    pub fn update_known_server_state(&mut self, pos: BlockPos, state: BlockState) -> bool {
        if let Some(s) = self.server_state.get_mut(&pos) {
            s.block_state = state;
            true
        } else {
            false
        }
    }

    pub fn end_prediction_up_to(&mut self, seq: u32, world: &World) {
        let mut to_remove = Vec::new();
        for (pos, state) in &self.server_state {
            if state.seq > seq {
                continue;
            }
            to_remove.push(*pos);

            // syncBlockState
            let client_block_state = world.get_block_state(*pos).unwrap_or_default();
            let server_block_state = state.block_state;
            if client_block_state == server_block_state {
                continue;
            }
            world.set_block_state(*pos, server_block_state);
            // TODO: implement these two functions
            // if is_colliding(player, *pos, server_block_state) {
            //     abs_snap_to(state.player_pos);
            // }
        }

        for pos in to_remove {
            self.server_state.remove(&pos);
        }
    }
}

#[cfg(test)]
mod prediction_tests {
    use azalea_registry::builtin::BlockKind;

    use super::*;

    #[test]
    fn pending_prediction_is_visible_until_the_ack_sequence_ends_it() {
        let pos = BlockPos::new(1, 64, 2);
        let mut handler = BlockStatePredictionHandler::default();
        let seq = handler.start_predicting();
        handler.retain_known_server_state(pos, BlockKind::Stone.into(), Vec3::ZERO);

        assert!(handler.is_prediction_pending(pos));
        handler.end_prediction_up_to(seq, &World::default());
        assert!(!handler.is_prediction_pending(pos));
    }
}

/// An event that makes one of our clients simulate a right-click.
///
/// This event just inserts the [`StartUseItemQueued`] component on the given
/// entity.
#[doc(alias("right click"))]
#[derive(Message)]
pub struct StartUseItemEvent {
    pub entity: Entity,
    pub hand: InteractionHand,
    /// See [`StartUseItemQueued::force_block`].
    pub force_block: Option<BlockPos>,
}
pub fn handle_start_use_item_event(
    mut commands: Commands,
    mut events: MessageReader<StartUseItemEvent>,
) {
    for event in events.read() {
        commands.entity(event.entity).insert(StartUseItemQueued {
            hand: event.hand,
            force_block: event.force_block,
        });
    }
}

/// A component that makes our client simulate a right-click on the next
/// [`GameTick`]. It's removed after that tick.
///
/// You may find it more convenient to use [`StartUseItemEvent`] instead, which
/// just inserts this component for you.
///
/// [`GameTick`]: azalea_core::tick::GameTick
#[derive(Component, Debug)]
pub struct StartUseItemQueued {
    pub hand: InteractionHand,
    /// Optionally force us to send a [`ServerboundUseItemOn`] on the given
    /// block.
    ///
    /// This is useful if you want to interact with a block without looking at
    /// it, but should be avoided to stay compatible with anticheats.
    pub force_block: Option<BlockPos>,
}
/// The latest block that one of our right clicks was predicted to place.
///
/// Vanilla places the block on the client as soon as the click lands and lets
/// the server correct it; azalea doesn't change its world, so this records what
/// the click did instead. `seq` grows with every new placement.
#[derive(Clone, Component, Copy, Debug)]
pub struct PredictedPlacement {
    pub placement: predict::Placement,
    pub seq: u32,
}

/// The latest block that one of our right clicks used directly
/// (`useWithoutItem`), like flipping a lever or opening a door.
///
/// When vanilla's client changes the block itself, azalea does the same to its
/// world as a prediction the server can roll back, and `block_use` lists the
/// new states. `seq` grows with every new use.
#[derive(Clone, Component, Debug)]
pub struct PredictedBlockUse {
    pub pos: BlockPos,
    pub block_use: predict::BlockUse,
    pub seq: u32,
}

/// Vanilla's `Minecraft.startUseItem`: try each hand in turn (from the queued
/// hand onwards), first on the targeted entity or block and then by using the
/// held item, and stop at the first step that succeeds (or, for a block, at a
/// failure).
///
/// Every step sends its packet, like vanilla does; whether the next step runs
/// is decided by [`predict`], which reproduces the client-side result vanilla
/// computes before moving on.
#[allow(clippy::type_complexity)]
pub fn handle_start_use_item_queued(
    mut commands: Commands,
    mut query: Query<(
        Entity,
        &StartUseItemQueued,
        &mut BlockStatePredictionHandler,
        &HitResultComponent,
        &LookDirection,
        Option<&Mining>,
        (
            &LocalGameMode,
            &PlayerAbilities,
            &PermissionLevel,
            &PhysicsState,
            &Hunger,
            &Inventory,
            &Position,
            &EntityDimensions,
            &Attributes,
            Option<&FallFlying>,
            &WorldName,
        ),
    )>,
    worlds: Res<Worlds>,
    targets: Query<(&EntityKindComponent, Has<Dead>)>,
    placements: Query<&PredictedPlacement>,
    block_uses: Query<&PredictedBlockUse>,
) {
    for (
        entity,
        start_use_item,
        mut prediction_handler,
        hit_result,
        look_direction,
        mining,
        (
            game_mode,
            abilities,
            permission_level,
            physics_state,
            hunger,
            inventory,
            position,
            dimensions,
            attributes,
            fall_flying,
            world_name,
        ),
    ) in &mut query
    {
        commands.entity(entity).remove::<StartUseItemQueued>();

        // `if (!this.gameMode.isDestroying())`
        if mining.is_some() {
            continue;
        }
        let Some(world_lock) = worlds.get(world_name) else {
            continue;
        };
        let world = world_lock.read();

        // TODO: vanilla also skips this while `LocalPlayer.handsBusy` (rowing a
        // boat).

        let mut hit_result = (**hit_result).clone();
        if let Some(force_block) = start_use_item.force_block {
            let hit_result_matches = if let HitResult::Block(block_hit_result) = &hit_result {
                block_hit_result.block_pos == force_block
            } else {
                false
            };

            if !hit_result_matches {
                // we're not looking at the block, so make up some numbers
                hit_result = HitResult::Block(BlockHitResult {
                    location: force_block.center(),
                    direction: Direction::Up,
                    block_pos: force_block,
                    inside: false,
                    world_border: false,
                    miss: false,
                });
            }
        }

        let actor = predict::Actor {
            game_mode: game_mode.current,
            abilities,
            permission_level: **permission_level,
            sneaking: physics_state.trying_to_crouch,
            food: hunger.food,
            fall_flying: fall_flying.is_some_and(|f| **f),
            inventory,
            eye_position: position.up(dimensions.eye_height.into()),
            look_direction: *look_direction,
            block_interaction_range: attributes.block_interaction_range.calculate(),
            bounding_box: dimensions.make_bounding_box(**position),
        };
        let swing = |commands: &mut Commands, hand: InteractionHand| {
            commands.trigger(SendGamePacketEvent::new(entity, ServerboundSwing { hand }));
        };

        let hands: &[InteractionHand] = match start_use_item.hand {
            InteractionHand::MainHand => &[InteractionHand::MainHand, InteractionHand::OffHand],
            InteractionHand::OffHand => &[InteractionHand::OffHand],
        };
        for &hand in hands {
            match &hit_result {
                HitResult::Entity(r) => {
                    // the pick range already limits this to the entity
                    // interaction range
                    commands.trigger(EntityInteractEvent {
                        client: entity,
                        target: r.entity,
                        location: Some(r.location),
                        hand,
                    });
                    let target = targets
                        .get(r.entity)
                        .ok()
                        .map(|(kind, dead)| predict::Target {
                            kind: **kind,
                            alive: !dead,
                        });
                    if let Some(target) = target {
                        let result = predict::interact(&actor, hand, &target);
                        if let predict::InteractionResult::Success(source) = result {
                            if source == predict::Swing::Client {
                                swing(&mut commands, hand);
                            }
                            break;
                        }
                    }
                }
                HitResult::Block(r) if !r.miss => {
                    // TODO: vanilla fails here when the block is outside the
                    // world border.
                    let seq = prediction_handler.start_predicting();
                    commands.trigger(SendGamePacketEvent::new(
                        entity,
                        ServerboundUseItemOn {
                            hand,
                            block_hit: r.into(),
                            seq,
                        },
                    ));
                    let predict::UseOn {
                        result,
                        placement,
                        block_use,
                    } = predict::use_item_on_placing(&world, &actor, hand, r);
                    if let Some(placement) = placement {
                        let seq = placements.get(entity).map_or(0, |last| last.seq) + 1;
                        commands
                            .entity(entity)
                            .insert(PredictedPlacement { placement, seq });
                    }
                    if let Some(block_use) = block_use {
                        if let predict::BlockUse::Client(changes) = &block_use {
                            for (pos, state) in changes {
                                let old = world.get_block_state(*pos).unwrap_or_default();
                                prediction_handler.retain_known_server_state(*pos, old, **position);
                                world.set_block_state(*pos, *state);
                            }
                        }
                        let seq = block_uses.get(entity).map_or(0, |last| last.seq) + 1;
                        commands.entity(entity).insert(PredictedBlockUse {
                            pos: r.block_pos,
                            block_use,
                            seq,
                        });
                    }
                    match result {
                        predict::InteractionResult::Success(source) => {
                            if source == predict::Swing::Client {
                                swing(&mut commands, hand);
                            }
                            break;
                        }
                        predict::InteractionResult::Fail => break,
                        _ => {}
                    }
                }
                HitResult::Block(_) => {}
            }

            if !actor.item(hand).is_empty() {
                let seq = prediction_handler.start_predicting();
                commands.trigger(SendGamePacketEvent::new(
                    entity,
                    ServerboundUseItem {
                        hand,
                        seq,
                        x_rot: look_direction.x_rot(),
                        y_rot: look_direction.y_rot(),
                    },
                ));
                if let predict::InteractionResult::Success(source) =
                    predict::use_item(&world, &actor, hand)
                {
                    if source == predict::Swing::Client {
                        swing(&mut commands, hand);
                    }
                    break;
                }
            }
        }
    }
}

/// An ECS `Event` that makes the client tell the server that we right-clicked
/// an entity.
#[derive(Clone, Debug, EntityEvent)]
pub struct EntityInteractEvent {
    #[event_target]
    pub client: Entity,
    pub target: Entity,
    /// The position on the entity that we'll tell the server that we clicked
    /// on.
    ///
    /// This doesn't matter for most entities. If it's set to `None` but we're
    /// looking at the target, it'll use the correct value. If it's `None` and
    /// we're not looking at the entity, then it'll arbitrary send the target's
    /// exact position.
    pub location: Option<Vec3>,
    /// The hand we're interacting with.
    pub hand: InteractionHand,
}

pub fn handle_entity_interact(
    trigger: On<EntityInteractEvent>,
    mut commands: Commands,
    client_query: Query<(&PhysicsState, &EntityIdIndex, &HitResultComponent)>,
    target_query: Query<&Position>,
) {
    let Some((physics_state, entity_id_index, hit_result)) = client_query.get(trigger.client).ok()
    else {
        warn!(
            "tried to interact with an entity but the client didn't have the required components"
        );
        return;
    };

    // TODO: worldborder check

    let Some(entity_id) = entity_id_index.get_by_ecs_entity(trigger.target) else {
        warn!("tried to interact with an entity that isn't known by the client");
        return;
    };

    let location = if let Some(l) = trigger.location {
        l
    } else {
        // if we're looking at the entity, use that
        if let Some(entity_hit_result) = hit_result.as_entity_hit_result()
            && entity_hit_result.entity == trigger.target
        {
            entity_hit_result.location
        } else {
            // if we're not looking at the entity, make up a value that's good enough by
            // using the entity's position
            let Ok(target_position) = target_query.get(trigger.target) else {
                warn!("tried to look at an entity without the entity having a position");
                return;
            };
            **target_position
        }
    };

    // `MultiPlayerGameMode.interact` sends exactly one packet (the separate
    // interact/interact-at pair is gone since the packet carries the location).
    let interact = ServerboundInteract {
        entity_id,
        hand: trigger.hand,
        location: LpVec3::from(location),
        using_secondary_action: physics_state.trying_to_crouch,
    };
    commands.trigger(SendGamePacketEvent::new(trigger.client, interact));
}

/// Whether we can't interact with the block, based on your gamemode.
///
/// If this is false, then we can interact with the block.
///
/// The world, block position, and inventory are used for the adventure mode
/// check.
pub fn check_is_interaction_restricted(
    world: &World,
    block_pos: BlockPos,
    game_mode: &GameMode,
    inventory: &Inventory,
) -> bool {
    match game_mode {
        GameMode::Adventure => {
            // vanilla checks for abilities.mayBuild here but servers have no
            // way of modifying that

            let held_item = inventory.held_item();
            match &held_item {
                ItemStack::Present(item) => {
                    let block = world.chunks.get_block_state(block_pos);
                    let Some(block) = block else {
                        // block isn't loaded so just say that it is restricted
                        return true;
                    };
                    check_block_can_be_broken_by_item_in_adventure_mode(item, &block)
                }
                _ => true,
            }
        }
        GameMode::Spectator => true,
        _ => false,
    }
}

/// Check if the item has the `CanDestroy` tag for the block.
pub fn check_block_can_be_broken_by_item_in_adventure_mode(
    item: &ItemStackData,
    _block: &BlockState,
) -> bool {
    // minecraft caches the last checked block but that's kind of an unnecessary
    // optimization and makes the code too complicated

    if item.get_component::<components::CanBreak>().is_none() {
        // no CanDestroy tag
        return false;
    };

    false

    // for block_predicate in can_destroy {
    //     // TODO
    //     // defined in BlockPredicateArgument.java
    // }

    // true
}

pub fn can_use_game_master_blocks(
    abilities: &PlayerAbilities,
    permission_level: &PermissionLevel,
) -> bool {
    abilities.instant_break && **permission_level >= 2
}

/// Swing your arm.
///
/// This is purely a visual effect and won't interact with anything in the
/// world.
#[derive(Clone, Debug, EntityEvent)]
pub struct SwingArmEvent {
    pub entity: Entity,
}
pub fn handle_swing_arm_trigger(swing_arm: On<SwingArmEvent>, mut commands: Commands) {
    commands.trigger(SendGamePacketEvent::new(
        swing_arm.entity,
        ServerboundSwing {
            hand: InteractionHand::MainHand,
        },
    ));
}

#[allow(clippy::type_complexity)]
fn update_attributes_for_gamemode(
    query: Query<(&mut Attributes, &LocalGameMode), (With<LocalEntity>, Changed<LocalGameMode>)>,
) {
    for (mut attributes, game_mode) in query {
        if game_mode.current == GameMode::Creative {
            attributes
                .block_interaction_range
                .insert(creative_block_interaction_range_modifier());
            attributes
                .entity_interaction_range
                .insert(creative_entity_interaction_range_modifier());
        } else {
            attributes
                .block_interaction_range
                .remove(&creative_block_interaction_range_modifier().id);
            attributes
                .entity_interaction_range
                .remove(&creative_entity_interaction_range_modifier().id);
        }
    }
}
