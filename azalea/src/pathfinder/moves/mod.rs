pub mod basic;
pub mod parkour;
pub mod uncommon;

use std::{
    fmt::{self, Debug},
    sync::Arc,
};

use azalea_block::BlockState;
use azalea_client::{
    PhysicsState, SprintDirection, StartSprintEvent, StartWalkEvent, WalkDirection,
    interact::StartUseItemQueued, inventory::SetSelectedHotbarSlotEvent,
    mining::StartMiningBlockEvent,
};
use azalea_core::position::{BlockPos, Vec3};
use azalea_inventory::Menu;
use azalea_protocol::packets::game::s_interact::InteractionHand;
use azalea_registry::builtin::BlockKind;
use azalea_world::World;
use bevy_ecs::{entity::Entity, message::MessageWriter, system::Commands, world::EntityWorldMut};
use parking_lot::RwLock;
use tracing::debug;

use super::{
    astar,
    custom_state::CustomPathfinderStateRef,
    mining::MiningCache,
    placing::{PlacementCache, placeable_in_hotbar},
    positions::RelBlockPos,
    world::{CachedWorld, is_block_state_passable},
};
use crate::{
    auto_tool::best_tool_in_hotbar_for_block,
    bot::{JumpEvent, LookAtEvent},
    pathfinder::player_pos_to_block_pos,
};

type Edge = astar::Edge<RelBlockPos, MoveData>;

pub type SuccessorsFn = fn(&mut MovesCtx, RelBlockPos);

/// Re-implement certain bugs and quirks that Baritone has, and disable
/// movements that Baritone doesn't have.
///
/// Meant to help with debugging when directly comparing against Baritone.
pub const BARITONE_COMPAT: bool = false;

pub fn default_move(ctx: &mut MovesCtx, node: RelBlockPos) {
    basic::basic_move(ctx, node);
    parkour::parkour_move(ctx, node);
    uncommon::uncommon_move(ctx, node);
}

/// 一次移动最多声明几处世界替换。挖和放共用这个额度——它们是同一件事。
const MAX_REPLACEMENTS_PER_MOVEMENT: usize = 6;

/// A block position relative to the source node of a movement.
///
/// Keeping side effects relative makes the declaration valid both while A*
/// searches in [`RelBlockPos`] space and after the path is mapped back to
/// absolute [`BlockPos`] coordinates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockOffset {
    x: i32,
    y: i32,
    z: i32,
}

impl BlockOffset {
    pub(crate) fn between(source: RelBlockPos, target: RelBlockPos) -> Self {
        Self {
            x: i32::from(target.x) - i32::from(source.x),
            y: target.y - source.y,
            z: i32::from(target.z) - i32::from(source.z),
        }
    }

    fn apply(self, source: BlockPos) -> BlockPos {
        BlockPos::new(source.x + self.x, source.y + self.y, source.z + self.z)
    }
}

/// 替换之后那一格应该是什么。
///
/// 挖和放不是两件事：挖是把一格换成空气，放是把一格换成实心方块。规划期只关心
/// 「换完之后这里能不能通行、能不能站」，具体放哪一种方块是执行期按库存挑的。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplaceTarget {
    /// 空气。执行期表现为挖掉这一格。
    Air,
    /// 一块实心方块。执行期表现为放置。
    Solid,
}

/// 一处世界替换：把 `at` 这一格换成 `to`。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Replacement {
    at: BlockOffset,
    to: ReplaceTarget,
}

impl Replacement {
    /// Resolve this replacement's position for a concrete movement source.
    pub fn at(&self, source: BlockPos) -> BlockPos {
        self.at.apply(source)
    }

    pub fn target(&self) -> ReplaceTarget {
        self.to
    }
}

/// The exact world changes a movement is allowed to make.
///
/// Baritone keeps these in two places (`positionsToBreak` / `positionToPlace`);
/// here they are one list, because breaking and placing are the same operation
/// with a different target. Movement execution may react to the world changing,
/// but it may never touch a position this planned edge did not declare.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MovementSideEffects {
    replacements: [Option<Replacement>; MAX_REPLACEMENTS_PER_MOVEMENT],
}

impl MovementSideEffects {
    /// 声明这些格会被挖成空气。
    pub(crate) fn to_air<const N: usize>(source: RelBlockPos, blocks: [RelBlockPos; N]) -> Self {
        Self::to_air_iter(source, blocks)
    }

    pub(crate) fn to_air_iter(
        source: RelBlockPos,
        blocks: impl IntoIterator<Item = RelBlockPos>,
    ) -> Self {
        let mut side_effects = Self::default();
        for block in blocks {
            side_effects.push(source, block, ReplaceTarget::Air);
        }
        side_effects
    }

    /// 追加声明：这一格会被放上实心方块。
    #[allow(dead_code)]
    pub(crate) fn and_solid(mut self, source: RelBlockPos, block: RelBlockPos) -> Self {
        self.push(source, block, ReplaceTarget::Solid);
        self
    }

    fn push(&mut self, source: RelBlockPos, block: RelBlockPos, to: ReplaceTarget) {
        let slot = self
            .replacements
            .iter_mut()
            .find(|slot| slot.is_none())
            .expect("a movement declared too many world replacements");
        *slot = Some(Replacement {
            at: BlockOffset::between(source, block),
            to,
        });
    }

    /// Resolve every declared replacement for a concrete movement source.
    pub fn replacements(
        &self,
        source: BlockPos,
    ) -> impl Iterator<Item = (BlockPos, ReplaceTarget)> + use<'_> {
        self.replacements
            .iter()
            .flatten()
            .map(move |replacement| (replacement.at.apply(source), replacement.to))
    }

    /// 声明里要换成空气的那些格——也就是允许挖的坐标。
    pub fn blocks_to_break(&self, source: BlockPos) -> impl Iterator<Item = BlockPos> + use<'_> {
        self.declared(source, ReplaceTarget::Air)
    }

    /// 声明里要换成实心的第一格——也就是允许放的坐标。
    pub fn block_to_place(&self, source: BlockPos) -> Option<BlockPos> {
        self.declared(source, ReplaceTarget::Solid).next()
    }

    fn declared(
        &self,
        source: BlockPos,
        target: ReplaceTarget,
    ) -> impl Iterator<Item = BlockPos> + use<'_> {
        self.replacements
            .iter()
            .flatten()
            .filter(move |replacement| replacement.to == target)
            .map(move |replacement| replacement.at.apply(source))
    }

    fn allows_break(&self, source: BlockPos, block: BlockPos) -> bool {
        self.blocks_to_break(source)
            .any(|declared| declared == block)
    }

    #[allow(dead_code)]
    fn allows_place(&self, source: BlockPos, block: BlockPos) -> bool {
        self.declared(source, ReplaceTarget::Solid)
            .any(|declared| declared == block)
    }
}

#[derive(Clone)]
pub struct MoveData {
    /// Use the context to determine what events should be sent to complete this
    /// movement.
    pub execute: &'static (dyn Fn(ExecuteCtx) + Send + Sync),
    /// Whether we've reached the target.
    pub is_reached: &'static (dyn Fn(IsReachedCtx) -> bool + Send + Sync),
    /// Exact positions this movement may change, relative to its source node.
    pub side_effects: MovementSideEffects,
}

impl MoveData {
    pub(crate) fn new(
        execute: &'static (dyn Fn(ExecuteCtx) + Send + Sync),
        is_reached: &'static (dyn Fn(IsReachedCtx) -> bool + Send + Sync),
    ) -> Self {
        Self {
            execute,
            is_reached,
            side_effects: MovementSideEffects::default(),
        }
    }

    pub(crate) fn with_side_effects(mut self, side_effects: MovementSideEffects) -> Self {
        self.side_effects = side_effects;
        self
    }
}
impl Debug for MoveData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MoveData")
            .field("side_effects", &self.side_effects)
            .finish()
    }
}

pub struct ExecuteCtx<'s, 'w1, 'w2, 'w3, 'w4, 'w5, 'w6, 'a> {
    pub entity: Entity,
    /// The node that we're trying to reach.
    pub target: BlockPos,
    /// The last node that we reached.
    pub start: BlockPos,
    pub position: Vec3,
    pub physics: &'a azalea_entity::Physics,
    pub is_currently_mining: bool,
    pub can_mine: bool,
    pub can_place: bool,
    pub side_effects: MovementSideEffects,
    pub world: Arc<RwLock<World>>,
    pub menu: Menu,

    pub commands: &'a mut Commands<'w1, 's>,
    pub look_at_events: &'a mut MessageWriter<'w2, LookAtEvent>,
    pub sprint_events: &'a mut MessageWriter<'w3, StartSprintEvent>,
    pub walk_events: &'a mut MessageWriter<'w4, StartWalkEvent>,
    pub jump_events: &'a mut MessageWriter<'w5, JumpEvent>,
    pub start_mining_events: &'a mut MessageWriter<'w6, StartMiningBlockEvent>,
}

impl ExecuteCtx<'_, '_, '_, '_, '_, '_, '_, '_> {
    pub fn on_tick_start(&mut self) {
        self.set_sneaking(false);
    }

    pub fn look_at(&mut self, position: Vec3) {
        self.look_at_events.write(LookAtEvent {
            entity: self.entity,
            position: Vec3 {
                x: position.x,
                // look forward
                y: self.position.up(1.53).y,
                z: position.z,
            },
        });
    }

    pub fn look_at_exact(&mut self, position: Vec3) {
        self.look_at_events.write(LookAtEvent {
            entity: self.entity,
            position,
        });
    }

    pub fn sprint(&mut self, direction: SprintDirection) {
        self.sprint_events.write(StartSprintEvent {
            entity: self.entity,
            direction,
        });
    }

    pub fn walk(&mut self, direction: WalkDirection) {
        self.walk_events.write(StartWalkEvent {
            entity: self.entity,
            direction,
        });
    }

    pub fn jump(&mut self) {
        self.jump_events.write(JumpEvent {
            entity: self.entity,
        });
    }

    fn set_sneaking(&mut self, sneaking: bool) {
        self.commands
            .entity(self.entity)
            .queue(move |mut entity: EntityWorldMut<'_>| {
                if let Some(mut physics_state) = entity.get_mut::<PhysicsState>() {
                    physics_state.trying_to_crouch = sneaking;
                }
            });
    }
    pub fn sneak(&mut self) {
        self.set_sneaking(true);
    }

    pub fn jump_if_in_water(&mut self) {
        if self.physics.is_in_water() {
            self.jump();
        }
    }

    /// Returns whether this block could be mined.
    pub fn should_mine(&mut self, block: BlockPos) -> bool {
        if !self.side_effects.allows_break(self.start, block) {
            return false;
        }
        let block_state = self.world.read().get_block_state(block).unwrap_or_default();
        should_mine_block_state(block_state)
    }

    /// Mine the block at the given position.
    ///
    /// Returns whether the block is being mined.
    pub fn mine(&mut self, block: BlockPos) -> bool {
        if !self.can_mine || !self.side_effects.allows_break(self.start, block) {
            return false;
        }

        let block_state = self.world.read().get_block_state(block).unwrap_or_default();
        if is_block_state_passable(block_state) {
            // block is already passable, no need to mine it
            return false;
        }

        let best_tool_result = best_tool_in_hotbar_for_block(block_state, &self.menu);
        debug!("best tool for {block_state:?}: {best_tool_result:?}");

        self.commands.trigger(SetSelectedHotbarSlotEvent {
            entity: self.entity,
            slot: best_tool_result.index as u8,
        });

        self.is_currently_mining = true;

        self.walk(WalkDirection::None);
        self.look_at_exact(block.center());
        self.start_mining_events.write(StartMiningBlockEvent {
            entity: self.entity,
            position: block,
            force: true,
        });

        true
    }

    /// Place a block at the given position.
    ///
    /// Returns whether a placement was actually sent.
    ///
    /// 挖的对称面（见 [`ReplaceTarget`]）：同样只碰规划时声明过的坐标，同样先把
    /// 手上换成合适的东西再动手。区别只在于换成的是方块而不是空气。
    pub fn place(&mut self, block: BlockPos) -> bool {
        if !self.can_place || !self.side_effects.allows_place(self.start, block) {
            return false;
        }
        if !is_block_state_passable(self.get_block_state(block)) {
            // 已经有东西了，不用放——和 mine 碰上已经是空气一样，这不是失败。
            return false;
        }

        // 贴着下面那一格的上表面放：force_block 会伪造 Direction::Up 的命中，
        // 于是方块正好落在 block 上。下面是空的就贴不住，什么也别发。
        let against = block.down(1);
        if is_block_state_passable(self.get_block_state(against)) {
            return false;
        }

        let Some(slot) = placeable_in_hotbar(&self.menu) else {
            return false;
        };
        self.commands.trigger(SetSelectedHotbarSlotEvent {
            entity: self.entity,
            slot: slot as u8,
        });

        self.walk(WalkDirection::None);
        self.look_at_exact(against.center());
        self.commands
            .entity(self.entity)
            .insert(StartUseItemQueued {
                hand: InteractionHand::MainHand,
                force_block: Some(against),
            });

        true
    }

    /// Mine the given block, but make sure the player is standing at the start
    /// of the current node first.
    pub fn mine_while_at_start(&mut self, block: BlockPos) -> bool {
        let horizontal_distance_from_start = (self.start.center() - self.position)
            .horizontal_distance_squared()
            .sqrt();
        let at_start_position = player_pos_to_block_pos(self.position) == self.start
            && horizontal_distance_from_start < 0.25;

        if self.should_mine(block) {
            if at_start_position {
                self.look_at(block.center());
                self.mine(block);
            } else {
                self.look_at(self.start.center());
                self.walk(WalkDirection::Forward);
            }
            true
        } else {
            false
        }
    }

    pub fn get_block_state(&self, block: BlockPos) -> BlockState {
        self.world.read().get_block_state(block).unwrap_or_default()
    }
}

pub fn should_mine_block_state(block_state: BlockState) -> bool {
    if is_block_state_passable(block_state) || BlockKind::from(block_state) == BlockKind::Water {
        // block is already passable, no need to mine it
        return false;
    }

    true
}

pub struct IsReachedCtx<'a> {
    /// The node that we're trying to reach.
    pub target: BlockPos,
    /// The last node that we reached.
    pub start: BlockPos,
    pub position: Vec3,
    pub physics: &'a azalea_entity::Physics,
}

/// Returns whether the entity is at the node and should start going to the
/// next node.
#[must_use]
pub fn default_is_reached(
    IsReachedCtx {
        position,
        target,
        physics,
        ..
    }: IsReachedCtx,
) -> bool {
    let block_pos = player_pos_to_block_pos(position);
    if block_pos == target {
        return true;
    }
    // it's fine if we go over the target while swimming
    if physics.is_in_water() && block_pos.down(1) == target {
        return true;
    }

    false
}

pub struct MovesCtx<'a> {
    pub edges: &'a mut Vec<Edge>,
    pub world: &'a CachedWorld,
    pub mining_cache: &'a MiningCache,
    /// 规划期「能不能放、放一格多贵」的答案。挖和放是同一件事的两半。
    pub placement: &'a PlacementCache,
    pub custom_state: &'a CustomPathfinderStateRef,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn side_effects_resolve_relative_to_the_concrete_movement_source() {
        let relative_source = RelBlockPos::new(12, 70, -8);
        let first_break = relative_source.up(1);
        let second_break = RelBlockPos::new(13, 70, -8);
        let place = relative_source.down(1);
        let side_effects =
            MovementSideEffects::to_air(relative_source, [first_break, second_break])
                .and_solid(relative_source, place);

        let absolute_source = BlockPos::new(100, 70, -200);
        assert_eq!(
            side_effects
                .blocks_to_break(absolute_source)
                .collect::<Vec<_>>(),
            [absolute_source.up(1), absolute_source.east(1)]
        );
        assert_eq!(
            side_effects.block_to_place(absolute_source),
            Some(absolute_source.down(1))
        );
        assert!(side_effects.allows_break(absolute_source, absolute_source.up(1)));
        assert!(!side_effects.allows_break(absolute_source, absolute_source.west(1)));
    }

    /// 挖和放共用一份清单，所以「允许挖」和「允许放」不能互相顶替：声明放一格
    /// 不等于允许把它挖了，反过来也一样。
    #[test]
    fn a_declared_placement_is_not_permission_to_mine_the_same_position() {
        let source = RelBlockPos::new(0, 64, 0);
        let floor = source.down(1);
        let head = source.up(1);
        let side_effects = MovementSideEffects::to_air(source, [head]).and_solid(source, floor);

        let at = BlockPos::new(-30, 64, 12);
        assert!(side_effects.allows_place(at, at.down(1)));
        assert!(!side_effects.allows_break(at, at.down(1)));
        assert!(side_effects.allows_break(at, at.up(1)));
        assert!(!side_effects.allows_place(at, at.up(1)));

        assert_eq!(
            side_effects.replacements(at).collect::<Vec<_>>(),
            [
                (at.up(1), ReplaceTarget::Air),
                (at.down(1), ReplaceTarget::Solid),
            ]
        );
    }

    #[test]
    #[should_panic(expected = "a movement declared too many world replacements")]
    fn a_movement_cannot_declare_more_replacements_than_the_inline_capacity() {
        let source = RelBlockPos::new(0, 64, 0);
        let _ = MovementSideEffects::to_air_iter(
            source,
            (0..=MAX_REPLACEMENTS_PER_MOVEMENT).map(|offset| source.up(offset as i32)),
        );
    }
}
