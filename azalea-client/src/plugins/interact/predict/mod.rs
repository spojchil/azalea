//! Predicting the client-side outcome of a right click.
//!
//! Vanilla's `Minecraft.startUseItem` sends one packet per step (use on the
//! targeted entity or block, then use the held item, then the same for the off
//! hand), but whether it moves on to the next step depends on the result of
//! running the block/item/entity logic locally first. Azalea doesn't run that
//! logic, so this module transcribes the client-side branches of every
//! override that can change the flow (`useItemOn`, `useWithoutItem`, `useOn`,
//! `use`, `interact`), with the per-block/item/entity class tables generated
//! from the vanilla jar in [`data`].
//!
//! Only the *client* branch matters here (`level.isClientSide()` is true). The
//! server repeats the real logic for every packet it receives, so a wrong
//! prediction either skips a step vanilla would have taken, or takes a step
//! vanilla would have skipped (which can make the server do two things for one
//! click). Where the client branch depends on state the client doesn't track,
//! the code says so next to the approximation.
//!
//! Entities are handled conservatively for now: classes that override
//! `interact`/`mobInteract` are assumed to consume the click (which matches
//! what azalea did before), and only the shared `Entity`/`Mob` logic is
//! transcribed exactly.

pub mod data;

use std::collections::HashMap;

use azalea_block::{BlockState, BlockTrait, fluid_state::FluidKind};
use azalea_core::{
    aabb::Aabb,
    direction::{Axis, Direction},
    game_type::GameMode,
    hit_result::BlockHitResult,
    position::{BlockPos, Vec3},
};
use azalea_entity::{LookDirection, PlayerAbilities, inventory::Inventory, view_vector};
use azalea_inventory::{
    ItemStack,
    components::{
        self, BannerPatterns, BlocksAttacks, ChargedProjectiles, Consumable, CustomName, DyedColor,
        Equippable, Food, JukeboxPlayable, KineticWeapon, PotionContents, Recipes,
    },
};
use azalea_physics::clip::{BlockShapeType, ClipContext, FluidPickType, clip};
use azalea_protocol::packets::game::s_interact::InteractionHand;
use azalea_registry::{
    builtin::{BlockKind, EntityKind, ItemKind, Potion},
    tags,
};
use azalea_world::World;

use self::data::*;

/// Who swings the arm when an interaction succeeds
/// (`InteractionResult.SwingSource`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Swing {
    /// The client swings its own arm and tells the server (`SUCCESS`).
    Client,
    /// The server swings it (`SUCCESS_SERVER`).
    Server,
    /// Nobody swings (`CONSUME`).
    None,
}

/// Vanilla's `InteractionResult`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InteractionResult {
    Success(Swing),
    Fail,
    Pass,
    TryWithEmptyHand,
}

impl InteractionResult {
    pub const SUCCESS: Self = Self::Success(Swing::Client);
    pub const SUCCESS_SERVER: Self = Self::Success(Swing::Server);
    pub const CONSUME: Self = Self::Success(Swing::None);

    pub fn consumes_action(self) -> bool {
        matches!(self, Self::Success(_))
    }
}

use InteractionResult::{Fail, Pass, TryWithEmptyHand};
const SUCCESS: InteractionResult = InteractionResult::SUCCESS;
const SUCCESS_SERVER: InteractionResult = InteractionResult::SUCCESS_SERVER;
const CONSUME: InteractionResult = InteractionResult::CONSUME;

/// The local player, as far as right-click logic looks at it.
pub struct Actor<'a> {
    pub game_mode: GameMode,
    pub abilities: &'a PlayerAbilities,
    pub permission_level: u8,
    /// `isSecondaryUseActive`, which is `isShiftKeyDown`.
    pub sneaking: bool,
    pub food: u32,
    pub fall_flying: bool,
    pub inventory: &'a Inventory,
    pub eye_position: Vec3,
    pub look_direction: LookDirection,
    pub block_interaction_range: f64,
    /// The player's own hitbox; a placed block may not collide with it.
    pub bounding_box: Aabb,
}

impl Actor<'_> {
    pub fn item(&self, hand: InteractionHand) -> &ItemStack {
        let slot = match hand {
            InteractionHand::MainHand => components::EquipmentSlot::Mainhand,
            InteractionHand::OffHand => components::EquipmentSlot::Offhand,
        };
        self.inventory
            .get_equipment(slot)
            .unwrap_or(&ItemStack::Empty)
    }

    /// `Abilities.mayBuild`: the client derives it from the game mode
    /// (`GameType.updatePlayerAbilities`), the server never sends it.
    fn may_build(&self) -> bool {
        !matches!(self.game_mode, GameMode::Adventure | GameMode::Spectator)
    }

    /// `hasInfiniteMaterials`, which is `abilities.instabuild`.
    fn infinite_materials(&self) -> bool {
        self.abilities.instant_break
    }

    fn can_eat(&self, can_always_eat: bool) -> bool {
        self.abilities.invulnerable || can_always_eat || self.food < 20
    }

    fn can_use_game_master_blocks(&self) -> bool {
        self.abilities.instant_break && self.permission_level >= 2
    }

    /// `Item.getPlayerPOVHitResult`.
    fn pov_hit(&self, world: &World, fluid: FluidPickType) -> BlockHitResult {
        let to =
            self.eye_position + view_vector(self.look_direction) * self.block_interaction_range;
        clip(
            &world.chunks,
            ClipContext {
                from: self.eye_position,
                to,
                block_shape_type: BlockShapeType::Outline,
                fluid_pick_type: fluid,
            },
        )
    }
}

fn kind(item: &ItemStack) -> ItemKind {
    item.kind()
}

fn has<T: components::DataComponentTrait>(item: &ItemStack) -> bool {
    item.as_present()
        .is_some_and(|data| data.get_component::<T>().is_some())
}

fn properties(state: BlockState) -> HashMap<&'static str, &'static str> {
    Box::<dyn BlockTrait>::from(state).property_map()
}

fn property(state: BlockState, name: &str) -> Option<&'static str> {
    properties(state).get(name).copied()
}

fn flag(state: BlockState, name: &str) -> bool {
    property(state, name) == Some("true")
}

fn number(state: BlockState, name: &str) -> i32 {
    property(state, name)
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

fn facing(state: BlockState) -> Option<Direction> {
    Some(match property(state, "facing")? {
        "north" => Direction::North,
        "south" => Direction::South,
        "west" => Direction::West,
        "east" => Direction::East,
        "up" => Direction::Up,
        "down" => Direction::Down,
        _ => return None,
    })
}

fn axis(direction: Direction) -> Axis {
    match direction {
        Direction::Down | Direction::Up => Axis::Y,
        Direction::North | Direction::South => Axis::Z,
        Direction::West | Direction::East => Axis::X,
    }
}

fn same_axis(a: Direction, b: Direction) -> bool {
    matches!(
        (axis(a), axis(b)),
        (Axis::X, Axis::X) | (Axis::Y, Axis::Y) | (Axis::Z, Axis::Z)
    )
}

fn vertical(direction: Direction) -> bool {
    matches!(direction, Direction::Down | Direction::Up)
}

fn block_at(world: &World, pos: BlockPos) -> BlockState {
    world.get_block_state(pos).unwrap_or_default()
}

fn is_air(state: BlockState) -> bool {
    matches!(
        BlockKind::from(state),
        BlockKind::Air | BlockKind::CaveAir | BlockKind::VoidAir
    )
}

/// The `level.getFluidState(pos)` is a water/lava source.
fn is_source(world: &World, pos: BlockPos, fluid: FluidKind) -> bool {
    world
        .get_fluid_state(pos)
        .is_some_and(|f| f.kind == fluid && f.amount == 8)
}

/// `BlockPlaceContext.canPlace` for placing against `hit`.
///
/// Approximation: `canBeReplaced(context)` is taken from the block's default
/// replaceability; vanilla also lets snow layers, vines etc. be replaced
/// conditionally on the item.
fn can_place(world: &World, hit: &BlockHitResult) -> bool {
    let clicked = BlockKind::from(block_at(world, hit.block_pos));
    is_replaceable(clicked)
        || is_replaceable(BlockKind::from(block_at(
            world,
            hit.block_pos.offset_with_direction(hit.direction),
        )))
}

/// `SelectableSlotContainer.getHitSlot`.
fn hit_slot(hit: &BlockHitResult, block_facing: Direction, rows: i32, columns: i32) -> Option<i32> {
    if hit.direction != block_facing {
        return None;
    }
    let base = hit.block_pos.offset_with_direction(hit.direction);
    let relative = hit.location - base.to_vec3_floored();
    let (x, y) = match hit.direction {
        Direction::North => (1.0 - relative.x, relative.y),
        Direction::South => (relative.x, relative.y),
        Direction::West => (relative.z, relative.y),
        Direction::East => (1.0 - relative.z, relative.y),
        Direction::Down | Direction::Up => return None,
    };
    let section = |coordinate: f64, sections: i32| {
        let pixel = coordinate as f32 * 16.0;
        let size = 16.0 / sections as f32;
        ((pixel / size).floor() as i32).clamp(0, sections - 1)
    };
    let row = section(1.0 - y, rows);
    let column = section(x, columns);
    Some(column + row * columns)
}

/// `MultiPlayerGameMode.performUseItemOn`. The `useItemOn` packet is sent
/// regardless of the result.
pub fn use_item_on(
    world: &World,
    actor: &Actor,
    hand: InteractionHand,
    hit: &BlockHitResult,
) -> InteractionResult {
    use_item_on_placing(world, actor, hand, hit).0
}

/// A block that a successful right click is predicted to have placed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Placement {
    pub pos: BlockPos,
    pub block: BlockKind,
}

/// [`use_item_on`], plus where the held block item lands when the click is
/// predicted to place it.
///
/// `BlockItem.place` fails when the block would collide with the player
/// (`Level.isUnobstructed`). Approximations: the block's default state stands
/// in for its placement state, any non-empty collision shape counts as a full
/// cube, only our own hitbox is checked (not other entities), and like
/// [`can_place`] the spot only has to be replaceable, not survivable. The
/// server has the final say.
pub fn use_item_on_placing(
    world: &World,
    actor: &Actor,
    hand: InteractionHand,
    hit: &BlockHitResult,
) -> (InteractionResult, Option<Placement>) {
    let item = actor.item(hand);
    if actor.game_mode == GameMode::Spectator {
        return (CONSUME, None);
    }
    let have_something = !actor.item(InteractionHand::MainHand).is_empty()
        || !actor.item(InteractionHand::OffHand).is_empty();
    let suppress_using_block = actor.sneaking && have_something;
    if !suppress_using_block {
        let state = block_at(world, hit.block_pos);
        let item_use = block_use_item_on_result(world, actor, state, item, hand, hit);
        if item_use.consumes_action() {
            return (item_use, None);
        }
        if item_use == TryWithEmptyHand && hand == InteractionHand::MainHand {
            let used = block_use_without_item_result(actor, state, hit);
            if used.consumes_action() {
                return (used, None);
            }
        }
    }
    // Item cooldowns aren't tracked; vanilla returns PASS here while the item
    // is on cooldown.
    if item.is_empty() {
        return (Pass, None);
    }
    let result = item_use_on_result(world, actor, hand, item, hit);
    let placement = match (result, item_use_on(kind(item))) {
        (InteractionResult::Success(_), ItemUseOn::BlockItem) => item_places(kind(item))
            .and_then(|block| placement_pos(world, hit).map(|pos| Placement { pos, block })),
        _ => None,
    };
    if let Some(placement) = placement
        && obstructs(actor, placement)
    {
        return (Fail, None);
    }
    (result, placement)
}

/// Whether the placed block would collide with the player.
fn obstructs(actor: &Actor, placement: Placement) -> bool {
    use azalea_physics::collision::BlockWithShape;

    if BlockState::from(placement.block).is_collision_shape_empty() {
        return false;
    }
    let min = placement.pos.to_vec3_floored();
    actor.bounding_box.intersects_aabb(&Aabb {
        min,
        max: min + Vec3::new(1.0, 1.0, 1.0),
    })
}

/// `BlockPlaceContext.getClickedPos`: the clicked block itself when it can be
/// replaced, otherwise the one in front of the clicked face.
fn placement_pos(world: &World, hit: &BlockHitResult) -> Option<BlockPos> {
    if is_replaceable(BlockKind::from(block_at(world, hit.block_pos))) {
        return Some(hit.block_pos);
    }
    let next = hit.block_pos.offset_with_direction(hit.direction);
    is_replaceable(BlockKind::from(block_at(world, next))).then_some(next)
}

/// `BlockState.useItemOn`, client branch.
fn block_use_item_on_result(
    world: &World,
    actor: &Actor,
    state: BlockState,
    item: &ItemStack,
    hand: InteractionHand,
    hit: &BlockHitResult,
) -> InteractionResult {
    let block = BlockKind::from(state);
    let item_kind = kind(item);
    match block_use_item_on(block) {
        BlockUseItemOn::Default => TryWithEmptyHand,
        BlockUseItemOn::AbstractCauldronBlock => cauldron(state, item),
        BlockUseItemOn::BeehiveBlock => {
            // Shears only empty the hive on the server.
            if number(state, "honey_level") >= 5 && item_kind == ItemKind::GlassBottle {
                SUCCESS
            } else {
                TryWithEmptyHand
            }
        }
        BlockUseItemOn::CakeBlock => {
            if tags::items::CANDLES.contains(&item_kind)
                && number(state, "bites") == 0
                && item_places(item_kind).is_some_and(|b| tags::blocks::CANDLES.contains(&b))
            {
                SUCCESS
            } else {
                TryWithEmptyHand
            }
        }
        BlockUseItemOn::CampfireBlock => {
            if is_campfire_input(item_kind) {
                CONSUME
            } else {
                TryWithEmptyHand
            }
        }
        BlockUseItemOn::CandleBlock => {
            if item.is_empty() && actor.may_build() && flag(state, "lit") {
                SUCCESS
            } else {
                TryWithEmptyHand
            }
        }
        BlockUseItemOn::CandleCakeBlock => {
            if matches!(item_kind, ItemKind::FlintAndSteel | ItemKind::FireCharge) {
                Pass
            } else if hit.location.y - hit.block_pos.y as f64 > 0.5
                && item.is_empty()
                && flag(state, "lit")
            {
                SUCCESS
            } else {
                TryWithEmptyHand
            }
        }
        BlockUseItemOn::CeilingHangingSignBlock => {
            // Approximation: assumes the sign has no click commands
            // (`canExecuteClickCommands` needs the sign's block entity data).
            if is_hanging_sign_item(item_kind) && hit.direction == Direction::Down {
                Pass
            } else {
                sign_use_item_on(actor, item)
            }
        }
        BlockUseItemOn::WallHangingSignBlock => {
            let editable_side = facing(state).is_some_and(|f| same_axis(f, hit.direction));
            if is_hanging_sign_item(item_kind) && !editable_side {
                Pass
            } else {
                sign_use_item_on(actor, item)
            }
        }
        BlockUseItemOn::SignBlock => sign_use_item_on(actor, item),
        BlockUseItemOn::ChiseledBookShelfBlock => {
            if !tags::items::BOOKSHELF_BOOKS.contains(&item_kind) {
                return TryWithEmptyHand;
            }
            let Some(slot) = facing(state).and_then(|f| hit_slot(hit, f, 2, 3)) else {
                return Pass;
            };
            if flag(state, &format!("slot_{slot}_occupied")) {
                TryWithEmptyHand
            } else {
                SUCCESS
            }
        }
        BlockUseItemOn::ComposterBlock => {
            if number(state, "level") < 8 && is_compostable(item_kind) {
                SUCCESS
            } else {
                TryWithEmptyHand
            }
        }
        BlockUseItemOn::CopperGolemStatueBlock => {
            if tags::items::AXES.contains(&item_kind) {
                Pass
            } else {
                SUCCESS
            }
        }
        BlockUseItemOn::WeatheringCopperGolemStatueBlock => {
            if !tags::items::AXES.contains(&item_kind) {
                if item_kind == ItemKind::Honeycomb {
                    Pass
                } else {
                    SUCCESS
                }
            } else if block == BlockKind::CopperGolemStatue {
                // Approximation: assumes the statue block entity can turn back
                // into a golem.
                SUCCESS
            } else {
                Pass
            }
        }
        BlockUseItemOn::DecoratedPotBlock => SUCCESS,
        BlockUseItemOn::FlowerPotBlock => {
            let pottable = item_places(item_kind).is_some_and(is_pottable);
            if !pottable {
                TryWithEmptyHand
            } else if block != BlockKind::FlowerPot {
                CONSUME
            } else {
                SUCCESS
            }
        }
        BlockUseItemOn::JukeboxBlock => {
            if flag(state, "has_record") {
                TryWithEmptyHand
            } else if has::<JukeboxPlayable>(item) {
                SUCCESS
            } else {
                TryWithEmptyHand
            }
        }
        BlockUseItemOn::LecternBlock => {
            if flag(state, "has_book") {
                TryWithEmptyHand
            } else if tags::items::LECTERN_BOOKS.contains(&item_kind) {
                SUCCESS
            } else if item.is_empty() && hand == InteractionHand::MainHand {
                Pass
            } else {
                TryWithEmptyHand
            }
        }
        BlockUseItemOn::NoteBlock => {
            if tags::items::NOTEBLOCK_TOP_INSTRUMENTS.contains(&item_kind)
                && hit.direction == Direction::Up
            {
                Pass
            } else {
                TryWithEmptyHand
            }
        }
        BlockUseItemOn::PumpkinBlock => {
            if item_kind == ItemKind::Shears {
                SUCCESS
            } else {
                TryWithEmptyHand
            }
        }
        BlockUseItemOn::RedStoneOreBlock => {
            if item_places(item_kind).is_some() && can_place(world, hit) {
                Pass
            } else {
                SUCCESS
            }
        }
        BlockUseItemOn::RespawnAnchorBlock => {
            let chargeable = number(state, "charges") < 4;
            if item_kind == ItemKind::Glowstone && chargeable {
                SUCCESS
            } else if hand == InteractionHand::MainHand
                && kind(actor.item(InteractionHand::OffHand)) == ItemKind::Glowstone
                && chargeable
            {
                Pass
            } else {
                TryWithEmptyHand
            }
        }
        BlockUseItemOn::ShelfBlock => {
            if hand == InteractionHand::OffHand {
                return Pass;
            }
            if facing(state).and_then(|f| hit_slot(hit, f, 1, 3)).is_none() {
                return Pass;
            }
            if actor.item(InteractionHand::MainHand).is_empty() {
                Pass
            } else {
                SUCCESS
            }
        }
        BlockUseItemOn::SweetBerryBushBlock => {
            if number(state, "age") != 3 && item_kind == ItemKind::BoneMeal {
                Pass
            } else {
                TryWithEmptyHand
            }
        }
        BlockUseItemOn::TntBlock => {
            if matches!(item_kind, ItemKind::FlintAndSteel | ItemKind::FireCharge) {
                SUCCESS
            } else {
                TryWithEmptyHand
            }
        }
        BlockUseItemOn::VaultBlock => {
            if !item.is_empty() && property(state, "vault_state") == Some("active") {
                SUCCESS_SERVER
            } else {
                TryWithEmptyHand
            }
        }
    }
}

/// `SignBlock.useItemOn`, client branch.
///
/// Approximation: assumes the sign isn't waxed (that's block entity data).
fn sign_use_item_on(actor: &Actor, item: &ItemStack) -> InteractionResult {
    let applicator = matches!(
        kind(item),
        ItemKind::GlowInkSac | ItemKind::InkSac | ItemKind::Honeycomb
    ) || has::<components::Dye>(item);
    if applicator && actor.may_build() {
        SUCCESS
    } else {
        CONSUME
    }
}

/// `AbstractCauldronBlock.useItemOn` through `CauldronInteractions`.
fn cauldron(state: BlockState, item: &ItemStack) -> InteractionResult {
    let block = BlockKind::from(state);
    let item_kind = kind(item);
    let level = number(state, "level");
    let water_potion = || {
        item.as_present()
            .and_then(|data| data.get_component::<PotionContents>())
            .is_some_and(|p| p.potion == Some(Potion::Water))
    };
    // `addDefaultInteractions`: pouring a filled bucket in.
    if matches!(
        item_kind,
        ItemKind::WaterBucket | ItemKind::LavaBucket | ItemKind::PowderSnowBucket
    ) {
        // Lava and powder snow under water return CONSUME instead; both consume.
        return SUCCESS;
    }
    match block {
        BlockKind::Cauldron => {
            if item_kind == ItemKind::Potion && water_potion() {
                SUCCESS
            } else {
                TryWithEmptyHand
            }
        }
        BlockKind::WaterCauldron => {
            // Tags are checked before items.
            if tags::items::CAULDRON_CAN_REMOVE_DYE.contains(&item_kind) {
                return if has::<DyedColor>(item) {
                    SUCCESS
                } else {
                    TryWithEmptyHand
                };
            }
            match item_kind {
                ItemKind::Bucket if level == 3 => SUCCESS,
                ItemKind::GlassBottle => SUCCESS,
                ItemKind::Potion if level != 3 && water_potion() => SUCCESS,
                k if is_banner_item(k) => {
                    let layers = item
                        .as_present()
                        .and_then(|d| d.get_component::<BannerPatterns>())
                        .is_some_and(|p| !p.patterns.is_empty());
                    if layers { SUCCESS } else { TryWithEmptyHand }
                }
                k if is_dyed_shulker_box_item(k) => SUCCESS,
                _ => TryWithEmptyHand,
            }
        }
        BlockKind::LavaCauldron => {
            if item_kind == ItemKind::Bucket {
                SUCCESS
            } else {
                TryWithEmptyHand
            }
        }
        BlockKind::PowderSnowCauldron => {
            if item_kind == ItemKind::Bucket && level == 3 {
                SUCCESS
            } else {
                TryWithEmptyHand
            }
        }
        _ => TryWithEmptyHand,
    }
}

fn is_banner_item(kind: ItemKind) -> bool {
    use ItemKind::*;
    matches!(
        kind,
        WhiteBanner
            | GrayBanner
            | BlackBanner
            | BlueBanner
            | BrownBanner
            | CyanBanner
            | GreenBanner
            | LightBlueBanner
            | LightGrayBanner
            | LimeBanner
            | MagentaBanner
            | OrangeBanner
            | PinkBanner
            | PurpleBanner
            | RedBanner
            | YellowBanner
    )
}

fn is_dyed_shulker_box_item(kind: ItemKind) -> bool {
    use ItemKind::*;
    matches!(
        kind,
        WhiteShulkerBox
            | GrayShulkerBox
            | BlackShulkerBox
            | BlueShulkerBox
            | BrownShulkerBox
            | CyanShulkerBox
            | GreenShulkerBox
            | LightBlueShulkerBox
            | LightGrayShulkerBox
            | LimeShulkerBox
            | MagentaShulkerBox
            | OrangeShulkerBox
            | PinkShulkerBox
            | PurpleShulkerBox
            | RedShulkerBox
            | YellowShulkerBox
    )
}

/// `BlockState.useWithoutItem`, client branch.
fn block_use_without_item_result(
    actor: &Actor,
    state: BlockState,
    hit: &BlockHitResult,
) -> InteractionResult {
    let block = BlockKind::from(state);
    let if_ = |condition: bool, result: InteractionResult| if condition { result } else { Pass };
    match block_use_without_item(block) {
        BlockUseWithoutItem::Default => Pass,
        // Container blocks open their menu on the server and succeed either way.
        BlockUseWithoutItem::AbstractFurnaceBlock
        | BlockUseWithoutItem::AnvilBlock
        | BlockUseWithoutItem::BarrelBlock
        | BlockUseWithoutItem::BeaconBlock
        | BlockUseWithoutItem::BrewingStandBlock
        | BlockUseWithoutItem::CartographyTableBlock
        | BlockUseWithoutItem::ChestBlock
        | BlockUseWithoutItem::CrafterBlock
        | BlockUseWithoutItem::CraftingTableBlock
        | BlockUseWithoutItem::DecoratedPotBlock
        | BlockUseWithoutItem::DispenserBlock
        | BlockUseWithoutItem::DragonEggBlock
        | BlockUseWithoutItem::EnchantingTableBlock
        | BlockUseWithoutItem::EnderChestBlock
        | BlockUseWithoutItem::FenceGateBlock
        | BlockUseWithoutItem::GrindstoneBlock
        | BlockUseWithoutItem::HopperBlock
        | BlockUseWithoutItem::LeverBlock
        | BlockUseWithoutItem::LoomBlock
        | BlockUseWithoutItem::NoteBlock
        | BlockUseWithoutItem::ShulkerBoxBlock
        | BlockUseWithoutItem::SmithingTableBlock
        | BlockUseWithoutItem::StonecutterBlock => SUCCESS,
        BlockUseWithoutItem::BedBlock => SUCCESS_SERVER,
        BlockUseWithoutItem::SignBlock | BlockUseWithoutItem::LightBlock => CONSUME,
        BlockUseWithoutItem::FenceBlock | BlockUseWithoutItem::MovingPistonBlock => Pass,
        BlockUseWithoutItem::FlowerPotBlock => {
            if block == BlockKind::FlowerPot {
                CONSUME
            } else {
                SUCCESS
            }
        }
        BlockUseWithoutItem::DoorBlock | BlockUseWithoutItem::TrapDoorBlock => {
            if_(opens_by_hand(block), SUCCESS)
        }
        BlockUseWithoutItem::CandleCakeBlock => if_(actor.can_eat(false), SUCCESS),
        BlockUseWithoutItem::CakeBlock => {
            if actor.can_eat(false) {
                SUCCESS
            } else if actor.item(InteractionHand::MainHand).is_empty() {
                CONSUME
            } else {
                Pass
            }
        }
        BlockUseWithoutItem::ButtonBlock => {
            if flag(state, "powered") {
                CONSUME
            } else {
                SUCCESS
            }
        }
        BlockUseWithoutItem::CommandBlock
        | BlockUseWithoutItem::JigsawBlock
        | BlockUseWithoutItem::StructureBlock
        | BlockUseWithoutItem::TestBlock
        | BlockUseWithoutItem::TestInstanceBlock => {
            if_(actor.can_use_game_master_blocks(), SUCCESS)
        }
        BlockUseWithoutItem::BellBlock => if_(bell_proper_hit(state, hit), SUCCESS),
        BlockUseWithoutItem::CaveVinesBlock | BlockUseWithoutItem::CaveVinesPlantBlock => {
            if_(flag(state, "berries"), SUCCESS)
        }
        BlockUseWithoutItem::ChiseledBookShelfBlock => {
            let Some(slot) = facing(state).and_then(|f| hit_slot(hit, f, 2, 3)) else {
                return Pass;
            };
            if flag(state, &format!("slot_{slot}_occupied")) {
                SUCCESS
            } else {
                CONSUME
            }
        }
        BlockUseWithoutItem::ComparatorBlock
        | BlockUseWithoutItem::DaylightDetectorBlock
        | BlockUseWithoutItem::RepeaterBlock => if_(actor.may_build(), SUCCESS),
        BlockUseWithoutItem::ComposterBlock => if_(number(state, "level") == 8, SUCCESS),
        BlockUseWithoutItem::JukeboxBlock => if_(flag(state, "has_record"), SUCCESS),
        BlockUseWithoutItem::LecternBlock => {
            if flag(state, "has_book") {
                SUCCESS
            } else {
                CONSUME
            }
        }
        BlockUseWithoutItem::RedStoneWireBlock => {
            // Approximation: a dot or cross always toggles. Vanilla passes when
            // the recomputed connections give back the same state.
            let sides = ["north", "south", "east", "west"]
                .map(|side| property(state, side).is_some_and(|v| v != "none"));
            let cross = sides.iter().all(|s| *s);
            let dot = sides.iter().all(|s| !*s);
            if_(actor.may_build() && (cross || dot), SUCCESS)
        }
        BlockUseWithoutItem::RespawnAnchorBlock => {
            if number(state, "charges") == 0 {
                Pass
            } else {
                CONSUME
            }
        }
        BlockUseWithoutItem::SweetBerryBushBlock => if_(number(state, "age") > 1, SUCCESS),
    }
}

/// `BellBlock.isProperHit`.
fn bell_proper_hit(state: BlockState, hit: &BlockHitResult) -> bool {
    let click_y = hit.location.y - hit.block_pos.y as f64;
    if vertical(hit.direction) || click_y > 0.8124 {
        return false;
    }
    let Some(facing) = facing(state) else {
        return false;
    };
    match property(state, "attachment") {
        Some("floor") => same_axis(facing, hit.direction),
        Some("single_wall" | "double_wall") => !same_axis(facing, hit.direction),
        Some("ceiling") => true,
        _ => false,
    }
}

/// `ItemStack.useOn`, client branch. Only `Pass` lets the click carry on; a
/// success or a failure both end it.
fn item_use_on_result(
    world: &World,
    actor: &Actor,
    hand: InteractionHand,
    item: &ItemStack,
    hit: &BlockHitResult,
) -> InteractionResult {
    let item_kind = kind(item);
    let pos = hit.block_pos;
    let state = block_at(world, pos);
    let block = BlockKind::from(state);
    // `ItemStack.useOn`: without `mayBuild` (adventure mode) only items whose
    // `can_place_on` matches the block may be used; that predicate isn't
    // checked here (approximation: always PASS).
    if !actor.may_build() {
        return Pass;
    }
    match item_use_on(item_kind) {
        ItemUseOn::Default | ItemUseOn::PlaceOnWaterBlockItem => Pass,
        ItemUseOn::BlockItem | ItemUseOn::SolidBucketItem => {
            // Approximation: placement succeeds whenever the target spot is
            // replaceable (vanilla also needs a valid placement state that can
            // survive there). Either way the click ends here; only the arm
            // swing depends on it.
            if can_place(world, hit) {
                SUCCESS
            } else if has::<Consumable>(item) {
                item_use_default(actor, item)
            } else {
                Fail
            }
        }
        ItemUseOn::SpawnEggItem | ItemUseOn::DebugStickItem => SUCCESS,
        ItemUseOn::BrushItem => CONSUME,
        ItemUseOn::LeadItem => Pass,
        ItemUseOn::AxeItem => {
            let blocking_intent = hand == InteractionHand::MainHand
                && has::<BlocksAttacks>(actor.item(InteractionHand::OffHand))
                && !actor.sneaking;
            if blocking_intent {
                Pass
            } else if is_strippable(block) || is_scrapable(block) || is_waxed(block) {
                SUCCESS
            } else {
                Pass
            }
        }
        ItemUseOn::HoeItem => {
            let air_above = hit.direction != Direction::Down && is_air(block_at(world, pos.up(1)));
            let tillable = match block {
                BlockKind::GrassBlock
                | BlockKind::DirtPath
                | BlockKind::Dirt
                | BlockKind::CoarseDirt => air_above,
                BlockKind::RootedDirt => true,
                _ => false,
            };
            if tillable { SUCCESS } else { Pass }
        }
        ItemUseOn::ShovelItem => {
            if hit.direction == Direction::Down {
                Pass
            } else if (is_flattenable(block) && is_air(block_at(world, pos.up(1))))
                || (tags::blocks::CAMPFIRES.contains(&block) && flag(state, "lit"))
            {
                SUCCESS
            } else {
                Pass
            }
        }
        ItemUseOn::MinecartItem => {
            if tags::blocks::RAILS.contains(&block) {
                SUCCESS
            } else {
                Fail
            }
        }
        ItemUseOn::HangingEntityItem => {
            // Vanilla also fails when the frame/painting can't survive there;
            // either way the click ends.
            if vertical(hit.direction) {
                Fail
            } else {
                SUCCESS
            }
        }
        ItemUseOn::ArmorStandItem => {
            if hit.direction == Direction::Down {
                Fail
            } else {
                SUCCESS
            }
        }
        ItemUseOn::PotionItem => {
            let water = item
                .as_present()
                .and_then(|d| d.get_component::<PotionContents>())
                .is_some_and(|p| p.potion == Some(Potion::Water));
            if hit.direction != Direction::Down
                && tags::blocks::CONVERTIBLE_TO_MUD.contains(&block)
                && water
            {
                SUCCESS
            } else {
                Pass
            }
        }
        ItemUseOn::BoneMealItem => {
            // Approximation: vanilla returns PASS on the client whenever the
            // block is a valid bone meal target (`growCrop`), which isn't
            // modelled; this only checks the underwater-plant branch.
            let relative = pos.offset_with_direction(hit.direction);
            let sturdy = !is_replaceable(block) && !is_air(state);
            if sturdy
                && BlockKind::from(block_at(world, relative)) == BlockKind::Water
                && is_source(world, relative, FluidKind::Water)
            {
                SUCCESS
            } else {
                Pass
            }
        }
        ItemUseOn::CompassItem => {
            if block == BlockKind::Lodestone {
                SUCCESS
            } else {
                Pass
            }
        }
        ItemUseOn::MapItem => {
            if tags::blocks::BANNERS.contains(&block) {
                SUCCESS
            } else {
                Pass
            }
        }
        ItemUseOn::EndCrystalItem => {
            // Succeeds or fails; both end the click.
            if matches!(block, BlockKind::Obsidian | BlockKind::Bedrock)
                && is_air(block_at(world, pos.up(1)))
            {
                SUCCESS
            } else {
                Fail
            }
        }
        ItemUseOn::EnderEyeItem => {
            if block == BlockKind::EndPortalFrame && !flag(state, "eye") {
                SUCCESS
            } else {
                Pass
            }
        }
        // Light the block or place fire; both a success and a failure end the click.
        ItemUseOn::FireChargeItem | ItemUseOn::FlintAndSteelItem => SUCCESS,
        ItemUseOn::FireworkRocketItem => {
            if actor.fall_flying {
                Pass
            } else {
                SUCCESS
            }
        }
        ItemUseOn::HoneycombItem => {
            if is_waxable(block) {
                SUCCESS
            } else {
                Pass
            }
        }
        ItemUseOn::ShearsItem => {
            let max_age = match block {
                BlockKind::Kelp
                | BlockKind::WeepingVines
                | BlockKind::TwistingVines
                | BlockKind::CaveVines => Some(25),
                _ => None,
            };
            if max_age.is_some_and(|max| number(state, "age") < max) {
                SUCCESS
            } else {
                Pass
            }
        }
    }
}

/// `ItemStack.use`, client branch. The click ends only on a success; a failure
/// still lets the other hand try.
pub fn use_item(world: &World, actor: &Actor, hand: InteractionHand) -> InteractionResult {
    let item = actor.item(hand);
    if actor.game_mode == GameMode::Spectator {
        return Pass;
    }
    // Item cooldowns aren't tracked; vanilla returns PASS while on cooldown.
    let item_kind = kind(item);
    match item_use(item_kind) {
        ItemUse::Default => item_use_default(actor, item),
        ItemUse::BundleItem
        | ItemUse::EggItem
        | ItemUse::EmptyMapItem
        | ItemUse::EnderpearlItem
        | ItemUse::ExperienceBottleItem
        | ItemUse::FishingRodItem
        | ItemUse::LingeringPotionItem
        | ItemUse::SnowballItem
        | ItemUse::SplashPotionItem
        | ItemUse::WindChargeItem
        | ItemUse::WritableBookItem
        | ItemUse::WrittenBookItem => SUCCESS,
        ItemUse::SpyglassItem => CONSUME,
        ItemUse::FoodOnAStickItem => Pass,
        ItemUse::SpawnEggItem => {
            let hit = actor.pov_hit(world, FluidPickType::SourceOnly);
            if hit.miss { Pass } else { SUCCESS }
        }
        ItemUse::BoatItem => {
            // Approximation: doesn't check for entities around the eye or
            // whether the boat collides where it lands.
            let hit = actor.pov_hit(world, FluidPickType::Any);
            if hit.miss { Pass } else { SUCCESS }
        }
        ItemUse::BucketItem => {
            let empty = item_kind == ItemKind::Bucket;
            let fluid = if empty {
                FluidPickType::SourceOnly
            } else {
                FluidPickType::None
            };
            let hit = actor.pov_hit(world, fluid);
            if hit.miss {
                return Pass;
            }
            if empty {
                let state = block_at(world, hit.block_pos);
                let pickup = is_source(world, hit.block_pos, FluidKind::Water)
                    || is_source(world, hit.block_pos, FluidKind::Lava)
                    || BlockKind::from(state) == BlockKind::PowderSnow;
                if pickup { SUCCESS } else { Fail }
            } else {
                // Approximation: emptying is assumed to work.
                SUCCESS
            }
        }
        ItemUse::PlaceOnWaterBlockItem => {
            let hit = actor.pov_hit(world, FluidPickType::SourceOnly);
            let above = hit.block_pos.up(1);
            if !hit.miss && is_replaceable(BlockKind::from(block_at(world, above))) {
                SUCCESS
            } else {
                Fail
            }
        }
        ItemUse::BottleItem => {
            // The dragon's breath branch (an ender dragon's cloud nearby) isn't
            // modelled.
            let hit = actor.pov_hit(world, FluidPickType::SourceOnly);
            if !hit.miss
                && world
                    .get_fluid_state(hit.block_pos)
                    .is_some_and(|f| f.kind == FluidKind::Water)
            {
                SUCCESS
            } else {
                Pass
            }
        }
        ItemUse::BowItem => {
            if actor.infinite_materials() || has_projectile(actor, false) {
                CONSUME
            } else {
                Fail
            }
        }
        ItemUse::CrossbowItem => {
            let charged = item
                .as_present()
                .and_then(|d| d.get_component::<ChargedProjectiles>())
                .is_some_and(|c| !c.items.is_empty());
            if charged || actor.infinite_materials() || has_projectile(actor, true) {
                CONSUME
            } else {
                Fail
            }
        }
        ItemUse::EnderEyeItem => {
            let hit = actor.pov_hit(world, FluidPickType::None);
            if !hit.miss
                && BlockKind::from(block_at(world, hit.block_pos)) == BlockKind::EndPortalFrame
            {
                Pass
            } else {
                SUCCESS_SERVER
            }
        }
        ItemUse::FireworkRocketItem => {
            if actor.fall_flying {
                SUCCESS
            } else {
                Pass
            }
        }
        ItemUse::InstrumentItem => {
            if has::<components::Instrument>(item) {
                CONSUME
            } else {
                Fail
            }
        }
        ItemUse::KnowledgeBookItem => {
            let recipes = item
                .as_present()
                .and_then(|d| d.get_component::<Recipes>())
                .is_some_and(|r| !r.recipes.is_empty());
            if recipes { SUCCESS } else { Fail }
        }
        ItemUse::TridentItem => {
            // Approximation: riptide outside water/rain fails in vanilla; the
            // enchantment isn't checked here.
            let breaks = item.as_present().is_some_and(|d| {
                let max = d.get_component::<components::MaxDamage>().map(|m| m.amount);
                let damage = d.get_component::<components::Damage>().map(|m| m.amount);
                matches!((max, damage), (Some(max), Some(damage)) if damage + 1 >= max)
            });
            if breaks { Fail } else { CONSUME }
        }
    }
}

/// `Item.use` in the base class: consumables, swappable equipment, shields,
/// spears.
fn item_use_default(actor: &Actor, item: &ItemStack) -> InteractionResult {
    let Some(data) = item.as_present() else {
        return Pass;
    };
    if data.get_component::<Consumable>().is_some() {
        let can_always_eat = data.get_component::<Food>().map(|f| f.can_always_eat);
        let can_consume = match can_always_eat {
            Some(always) => actor.can_eat(always),
            None => true,
        };
        return if can_consume { CONSUME } else { Fail };
    }
    if let Some(equippable) = data.get_component::<Equippable>()
        && equippable.swappable
    {
        let allowed = equippable
            .allowed_entities
            .as_ref()
            .is_none_or(|set| set.contains(EntityKind::Player));
        if !allowed {
            return Pass;
        }
        // Approximation: the curse of binding (PREVENT_ARMOR_CHANGE) isn't
        // checked.
        let equipped = actor.inventory.get_equipment(equippable.slot);
        let same = match (equipped.and_then(|e| e.as_present()), item.as_present()) {
            (Some(a), Some(b)) => a.is_same_item_and_components(b),
            _ => false,
        };
        return if same { Fail } else { SUCCESS };
    }
    if data.get_component::<BlocksAttacks>().is_some()
        || data.get_component::<KineticWeapon>().is_some()
    {
        return CONSUME;
    }
    Pass
}

/// `Player.getProjectile` is non-empty: an arrow in a hand (or a firework
/// rocket for a crossbow), or an arrow anywhere in the inventory.
fn has_projectile(actor: &Actor, crossbow: bool) -> bool {
    let held_ok = |kind: ItemKind| {
        tags::items::ARROWS.contains(&kind) || (crossbow && kind == ItemKind::FireworkRocket)
    };
    if held_ok(kind(actor.item(InteractionHand::OffHand)))
        || held_ok(kind(actor.item(InteractionHand::MainHand)))
    {
        return true;
    }
    let player = actor.inventory.inventory_menu.as_player();
    player
        .inventory
        .iter()
        .chain(player.armor.iter())
        .chain(std::iter::once(&player.offhand))
        .any(|item| tags::items::ARROWS.contains(&kind(item)))
}

/// The targeted entity, as far as right-click logic looks at it.
pub struct Target {
    pub kind: EntityKind,
    pub alive: bool,
}

/// `Player.interactOn`, client branch. The interact packet is sent regardless.
pub fn interact(actor: &Actor, hand: InteractionHand, target: &Target) -> InteractionResult {
    if actor.game_mode == GameMode::Spectator {
        return Pass;
    }
    let item = actor.item(hand);
    let result = entity_interact_result(item, target);
    if result.consumes_action() {
        return result;
    }
    let living =
        is_mob(target.kind) || matches!(target.kind, EntityKind::Player | EntityKind::ArmorStand);
    if !item.is_empty() && living {
        let used = match item_interact_living_entity(kind(item)) {
            ItemInteractLivingEntity::Default => Pass,
            // Approximation: assumes the sheep isn't sheared and is another color.
            ItemInteractLivingEntity::DyeItem => {
                if target.kind == EntityKind::Sheep {
                    SUCCESS
                } else {
                    Pass
                }
            }
            ItemInteractLivingEntity::NameTagItem => {
                if has::<CustomName>(item) && target.kind != EntityKind::Player {
                    SUCCESS
                } else {
                    Pass
                }
            }
        };
        if used.consumes_action() {
            return used;
        }
    }
    Pass
}

/// `Entity.interact` (and `Mob.interact`), client branch.
fn entity_interact_result(item: &ItemStack, target: &Target) -> InteractionResult {
    // Entities with their own `interact` are assumed to consume the click
    // (stage one: not transcribed yet).
    if entity_interact(target.kind) != EntityInteract::Default {
        return CONSUME;
    }
    let item_kind = kind(item);
    if is_mob(target.kind) {
        if !target.alive {
            return Pass;
        }
        // `Mob.checkAndHandleImportantInteractions`.
        if item_kind == ItemKind::NameTag && has::<CustomName>(item) {
            return SUCCESS;
        }
        if is_spawn_egg(item_kind) {
            return SUCCESS_SERVER;
        }
    }
    // `Entity.interact`. Approximation: leash state (who holds the leash, leash
    // knots to shear) isn't tracked, so only the "attach a lead" branch is
    // modelled; every mob is `Leashable`.
    if is_mob(target.kind) && target.alive && item_kind == ItemKind::Lead {
        return CONSUME;
    }
    if is_mob(target.kind) {
        // `mobInteract`: overriding classes are assumed to consume the click
        // (stage one: not transcribed yet).
        if entity_mob_interact(target.kind) != EntityMobInteract::Default {
            return CONSUME;
        }
    }
    Pass
}
