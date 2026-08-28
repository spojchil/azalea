use std::f32::consts::SQRT_2;

use azalea_block::{BlockState, properties};
use azalea_client::{SprintDirection, WalkDirection};
use azalea_core::{
    direction::CardinalDirection,
    position::{BlockPos, Vec3},
};

use super::{
    Edge, ExecuteCtx, IsReachedCtx, MoveData, MovementSideEffects, MovesCtx, default_is_reached,
};
use crate::pathfinder::{
    astar, costs::*, moves::BARITONE_COMPAT, player_pos_to_block_pos, positions::RelBlockPos,
};

pub fn basic_move(ctx: &mut MovesCtx, node: RelBlockPos) {
    forward_move(ctx, node);
    ascend_move(ctx, node);
    descend_move(ctx, node);
    diagonal_move(ctx, node);
    downward_move(ctx, node);
    pillar_move(ctx, node);
}

fn forward_move(ctx: &mut MovesCtx, pos: RelBlockPos) {
    let mut base_cost = SPRINT_ONE_BLOCK_COST;
    // it's for us cheaper to have the water cost be applied when leaving the water
    // rather than when entering
    let currently_in_water = ctx.world.is_block_water(pos);
    if currently_in_water {
        if BARITONE_COMPAT {
            base_cost = WALK_ONE_BLOCK_COST;
        } else {
            base_cost = WALK_ONE_IN_WATER_COST;
        }
    }

    for dir in CardinalDirection::iter() {
        let offset = RelBlockPos::new(dir.x(), 0, dir.z());

        let new_pos = pos + offset;

        let break_cost = if currently_in_water {
            let dest_in_water = ctx.world.is_block_water(new_pos);
            if !dest_in_water {
                continue;
            }

            ctx.world
                .cost_for_breaking_block(new_pos.up(1), ctx.mining_cache)
        } else {
            ctx.world.cost_for_standing(new_pos, ctx.mining_cache)
        };
        if break_cost == f32::INFINITY {
            continue;
        }

        let cost = base_cost + break_cost;

        ctx.edges.push(Edge {
            movement: astar::Movement {
                target: new_pos,
                data: MoveData::new(&execute_forward_move, &default_is_reached)
                    .with_side_effects(MovementSideEffects::to_air(pos, [new_pos, new_pos.up(1)])),
            },
            cost,
        })
    }
}

fn execute_forward_move(mut ctx: ExecuteCtx) {
    let center = ctx.target.center();
    ctx.jump_if_in_water();

    if ctx.mine_while_at_start(ctx.target.up(1)) {
        return;
    }
    if ctx.mine_while_at_start(ctx.target) {
        return;
    }

    ctx.look_at(center);
    ctx.sprint(SprintDirection::Forward);
}

fn ascend_move(ctx: &mut MovesCtx, pos: RelBlockPos) {
    // the block we're standing on must be solid (so we don't try to ascend from a
    // bottom slab to a normal block in a way that's not possible)

    let is_unusual_shape = !ctx.world.is_block_solid(pos.down(1));
    let mut stair_facing = None;

    if is_unusual_shape && !ctx.world.is_block_water(pos) {
        // this is potentially expensive but it's rare enough that it shouldn't matter
        // much
        let block_below = ctx.world.get_block_state(pos.down(1));

        let Some(found_stair_facing) = validate_stair_and_get_facing(block_below) else {
            // return if it's not a stair or it's not facing the right way (like, if it's
            // upside down or something)
            return;
        };

        stair_facing = Some(found_stair_facing);
    }

    let break_cost_1 = ctx
        .world
        .cost_for_breaking_block(pos.up(2), ctx.mining_cache);
    if break_cost_1 == f32::INFINITY {
        return;
    }

    let base_cost =
        f32::max(WALK_ONE_BLOCK_COST, *JUMP_ONE_BLOCK_COST) + JUMP_PENALTY + break_cost_1;

    for dir in CardinalDirection::iter() {
        if let Some(stair_facing) = stair_facing {
            let expected_stair_facing = cardinal_direction_to_facing_property(dir);
            if stair_facing != expected_stair_facing {
                continue;
            }
        }

        let offset = RelBlockPos::new(dir.x(), 1, dir.z());

        let break_cost_2 = ctx.world.cost_for_standing(pos + offset, ctx.mining_cache);
        if break_cost_2 == f32::INFINITY {
            continue;
        }

        let cost = base_cost + break_cost_2;

        ctx.edges.push(Edge {
            movement: astar::Movement {
                target: pos + offset,
                data: MoveData::new(&execute_ascend_move, &ascend_is_reached).with_side_effects(
                    MovementSideEffects::to_air(
                        pos,
                        [pos.up(2), pos + offset, (pos + offset).up(1)],
                    ),
                ),
            },
            cost,
        })
    }
}
fn execute_ascend_move(mut ctx: ExecuteCtx) {
    let ExecuteCtx {
        target,
        start,
        position,
        physics,
        ..
    } = ctx;

    ctx.jump_if_in_water();

    if ctx.mine_while_at_start(start.up(2)) {
        return;
    }
    if ctx.mine_while_at_start(target) {
        return;
    }
    if ctx.mine_while_at_start(target.up(1)) {
        return;
    }

    let target_center = target.center();

    ctx.look_at(target_center);
    ctx.walk(WalkDirection::Forward);

    // these checks are to make sure we don't fall if our velocity is too high in
    // the wrong direction

    let x_axis = target.x - start.x; // -1, 0, or 1
    let z_axis = target.z - start.z; // -1, 0, or 1

    let x_axis_abs = x_axis.abs(); // either 0 or 1
    let z_axis_abs = z_axis.abs(); // either 0 or 1

    let flat_distance_to_next = x_axis_abs as f64 * (target_center.x - position.x)
        + z_axis_abs as f64 * (target_center.z - position.z);
    let side_distance = z_axis_abs as f64 * (target_center.x - position.x).abs()
        + x_axis_abs as f64 * (target_center.z - position.z).abs();

    let lateral_motion =
        x_axis_abs as f64 * physics.velocity.z + z_axis_abs as f64 * physics.velocity.x;
    if lateral_motion.abs() > 0.1 {
        return;
    }

    if flat_distance_to_next > 1.2 || side_distance > 0.2 {
        return;
    }

    // if the target block is a stair that's facing in the direction we're going, we
    // shouldn't jump
    let block_below_target = ctx.get_block_state(target.down(1));
    if let Some(stair_facing) = validate_stair_and_get_facing(block_below_target) {
        let expected_stair_facing = match (x_axis, z_axis) {
            (0, 1) => Some(properties::FacingCardinal::North),
            (1, 0) => Some(properties::FacingCardinal::East),
            (0, -1) => Some(properties::FacingCardinal::South),
            (-1, 0) => Some(properties::FacingCardinal::West),
            _ => None,
        };
        if let Some(expected_stair_facing) = expected_stair_facing
            && stair_facing == expected_stair_facing
        {
            return;
        }
    }

    if player_pos_to_block_pos(position) == start {
        // only jump if the target is more than 0.5 blocks above us
        if target.y as f64 - position.y > 0.5 {
            ctx.jump();
        }
    }
}
#[must_use]
pub fn ascend_is_reached(
    IsReachedCtx {
        position, target, ..
    }: IsReachedCtx,
) -> bool {
    BlockPos::from(position) == target || BlockPos::from(position) == target.down(1)
}

fn validate_stair_and_get_facing(block_state: BlockState) -> Option<properties::FacingCardinal> {
    let top_bottom = block_state.property::<properties::TopBottom>();
    if top_bottom != Some(properties::TopBottom::Bottom) {
        return None;
    }

    block_state.property::<properties::FacingCardinal>()
}
fn cardinal_direction_to_facing_property(dir: CardinalDirection) -> properties::FacingCardinal {
    match dir {
        CardinalDirection::North => properties::FacingCardinal::North,
        CardinalDirection::East => properties::FacingCardinal::East,
        CardinalDirection::South => properties::FacingCardinal::South,
        CardinalDirection::West => properties::FacingCardinal::West,
    }
}

fn descend_move(ctx: &mut MovesCtx, pos: RelBlockPos) {
    for dir in CardinalDirection::iter() {
        let dir_delta = RelBlockPos::new(dir.x(), 0, dir.z());
        let new_horizontal_position = pos + dir_delta;

        let break_cost_1 = ctx
            .world
            .cost_for_passing(new_horizontal_position, ctx.mining_cache);
        if break_cost_1 == f32::INFINITY {
            continue;
        }

        let mut fall_distance = ctx.world.fall_distance(new_horizontal_position);
        if fall_distance > 3 {
            continue;
        }

        let mut into_water = false;
        if fall_distance == 0 {
            if ctx.world.is_block_water(new_horizontal_position.down(1)) {
                fall_distance = 1;
                into_water = true;
            } else {
                continue;
            }
        }

        let new_position = new_horizontal_position.down(fall_distance as i32);

        // only mine if we're descending 1 block
        let mut break_cost_2;
        if into_water {
            break_cost_2 = ctx
                .world
                .cost_for_breaking_block(new_position.up(1), ctx.mining_cache);
            if break_cost_2 == f32::INFINITY {
                continue;
            }
            break_cost_2 += ENTER_WATER_PENALTY;
        } else if fall_distance == 1 {
            break_cost_2 = ctx.world.cost_for_standing(new_position, ctx.mining_cache);
            if break_cost_2 == f32::INFINITY {
                continue;
            }
        } else {
            // check whether we can stand on the target position
            if !ctx.world.is_standable(new_position) {
                continue;
            }
            break_cost_2 = 0.;
        }

        if BARITONE_COMPAT && fall_distance > 1 {
            fall_distance += 1;
        }

        let cost = WALK_OFF_BLOCK_COST
            + f32::max(
                *FALL_N_BLOCKS_COST
                    .get(fall_distance as usize)
                    .expect("already checked bounds on fall distance"),
                CENTER_AFTER_FALL_COST,
            )
            + break_cost_1
            + break_cost_2;

        ctx.edges.push(Edge {
            movement: astar::Movement {
                target: new_position,
                data: MoveData::new(&execute_descend_move, &descend_is_reached).with_side_effects(
                    MovementSideEffects::to_air_iter(
                        pos,
                        (0..=(pos.y - new_position.y + 1)).map(|offset| new_position.up(offset)),
                    ),
                ),
            },
            cost,
        })
    }
}
pub fn execute_descend_move(mut ctx: ExecuteCtx) {
    let ExecuteCtx {
        target,
        start,
        position,
        ..
    } = ctx;

    for i in (0..=(start.y - target.y + 1)).rev() {
        if ctx.mine_while_at_start(target.up(i)) {
            return;
        }
    }

    let start_center = start.center();
    let center = target.center();

    let horizontal_distance_from_target = (center - position).horizontal_distance_squared().sqrt();
    let horizontal_distance_from_start = (start.center() - position)
        .horizontal_distance_squared()
        .sqrt();

    let dest_ahead = Vec3::new(
        start_center.x + (center.x - start_center.x) * 1.5,
        center.y,
        start_center.z + (center.z - start_center.z) * 1.5,
    );

    if (BlockPos::from(position).horizontal_distance_squared_to(target) > 0)
        || horizontal_distance_from_target > 0.25
    {
        if horizontal_distance_from_start < 1.25 {
            // this basically just exists to avoid doing spins while we're falling
            ctx.look_at(dest_ahead);
            ctx.walk(WalkDirection::Forward);
        } else {
            ctx.look_at(center);
            ctx.walk(WalkDirection::Forward);
        }
    } else {
        ctx.walk(WalkDirection::None);
    }
}
#[must_use]
pub fn descend_is_reached(
    IsReachedCtx {
        target,
        start,
        position,
        physics,
        ..
    }: IsReachedCtx,
) -> bool {
    let dest_ahead = BlockPos::new(
        start.x + (target.x - start.x) * 2,
        target.y,
        start.z + (target.z - start.z) * 2,
    );

    if player_pos_to_block_pos(position) == target
        || player_pos_to_block_pos(position) == dest_ahead
    {
        if (position.y - target.y as f64) < 0.5 {
            return true;
        }
    } else if player_pos_to_block_pos(position).up(1) == target && physics.on_ground() {
        return true;
    }
    false
}

fn diagonal_move(ctx: &mut MovesCtx, pos: RelBlockPos) {
    let mut base_cost = SPRINT_ONE_BLOCK_COST;

    let currently_in_water = ctx.world.is_block_water(pos);
    if currently_in_water {
        if BARITONE_COMPAT {
            base_cost = WALK_ONE_BLOCK_COST;
        } else {
            base_cost = WALK_ONE_IN_WATER_COST;
        }
    }

    // add 0.001 as a tie-breaker to avoid unnecessarily going diagonal
    base_cost = base_cost.mul_add(SQRT_2, 0.001);

    for dir in CardinalDirection::iter() {
        let right = dir.right();
        let offset = RelBlockPos::new(dir.x() + right.x(), 0, dir.z() + right.z());
        let left_pos = RelBlockPos::new(pos.x + dir.x(), pos.y, pos.z + dir.z());
        let right_pos = RelBlockPos::new(pos.x + right.x(), pos.y, pos.z + right.z());

        let mut cost = base_cost;

        let left_passable;
        let right_passable;

        if currently_in_water {
            left_passable =
                ctx.world.is_block_water(left_pos) && ctx.world.is_block_passable(left_pos.up(1));
            if !left_passable {
                // don't bother hugging corners while in water
                continue;
            }
            right_passable =
                ctx.world.is_block_water(right_pos) && ctx.world.is_block_passable(right_pos.up(1));
            if !right_passable {
                continue;
            }
        } else {
            left_passable = ctx.world.is_passable(left_pos);
            right_passable = ctx.world.is_passable(right_pos);
            if !left_passable && !right_passable {
                continue;
            }
        }

        let new_position = pos + offset;
        if currently_in_water {
            if !ctx.world.is_block_water(new_position)
                || !ctx.world.is_block_passable(new_position.up(1))
            {
                continue;
            }
        } else if !ctx.world.is_standable(new_position) {
            continue;
        }

        if !left_passable || !right_passable {
            if !BARITONE_COMPAT {
                // add a bit of cost because it'll probably be hugging a wall here
                cost += WALK_ONE_BLOCK_COST / 2.;
            } else {
                cost = WALK_ONE_BLOCK_COST * (SQRT_2 - 0.001) * SQRT_2;
            }
        }

        ctx.edges.push(Edge {
            movement: astar::Movement {
                target: new_position,
                data: MoveData::new(&execute_diagonal_move, &default_is_reached),
            },
            cost,
        })
    }
}
fn execute_diagonal_move(mut ctx: ExecuteCtx) {
    let target_center = ctx.target.center();

    ctx.jump_if_in_water();

    ctx.look_at(target_center);
    ctx.sprint(SprintDirection::Forward);
}

/// Go directly down, usually by mining.
fn downward_move(ctx: &mut MovesCtx, pos: RelBlockPos) {
    // make sure we land on a solid block after breaking the one below us
    if !ctx.world.is_block_solid(pos.down(2)) {
        return;
    }

    let break_cost = ctx
        .world
        .cost_for_breaking_block(pos.down(1), ctx.mining_cache);
    if break_cost == f32::INFINITY {
        return;
    }

    let cost = FALL_N_BLOCKS_COST[1] + break_cost;

    ctx.edges.push(Edge {
        movement: astar::Movement {
            target: pos.down(1),
            data: MoveData::new(&execute_downward_move, &default_is_reached)
                .with_side_effects(MovementSideEffects::to_air(pos, [pos.down(1)])),
        },
        cost,
    })
}
/// Go directly up by placing a block under yourself mid-jump.
///
/// 抄 Baritone
/// `MovementPillar`：破头顶上方那一格、在起点脚下放一块、落到它上面。
/// 这是 A\* 里唯一一条**自己造出支撑**的边，所以不能拿 `cost_for_standing` 判断
/// 落脚点——落脚点现在还是空的，正是这条边要把它填上。
fn pillar_move(ctx: &mut MovesCtx, pos: RelBlockPos) {
    // 站在水里垫不了（Baritone 同样禁止：会被水冲走）。
    if ctx.world.is_block_water(pos) {
        return;
    }
    // 要有能贴的面：脚下那一格必须是实心的。
    if !ctx.world.is_block_solid(pos.down(1)) {
        return;
    }

    let place_cost = ctx.placement.cost_for_placing(pos, ctx.world);
    if place_cost == f32::INFINITY {
        return;
    }

    // 落脚之后头顶是 pos.up(2)，挡路就得先挖。
    let break_cost = ctx
        .world
        .cost_for_breaking_block(pos.up(2), ctx.mining_cache);
    if break_cost == f32::INFINITY {
        return;
    }

    let cost = *JUMP_ONE_BLOCK_COST + JUMP_PENALTY + place_cost + break_cost;

    ctx.edges.push(Edge {
        movement: astar::Movement {
            target: pos.up(1),
            data: MoveData::new(&execute_pillar_move, &pillar_is_reached).with_side_effects(
                MovementSideEffects::to_air(pos, [pos.up(2)]).and_solid(pos, pos),
            ),
        },
        cost,
    })
}

/// 跳到目标格的高度**不算**到达——半空中经过时脚下还是空的，这时候判成功会让
/// 执行器把这条边交掉，身体随即掉回原处。落地才算数：
/// 落得住就说明那一块真的放上了。
fn pillar_is_reached(
    IsReachedCtx {
        position,
        target,
        physics,
        ..
    }: IsReachedCtx,
) -> bool {
    player_pos_to_block_pos(position) == target && physics.on_ground()
}

fn execute_pillar_move(mut ctx: ExecuteCtx) {
    let ExecuteCtx {
        start,
        target,
        position,
        ..
    } = ctx;

    // 头顶挡路先挖，挖着的时候别跳。
    if ctx.mine_while_at_start(start.up(2)) {
        return;
    }

    // 全程蹲着：跳起来的时候不会走出这一格，右键也不会误触脚下方块的界面。
    ctx.sneak();

    // 没站正就先站正——偏出去放出来的柱子就歪了，人也踩不上。
    let target_center = target.center();
    let horizontal_distance = (target_center - position)
        .horizontal_distance_squared()
        .sqrt();
    if horizontal_distance > 0.17 {
        ctx.look_at(target_center);
        ctx.walk(WalkDirection::Forward);
        return;
    }
    ctx.walk(WalkDirection::None);

    // 脚下已经垫好了，剩下的交给落地判定。
    if !ctx.get_block_state(start).is_air() {
        return;
    }

    // 身体得先让开 start 这一格才放得下去，所以跳到高过落脚点再放。
    if position.y > f64::from(target.y) + 0.1 {
        ctx.place(start);
    } else {
        ctx.jump();
    }
}

fn execute_downward_move(mut ctx: ExecuteCtx) {
    let ExecuteCtx {
        target, position, ..
    } = ctx;

    let target_center = target.center();

    let horizontal_distance_from_target = (target_center - position)
        .horizontal_distance_squared()
        .sqrt();

    if horizontal_distance_from_target > 0.25 {
        ctx.look_at(target_center);
        ctx.walk(WalkDirection::Forward);
    } else if ctx.mine_while_at_start(target) {
        ctx.walk(WalkDirection::None);
    } else if BlockPos::from(position) != target {
        ctx.look_at(target_center);
        ctx.walk(WalkDirection::Forward);
    } else {
        ctx.walk(WalkDirection::None);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use azalea_core::position::ChunkPos;
    use azalea_inventory::{ItemStack, ItemStackData, Menu, Player};
    use azalea_registry::builtin::{BlockKind, ItemKind};
    use azalea_world::{Chunk, ChunkStorage, PartialWorld, World};
    use parking_lot::RwLock;

    use super::*;
    use crate::pathfinder::{
        custom_state::CustomPathfinderState, mining::MiningCache, placing::PlacementCache,
        world::CachedWorld,
    };

    /// 一层地面，站在 origin，头顶全空。
    fn world_with_floor(origin: BlockPos) -> (Arc<RwLock<World>>, PartialWorld) {
        let mut partial_world = PartialWorld::default();
        let mut chunks = ChunkStorage::default();
        let floor = origin.down(1);
        partial_world
            .chunks
            .set(&ChunkPos::from(floor), Some(Chunk::default()), &mut chunks);
        partial_world
            .chunks
            .set_block_state(floor, BlockKind::Stone.into(), &chunks);
        (Arc::new(RwLock::new(chunks.into())), partial_world)
    }

    fn hotbar_holding(kind: ItemKind) -> Menu {
        let mut menu = Menu::Player(Player::default());
        let first = *menu.hotbar_slots_range().start();
        *menu.slot_mut(first).expect("快捷栏第一格存在") =
            ItemStack::Present(ItemStackData::new(kind, 64));
        menu
    }

    fn pillar_edges(
        world: Arc<RwLock<World>>,
        origin: BlockPos,
        placement: PlacementCache,
    ) -> Vec<Edge> {
        let cached_world = CachedWorld::new(world, origin);
        let mining_cache = MiningCache::new(Some(Menu::Player(Player::default())));
        let custom_state = CustomPathfinderState::default();
        let custom_state_ref = custom_state.0.read();

        let mut edges = Vec::new();
        let mut ctx = MovesCtx {
            edges: &mut edges,
            world: &cached_world,
            mining_cache: &mining_cache,
            placement: &placement,
            custom_state: &custom_state_ref,
        };
        pillar_move(&mut ctx, RelBlockPos::get_origin(origin));
        edges
    }

    fn on_floor(placement: PlacementCache) -> Vec<Edge> {
        let origin = BlockPos::new(0, 1, 0);
        let (world, _owner) = world_with_floor(origin);
        pillar_edges(world, origin, placement)
    }

    /// 没有可放的方块就不该生出垫柱边——否则 A* 会规划一条执行不了的路。
    #[test]
    fn without_anything_to_place_there_is_no_pillar() {
        assert!(on_floor(PlacementCache::new(None)).is_empty());
        assert!(
            on_floor(PlacementCache::new(Some(hotbar_holding(
                ItemKind::DiamondPickaxe
            ))))
            .is_empty(),
            "手上只有镐子也垫不了"
        );
    }

    /// 有料就该多出一条往上一格的边，声明的替换正是「脚下变实心、
    /// 头顶上方变空气」。
    #[test]
    fn a_pillar_declares_the_block_it_will_stand_on() {
        let edges = on_floor(PlacementCache::new(Some(hotbar_holding(
            ItemKind::Cobblestone,
        ))));
        assert_eq!(edges.len(), 1, "脚下有实心、头顶空着时恰好一条垫柱边");

        let origin = BlockPos::new(0, 1, 0);
        let edge = &edges[0];
        assert_eq!(edge.movement.target.apply(origin), origin.up(1));

        let side_effects = edge.movement.data.side_effects;
        assert_eq!(
            side_effects.block_to_place(origin),
            Some(origin),
            "垫的就是当前站的这一格"
        );
        assert_eq!(
            side_effects.blocks_to_break(origin).collect::<Vec<_>>(),
            [origin.up(2)],
            "只允许挖落脚之后的头顶那一格"
        );
        assert!(
            edge.cost > BLOCK_PLACEMENT_PENALTY,
            "放置代价必须计入，否则 A* 会当它和走路一样便宜"
        );
    }

    /// 策略不许放这种方块，就不该生出垫柱边——手上有料不等于准放。
    #[test]
    fn a_policy_that_forbids_the_block_forbids_the_pillar() {
        use crate::pathfinder::policy::{PolicySnapshot, ReplacePolicy, ReplaceTo};

        struct NoCobblestone;
        impl ReplacePolicy for NoCobblestone {
            fn allows_replace(&self, _: BlockPos, _: BlockState, to: ReplaceTo) -> bool {
                to != ReplaceTo::Block(BlockKind::Cobblestone)
            }
        }

        let origin = BlockPos::new(0, 1, 0);
        let (world, _owner) = world_with_floor(origin);
        let cached_world = CachedWorld::new(world, origin).with_replace_policy(
            PolicySnapshot::new(Some(Arc::new(NoCobblestone)), true, true),
        );
        let mining_cache = MiningCache::new(Some(Menu::Player(Player::default())));
        let placement = PlacementCache::new(Some(hotbar_holding(ItemKind::Cobblestone)));
        let custom_state = CustomPathfinderState::default();
        let custom_state_ref = custom_state.0.read();

        let mut edges = Vec::new();
        let mut ctx = MovesCtx {
            edges: &mut edges,
            world: &cached_world,
            mining_cache: &mining_cache,
            placement: &placement,
            custom_state: &custom_state_ref,
        };
        pillar_move(&mut ctx, RelBlockPos::get_origin(origin));
        assert!(edges.is_empty(), "手上只有圆石而策略不许放圆石");
    }

    /// 脚下悬空时贴不住，不该生出垫柱边。
    #[test]
    fn a_pillar_needs_something_to_place_against() {
        let origin = BlockPos::new(0, 1, 0);
        let mut partial_world = PartialWorld::default();
        let mut chunks = ChunkStorage::default();
        partial_world
            .chunks
            .set(&ChunkPos::from(origin), Some(Chunk::default()), &mut chunks);
        let world: Arc<RwLock<World>> = Arc::new(RwLock::new(chunks.into()));
        let _owner = partial_world;

        let edges = pillar_edges(
            world,
            origin,
            PlacementCache::new(Some(hotbar_holding(ItemKind::Cobblestone))),
        );
        assert!(edges.is_empty());
    }
}
