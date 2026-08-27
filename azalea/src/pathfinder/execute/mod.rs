pub mod patching;
pub mod simulation;

use std::{cmp, collections::VecDeque, time::Duration};

use azalea_block::{BlockState, BlockTrait};
use azalea_client::{
    StartSprintEvent, StartWalkEvent,
    local_player::WorldHolder,
    mining::{Mining, MiningSystems, StartMiningBlockEvent},
};
use azalea_core::{position::Vec3, tick::GameTick};
use azalea_entity::{Physics, Position, inventory::Inventory};
use azalea_physics::{PhysicsSystems, get_block_pos_below_that_affects_movement};
use azalea_world::{WorldName, Worlds};
use bevy_app::{App, Plugin};
use bevy_ecs::prelude::*;
use tracing::{debug, info, trace, warn};

use crate::{
    WalkDirection,
    bot::{JumpEvent, LookAtEvent},
    ecs::{
        entity::Entity,
        query::Without,
        system::{Commands, Query, Res},
    },
    pathfinder::{
        ComputePath, ExecutingPath, GotoEvent, Pathfinder, PathfinderSystems,
        astar::PathfinderTimeout,
        custom_state::CustomPathfinderState,
        debug::debug_render_path_with_particles,
        execute::simulation::SimulatingPathState,
        moves::{ExecuteCtx, IsReachedCtx},
        next_path_calculation_id, player_pos_to_block_pos, reserve_queued_goto,
    },
};

pub struct DefaultPathfinderExecutionPlugin;
impl Plugin for DefaultPathfinderExecutionPlugin {
    fn build(&self, _app: &mut App) {}

    fn finish(&self, app: &mut App) {
        if app.is_plugin_added::<simulation::SimulationPathfinderExecutionPlugin>() {
            info!("pathfinder simulation executor plugin is enabled, disabling default executor.");
            return;
        }

        app.add_systems(
            GameTick,
            (
                timeout_movement,
                patching::check_for_path_obstruction,
                check_node_reached,
                tick_execute_path,
                recalculate_near_end_of_path,
                recalculate_if_has_goal_but_no_path,
            )
                .chain()
                .after(PhysicsSystems)
                .after(azalea_client::movement::send_position)
                .after(MiningSystems)
                .after(debug_render_path_with_particles)
                .in_set(PathfinderSystems),
        );
    }
}

pub(super) fn retire_pathfinder_request(pathfinder: &mut Pathfinder) {
    next_path_calculation_id(&pathfinder.goto_id);
    pathfinder.queued_goto_id = None;
    pathfinder.goal = None;
    pathfinder.opts = None;
    pathfinder.is_calculating = false;
}

#[allow(clippy::type_complexity)]
pub fn tick_execute_path(
    mut commands: Commands,
    mut query: Query<(
        Entity,
        &mut ExecutingPath,
        &Position,
        &Physics,
        Option<&Mining>,
        &WorldHolder,
        &Inventory,
    )>,
    mut look_at_events: MessageWriter<LookAtEvent>,
    mut sprint_events: MessageWriter<StartSprintEvent>,
    mut walk_events: MessageWriter<StartWalkEvent>,
    mut jump_events: MessageWriter<JumpEvent>,
    mut start_mining_events: MessageWriter<StartMiningBlockEvent>,
) {
    for (entity, mut executing_path, position, physics, mining, world_holder, inventory) in
        &mut query
    {
        executing_path.ticks_since_last_node_reached += 1;

        if let Some(edge) = executing_path.path.front() {
            let mut ctx = ExecuteCtx {
                entity,
                target: edge.movement.target,
                position: **position,
                start: executing_path.last_reached_node,
                physics,
                is_currently_mining: mining.is_some(),
                can_mine: executing_path.allow_mining,
                side_effects: edge.movement.data.side_effects,
                world: world_holder.shared.clone(),
                menu: inventory.inventory_menu.clone(),

                commands: &mut commands,
                look_at_events: &mut look_at_events,
                sprint_events: &mut sprint_events,
                walk_events: &mut walk_events,
                jump_events: &mut jump_events,
                start_mining_events: &mut start_mining_events,
            };
            ctx.on_tick_start();
            trace!(
                "executing move, position: {}, last_reached_node: {}",
                **position, executing_path.last_reached_node
            );
            (edge.movement.data.execute)(ctx);
        }
    }
}

pub fn check_node_reached(
    mut query: Query<(
        Entity,
        &mut Pathfinder,
        &mut ExecutingPath,
        &Position,
        &Physics,
        &WorldName,
    )>,
    mut walk_events: MessageWriter<StartWalkEvent>,
    mut commands: Commands,
    worlds: Res<Worlds>,
) {
    for (entity, mut pathfinder, mut executing_path, position, physics, world_name) in &mut query {
        let Some(world) = worlds.get(world_name) else {
            warn!("entity is pathfinding but not in a valid world");
            continue;
        };

        'skip: loop {
            // we check if the goal was reached *before* actually executing the movement so
            // we don't unnecessarily execute a movement when it wasn't necessary

            // see if we already reached any future nodes and can skip ahead
            for (i, edge) in executing_path
                .path
                .clone()
                .into_iter()
                .enumerate()
                .take(30)
                .rev()
            {
                let movement = edge.movement;
                let is_reached_ctx = IsReachedCtx {
                    target: movement.target,
                    start: executing_path.last_reached_node,
                    position: **position,
                    physics,
                };
                let extra_check = if i == executing_path.path.len() - 1
                    // only do the extra check if we don't have a new path immediately queued up
                    && executing_path.is_empty_queued_path()
                {
                    // be extra strict about the velocity and centering if we're on the last node so
                    // we don't fall off

                    let x_difference_from_center = position.x - (movement.target.x as f64 + 0.5);
                    let z_difference_from_center = position.z - (movement.target.z as f64 + 0.5);

                    let block_pos_below = get_block_pos_below_that_affects_movement(*position);

                    let block_state_below = {
                        let world = world.read();
                        world
                            .chunks
                            .get_block_state(block_pos_below)
                            .unwrap_or(BlockState::AIR)
                    };
                    let block_below: Box<dyn BlockTrait> = block_state_below.into();
                    // friction for normal blocks is 0.6, for ice it's 0.98
                    let block_friction = block_below.behavior().friction as f64;

                    // if the block has the default friction, this will multiply by 1
                    // for blocks like ice, it'll multiply by a higher number
                    let scaled_velocity = physics.velocity * (0.4 / (1. - block_friction));

                    let x_predicted_offset = (x_difference_from_center + scaled_velocity.x).abs();
                    let z_predicted_offset = (z_difference_from_center + scaled_velocity.z).abs();

                    // this is to make sure we don't fall off immediately after finishing the path
                    (physics.on_ground() || physics.is_in_water())
                        && player_pos_to_block_pos(**position) == movement.target
                        // adding the delta like this isn't a perfect solution but it helps to make
                        // sure we don't keep going if our delta is high
                        && x_predicted_offset < 0.2
                        && z_predicted_offset < 0.2
                } else {
                    true
                };

                if (movement.data.is_reached)(is_reached_ctx) && extra_check {
                    executing_path.path = executing_path.path.split_off(i + 1);
                    executing_path.last_reached_node = movement.target;
                    executing_path.ticks_since_last_node_reached = 0;
                    trace!("reached node {}", movement.target);

                    if let Some(new_path) = executing_path.queued_path.take() {
                        debug!(
                            "swapped path to {:?}",
                            new_path.iter().take(10).collect::<Vec<_>>()
                        );
                        executing_path.path = new_path;

                        if executing_path.path.is_empty() {
                            info!("the path we just swapped to was empty, so reached end of path");
                            walk_events.write(StartWalkEvent {
                                entity,
                                direction: WalkDirection::None,
                            });
                            commands.entity(entity).remove::<ExecutingPath>();
                            if pathfinder.goal.is_none() {
                                pathfinder.opts = None;
                            }
                            break;
                        }

                        // run the function again since we just swapped
                        continue 'skip;
                    }

                    if executing_path.path.is_empty() {
                        debug!("pathfinder path is now empty");
                        walk_events.write(StartWalkEvent {
                            entity,
                            direction: WalkDirection::None,
                        });
                        commands.entity(entity).remove::<ExecutingPath>();
                        if pathfinder.queued_goto_id.is_none()
                            && let Some(goal) = pathfinder.goal.clone()
                            && goal.success(movement.target)
                        {
                            info!("goal was reached!");
                            // The segment that reached this goal may have been
                            // retained while a replacement calculation ran.
                            // Invalidate that worker before it can revive
                            // movement after arrival.
                            retire_pathfinder_request(&mut pathfinder);
                            commands.entity(entity).remove::<ComputePath>();
                        } else if executing_path.is_path_partial
                            && !pathfinder.is_calculating
                            && pathfinder.queued_goto_id.is_none()
                            && pathfinder
                                .opts
                                .as_ref()
                                .is_some_and(|opts| !opts.recalculate_partial_paths)
                        {
                            debug!("partial path ended with automatic recalculation disabled");
                            retire_pathfinder_request(&mut pathfinder);
                            commands.entity(entity).remove::<ComputePath>();
                        } else if pathfinder.goal.is_none() {
                            // A graceful stop keeps opts while the last movement
                            // is in flight so local patching remains available.
                            pathfinder.opts = None;
                        }
                    }

                    break;
                }
            }
            break;
        }
    }
}

#[allow(clippy::type_complexity)]
pub fn timeout_movement(
    mut query: Query<(
        Entity,
        &mut Pathfinder,
        &mut ExecutingPath,
        &Position,
        Option<&Mining>,
        &WorldName,
        &Inventory,
        Option<&CustomPathfinderState>,
        Option<&crate::pathfinder::world::PathfinderBlockSource>,
        Option<&mut SimulatingPathState>,
    )>,
    worlds: Res<Worlds>,
) {
    for (
        entity,
        mut pathfinder,
        mut executing_path,
        position,
        mining,
        world_name,
        inventory,
        custom_state,
        block_source,
        simulating_path_state,
    ) in &mut query
    {
        // A queued replacement or an active full-goal calculation owns the
        // current generation. Local timeout patching allocates a generation
        // of its own, so running it here would incorrectly stale that work.
        if pathfinder.queued_goto_id.is_some() || pathfinder.is_calculating {
            continue;
        }

        if !executing_path.path.is_empty() {
            let (start, end) = if let Some(s) = &simulating_path_state
                && let SimulatingPathState::Simulated(simulating_path_state) = &**s
            {
                (simulating_path_state.start, simulating_path_state.target)
            } else {
                (
                    executing_path.last_reached_node,
                    executing_path.path[0].movement.target,
                )
            };

            let (start, end) = (start.center_bottom(), end.center_bottom());
            // TODO: use an actual 2d point-line distance formula here instead of the 3d one
            // lol
            let xz_distance =
                point_line_distance_3d(&position.with_y(0.), &(start.with_y(0.), end.with_y(0.)));
            let y_distance = point_line_distance_1d(position.y, (start.y, end.y));

            let xz_tolerance = 3.;
            // longer moves have more y tolerance (in case we're climbing a hill or smth in
            // a single movement)
            let y_tolerance = start.horizontal_distance_to(end) / 2. + 1.5;

            if xz_distance > xz_tolerance || y_distance > y_tolerance {
                warn!(
                    "pathfinder went too far from path (xz_distance={xz_distance}/{xz_tolerance}, y_distance={y_distance}/{y_tolerance}, line is {start} to {end}, point at {}), trying to patch!",
                    **position
                );

                if let Some(mut simulating_path_state) = simulating_path_state {
                    // don't keep executing the simulation
                    *simulating_path_state = SimulatingPathState::Fail;
                }

                patch_path_from_timeout(
                    entity,
                    &mut executing_path,
                    &mut pathfinder,
                    &worlds,
                    position,
                    world_name,
                    custom_state,
                    block_source,
                    inventory,
                );
                continue;
            }
        }

        // don't timeout if we're mining
        if let Some(mining) = mining {
            // also make sure we're close enough to the block that's being mined
            if mining.pos.distance_squared_to(position.into()) < 6_i32.pow(2) {
                // also reset the ticks_since_last_node_reached so we don't timeout after we
                // finish mining
                executing_path.ticks_since_last_node_reached = 0;
                continue;
            }
        }

        let mut timeout = 2 * 20;

        if simulating_path_state.is_some() {
            // longer timeout if we're following a simulated path from the other execution
            // engine
            timeout = 5 * 20;
        }

        if executing_path.ticks_since_last_node_reached > timeout && !executing_path.path.is_empty()
        {
            warn!("pathfinder timeout, trying to patch path");

            patch_path_from_timeout(
                entity,
                &mut executing_path,
                &mut pathfinder,
                &worlds,
                position,
                world_name,
                custom_state,
                block_source,
                inventory,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn patch_path_from_timeout(
    entity: Entity,
    executing_path: &mut ExecutingPath,
    pathfinder: &mut Pathfinder,
    worlds: &Worlds,
    position: &Position,
    world_name: &WorldName,
    custom_state: Option<&CustomPathfinderState>,
    block_source: Option<&crate::pathfinder::world::PathfinderBlockSource>,
    inventory: &Inventory,
) {
    let graceful_stop_pending = graceful_stop_pending(pathfinder, executing_path);
    executing_path.queued_path = None;
    let cur_pos = player_pos_to_block_pos(**position);
    executing_path.last_reached_node = cur_pos;

    let world_lock = worlds
        .get(world_name)
        .expect("Entity tried to pathfind but the entity isn't in a valid world");
    let Some(opts) = pathfinder.opts.clone() else {
        warn!(
            "pathfinder was going to patch path because of timeout, but pathfinder.opts was None; ending the current segment"
        );
        executing_path.path.clear();
        executing_path.is_path_partial = true;
        retire_pathfinder_request(pathfinder);
        finish_timeout_patch(
            executing_path,
            graceful_stop_pending,
            patching::PatchOutcome::NoPath,
        );
        return;
    };

    let custom_state = custom_state.cloned().unwrap_or_default();

    // try to fix the path without recalculating everything.
    // (though, it'll still get fully recalculated by `recalculate_near_end_of_path`
    // if the new path is too short)
    let outcome = patching::patch_path(
        0..=cmp::min(20, executing_path.path.len() - 1),
        executing_path,
        pathfinder,
        inventory,
        entity,
        world_lock,
        custom_state,
        block_source.map(|source| source.0.clone()),
        opts,
    );
    finish_timeout_patch(executing_path, graceful_stop_pending, outcome);
}

pub(super) fn finish_timeout_patch(
    executing_path: &mut ExecutingPath,
    graceful_stop_pending: bool,
    outcome: patching::PatchOutcome,
) {
    restore_graceful_stop(executing_path, graceful_stop_pending);
    if outcome == patching::PatchOutcome::Applied {
        // A usable replacement is real recovery. Give that replacement its
        // own movement deadline instead of immediately patching it again.
        executing_path.ticks_since_last_node_reached = 0;
    }
}

pub(super) fn graceful_stop_pending(
    pathfinder: &Pathfinder,
    executing_path: &ExecutingPath,
) -> bool {
    pathfinder.goal.is_none()
        && executing_path
            .queued_path
            .as_ref()
            .is_some_and(VecDeque::is_empty)
}

pub(super) fn restore_graceful_stop(
    executing_path: &mut ExecutingPath,
    graceful_stop_pending: bool,
) {
    if graceful_stop_pending {
        // Timeout recovery may replace the current movement, but it must not
        // erase a graceful stop's "finish one movement, then stop" sentinel.
        executing_path.queued_path = Some(VecDeque::new());
        executing_path.is_path_partial = false;
    }
}

pub fn recalculate_near_end_of_path(
    mut query: Query<(Entity, &mut Pathfinder, &mut ExecutingPath)>,
    mut walk_events: MessageWriter<StartWalkEvent>,
    mut goto_events: MessageWriter<GotoEvent>,
    mut commands: Commands,
) {
    for (entity, mut pathfinder, mut executing_path) in &mut query {
        // No active goal can own an empty execution component. This includes
        // graceful-stop timeout recovery when patching produced no movement.
        // Handle it independently of retry/partial options so it cannot become
        // a motionless zombie with latched walk input.
        if pathfinder.goal.is_none() && executing_path.path.is_empty() {
            if pathfinder.queued_goto_id.is_none() {
                retire_pathfinder_request(&mut pathfinder);
            }
            commands.entity(entity).remove::<ComputePath>();
            commands.entity(entity).remove::<ExecutingPath>();
            walk_events.write(StartWalkEvent {
                entity,
                direction: WalkDirection::None,
            });
            continue;
        }

        let Some(mut opts) = pathfinder.opts.clone() else {
            continue;
        };

        // A caller may deliberately own full-goal replanning while still
        // letting Azalea execute and locally patch the returned segment. If a
        // patch truncated that segment to zero, retire the empty execution
        // component instead of leaving a motionless zombie.
        if executing_path.path.is_empty() && !opts.recalculate_partial_paths {
            if let Some(new_path) = executing_path.queued_path.take() {
                executing_path.path = new_path;
            }
            if executing_path.path.is_empty() {
                // An empty old segment may coexist with a replacement goto.
                // Retire the segment, but leave that active request intact.
                if !pathfinder.is_calculating && pathfinder.queued_goto_id.is_none() {
                    retire_pathfinder_request(&mut pathfinder);
                    commands.entity(entity).remove::<ComputePath>();
                }
                walk_events.write(StartWalkEvent {
                    entity,
                    direction: WalkDirection::None,
                });
                commands.entity(entity).remove::<ExecutingPath>();
            }
            continue;
        }

        // start recalculating if the path ends soon. 50 is arbitrary, that's just to
        // make us recalculate once when we start nearing the end. this doesn't account
        // for skipping nodes, though...
        // TODO: have a variable to store whether we've recalculated, and then check
        // that `&& path.len() <= 50` to see if we should recalculate.
        if (executing_path.path.len() == 50 || executing_path.path.len() < 5)
            && !pathfinder.is_calculating
            && pathfinder.queued_goto_id.is_none()
            && executing_path.is_path_partial
            && opts.recalculate_partial_paths
        {
            match pathfinder.goal.as_ref().cloned() {
                Some(goal) => {
                    debug!("Recalculating path because it's empty or ends soon");
                    debug!(
                        "recalculate_near_end_of_path executing_path.is_path_partial: {}",
                        executing_path.is_path_partial
                    );

                    opts.min_timeout = if executing_path.path.len() == 50 {
                        // we have quite some time until the node is reached, soooo we might as
                        // well burn some cpu cycles to get a good path
                        PathfinderTimeout::Time(Duration::from_secs(5))
                    } else {
                        PathfinderTimeout::Time(Duration::from_secs(1))
                    };

                    let calculation_id = reserve_queued_goto(&mut pathfinder);
                    goto_events.write(GotoEvent {
                        entity,
                        goal,
                        opts,
                        calculation_id: Some(calculation_id),
                    });
                    pathfinder.is_calculating = true;

                    if executing_path.path.is_empty() {
                        if let Some(new_path) = executing_path.queued_path.take() {
                            executing_path.path = new_path;
                            if executing_path.path.is_empty() {
                                info!(
                                    "the path we just swapped to was empty, so reached end of path"
                                );
                                walk_events.write(StartWalkEvent {
                                    entity,
                                    direction: WalkDirection::None,
                                });
                                commands.entity(entity).remove::<ExecutingPath>();
                                continue;
                            }
                        } else {
                            walk_events.write(StartWalkEvent {
                                entity,
                                direction: WalkDirection::None,
                            });
                            commands.entity(entity).remove::<ExecutingPath>();
                        }
                    }
                }
                _ => {
                    if executing_path.path.is_empty() {
                        // idk when this can happen but stop moving just in case
                        pathfinder.opts = None;
                        walk_events.write(StartWalkEvent {
                            entity,
                            direction: WalkDirection::None,
                        });
                        commands.entity(entity).remove::<ExecutingPath>();
                    }
                }
            }
        }
    }
}

pub fn recalculate_if_has_goal_but_no_path(
    mut query: Query<(Entity, &mut Pathfinder), Without<ExecutingPath>>,
    mut goto_events: MessageWriter<GotoEvent>,
) {
    for (entity, mut pathfinder) in &mut query {
        if pathfinder.goal.is_some()
            && !pathfinder.is_calculating
            && pathfinder.queued_goto_id.is_none()
            && let Some(goal) = pathfinder.goal.as_ref().cloned()
            && let Some(opts) = pathfinder.opts.clone()
        {
            if !opts.retry_on_no_path {
                debug!("retry_on_no_path is disabled, retiring the exhausted goal");
                pathfinder.goal = None;
                pathfinder.opts = None;
                continue;
            }
            debug!("Recalculating path because it has a goal but no ExecutingPath");
            let calculation_id = reserve_queued_goto(&mut pathfinder);
            goto_events.write(GotoEvent {
                entity,
                goal,
                opts,
                calculation_id: Some(calculation_id),
            });
            pathfinder.is_calculating = true;
        }
    }
}

// based on https://stackoverflow.com/a/36425155
/// Returns the distance of a point from a line.
///
/// This is used in the pathfinder for checking if the bot is too far from the
/// current path.
pub fn point_line_distance_3d(point: &Vec3, (start, end): &(Vec3, Vec3)) -> f64 {
    let start_to_end = end - start;
    let start_to_point = point - start;

    if start_to_point.dot(start_to_end) <= 0. {
        return start_to_point.length();
    }

    let end_to_point = point - end;
    if end_to_point.dot(start_to_end) >= 0. {
        return end_to_point.length();
    }

    start_to_end.cross(start_to_point).length() / start_to_end.length()
}
pub fn point_line_distance_1d(point: f64, (start, end): (f64, f64)) -> f64 {
    let min = start.min(end);
    let max = start.max(end);
    if point < min {
        min - point
    } else if point > max {
        point - max
    } else {
        0.
    }
}
