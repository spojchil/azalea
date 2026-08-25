//! A pathfinding plugin to make bots able to traverse the world.
//!
//! For the new functions on `Client` that the pathfinder adds, see
//! [`PathfinderClientExt`].
//!
//! Note that the pathfinder is highly optimized, but it will be very slow if
//! it's not compiled with optimizations enabled.
//!
//! For smoother and more realistic path execution, also see
//! [`SimulationPathfinderExecutionPlugin`].
//!
//! Much of the pathfinder's code is based on [Baritone](https://github.com/cabaletta/baritone). <3
//!
//! [`SimulationPathfinderExecutionPlugin`]: execute::simulation::SimulationPathfinderExecutionPlugin

pub mod astar;
pub mod costs;
pub mod custom_state;
pub mod debug;
pub mod execute;
pub mod goals;
mod goto_event;
#[cfg(test)]
mod lifecycle_tests;
pub mod mining;
pub mod moves;
pub mod positions;
pub mod simulation;
#[cfg(test)]
mod tests;
pub mod world;

use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{self, AtomicBool, AtomicUsize},
    },
    thread,
    time::{Duration, Instant},
};

use astar::Edge;
use azalea_client::{StartWalkEvent, inventory::InventorySystems, movement::MoveEventsSystems};
use azalea_core::{
    position::{BlockPos, Vec3},
    tick::GameTick,
};
use azalea_entity::{LocalEntity, Position, inventory::Inventory, metadata::Player};
use azalea_world::{WorldName, Worlds};
use bevy_app::{PreUpdate, Update};
use bevy_ecs::prelude::*;
use bevy_tasks::{AsyncComputeTaskPool, Task};
use custom_state::{CustomPathfinderState, CustomPathfinderStateRef};
use futures_lite::future;
pub use goto_event::{GotoEvent, PathfinderOpts};
use parking_lot::RwLock;
use positions::RelBlockPos;
use tokio::sync::broadcast::error::RecvError;
use tracing::{debug, error, info, warn};

use self::{
    debug::debug_render_path_with_particles, goals::Goal, mining::MiningCache, moves::SuccessorsFn,
};
use crate::{
    Client, WalkDirection,
    app::{App, Plugin},
    ecs::{
        component::Component,
        entity::Entity,
        query::{With, Without},
        system::{Commands, Query, Res},
    },
    pathfinder::{
        astar::{PathfinderTimeout, a_star},
        execute::{DefaultPathfinderExecutionPlugin, simulation::SimulatingPathState},
        moves::MovesCtx,
        world::CachedWorld,
    },
};

#[derive(Clone, Default)]
pub struct PathfinderPlugin;
impl Plugin for PathfinderPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<GotoEvent>()
            .add_message::<PathFoundEvent>()
            .add_message::<StopPathfindingEvent>()
            .add_systems(
                GameTick,
                debug_render_path_with_particles.in_set(PathfinderSystems),
            )
            .add_systems(PreUpdate, add_default_pathfinder.in_set(PathfinderSystems))
            .add_systems(
                Update,
                (
                    goto_listener,
                    handle_tasks,
                    stop_pathfinding_on_world_change,
                    path_found_listener,
                    handle_stop_pathfinding_event,
                )
                    .chain()
                    .before(MoveEventsSystems)
                    .before(InventorySystems)
                    .in_set(PathfinderSystems),
            )
            .add_plugins(DefaultPathfinderExecutionPlugin);
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, SystemSet)]
pub struct PathfinderSystems;

/// A component that makes this client able to pathfind.
#[derive(Clone, Component, Default)]
#[non_exhaustive]
pub struct Pathfinder {
    pub goal: Option<Arc<dyn Goal>>,
    pub opts: Option<PathfinderOpts>,
    pub is_calculating: bool,
    /// The generation of a stamped [`GotoEvent`] that has been queued but has
    /// not yet been consumed by [`goto_listener`].
    ///
    /// Unlike `goal` and `is_calculating`, this is set synchronously by
    /// [`PathfinderClientExt::start_goto_with_opts`]. Consumers can therefore
    /// distinguish a request that is still waiting in Bevy's message buffer
    /// from one that was consumed and immediately completed without a path.
    pub queued_goto_id: Option<usize>,
    pub goto_id: Arc<AtomicUsize>,
}

/// A component that's present on clients that are actively following a
/// pathfinder path.
#[derive(Clone, Component)]
pub struct ExecutingPath {
    pub path: VecDeque<astar::Edge<BlockPos, moves::MoveData>>,
    pub queued_path: Option<VecDeque<astar::Edge<BlockPos, moves::MoveData>>>,
    pub last_reached_node: BlockPos,
    // count ticks instead of using real time to make our timeouts more consistent, in case we lag
    // and our ticks take a while
    pub ticks_since_last_node_reached: usize,
    pub is_path_partial: bool,
}
impl ExecutingPath {
    pub fn is_empty_queued_path(&self) -> bool {
        self.queued_path.is_none() || self.queued_path.as_ref().is_some_and(|p| p.is_empty())
    }
}

#[derive(Clone, Debug, Message)]
#[non_exhaustive]
pub struct PathFoundEvent {
    pub entity: Entity,
    /// The calculation generation that produced this result. Results are
    /// checked again when they are applied so a stop or replacement that
    /// happens after A* finishes cannot revive an obsolete path.
    pub calculation_id: usize,
    pub start: BlockPos,
    pub path: Option<VecDeque<astar::Edge<BlockPos, moves::MoveData>>>,
    pub is_partial: bool,
    pub successors_fn: SuccessorsFn,
    pub allow_mining: bool,
}

#[allow(clippy::type_complexity)]
pub fn add_default_pathfinder(
    mut commands: Commands,
    mut query: Query<Entity, (Without<Pathfinder>, With<LocalEntity>, With<Player>)>,
) {
    for entity in &mut query {
        commands.entity(entity).insert(Pathfinder::default());
    }
}

pub trait PathfinderClientExt {
    /// Pathfind to the given goal and wait until either the target is reached
    /// or the pathfinding is canceled.
    ///
    /// You can use [`Self::start_goto`] instead if you don't want to wait.
    ///
    /// ```
    /// # use azalea::prelude::*;
    /// # use azalea::{BlockPos, pathfinder::goals::BlockPosGoal};
    /// # async fn example(bot: &Client) {
    /// bot.goto(BlockPosGoal(BlockPos::new(0, 70, 0))).await;
    /// # }
    /// ```
    fn goto(&self, goal: impl Goal + 'static) -> impl Future<Output = ()>;
    /// Same as [`Self::goto`], but allows you to set custom options for
    /// pathfinding, including disabling mining and setting custom moves.
    ///
    /// ```
    /// # use azalea::prelude::*;
    /// # use azalea::{BlockPos, pathfinder::{goals::BlockPosGoal, PathfinderOpts}};
    /// # async fn example(bot: &Client) {
    /// bot.goto_with_opts(
    ///     BlockPosGoal(BlockPos::new(0, 70, 0)),
    ///     PathfinderOpts::new().allow_mining(false),
    /// )
    /// .await;
    /// # }
    /// ```
    fn goto_with_opts(
        &self,
        goal: impl Goal + 'static,
        opts: PathfinderOpts,
    ) -> impl Future<Output = ()>;
    /// Start pathfinding to a given goal.
    ///
    /// ```
    /// # use azalea::prelude::*;
    /// # use azalea::{BlockPos, pathfinder::goals::BlockPosGoal};
    /// # fn example(bot: &Client) {
    /// bot.start_goto(BlockPosGoal(BlockPos::new(0, 70, 0)));
    /// # }
    /// ```
    fn start_goto(&self, goal: impl Goal + 'static);
    /// Same as [`Self::start_goto`], but allows you to set custom
    /// options for pathfinding, including disabling mining and setting custom
    /// moves.
    ///
    /// Also see [`Self::goto_with_opts`].
    fn start_goto_with_opts(&self, goal: impl Goal + 'static, opts: PathfinderOpts);
    /// Stop calculating a path, and stop moving once the current movement is
    /// finished.
    ///
    /// This behavior exists to prevent the bot from taking damage if
    /// `stop_pathfinding` was called while executing a parkour jump, but if
    /// it's undesirable then you may want to consider using
    /// [`Self::force_stop_pathfinding`] instead.
    fn stop_pathfinding(&self);
    /// Queue a request to stop calculating a path and stop executing the
    /// current movement without waiting for that movement to finish.
    ///
    /// This buffered API takes effect when the pathfinder stop listener runs.
    /// Use [`Client::force_retire_pathfinding`] when retirement must happen
    /// synchronously before this call returns.
    fn force_stop_pathfinding(&self);
    /// Waits forever until the bot no longer has a pathfinder goal.
    fn wait_until_goto_target_reached(&self) -> impl Future<Output = ()>;
    /// Returns true if the pathfinder has no queued request or active goal and
    /// isn't calculating a path.
    fn is_goto_target_reached(&self) -> bool;
    /// Whether the pathfinder is currently following a path.
    ///
    /// Also see [`Self::is_calculating_path`] and
    /// [`Self::is_goto_target_reached`].
    fn is_executing_path(&self) -> bool;
    /// Whether the pathfinder is currently calculating a path.
    ///
    /// Also see [`Self::is_executing_path`] and
    /// [`Self::is_goto_target_reached`].
    fn is_calculating_path(&self) -> bool;
}

impl PathfinderClientExt for Client {
    async fn goto(&self, goal: impl Goal + 'static) {
        self.goto_with_opts(goal, PathfinderOpts::new()).await;
    }
    async fn goto_with_opts(&self, goal: impl Goal + 'static, opts: PathfinderOpts) {
        self.start_goto_with_opts(goal, opts);
        self.wait_until_goto_target_reached().await;
    }
    fn start_goto(&self, goal: impl Goal + 'static) {
        self.start_goto_with_opts(goal, PathfinderOpts::new());
    }
    fn start_goto_with_opts(&self, goal: impl Goal + 'static, opts: PathfinderOpts) {
        let mut ecs = self.ecs.write();
        let mut event = GotoEvent::new(self.entity, goal, opts);
        if let Some(mut pathfinder) = ecs.get_mut::<Pathfinder>(self.entity) {
            // Reserve while the same ECS write lock that queues the request is
            // held. A later client request or synchronous retirement can now
            // make this still-buffered message stale before the listener runs.
            event.calculation_id = Some(reserve_queued_goto(&mut pathfinder));
        }
        ecs.write_message(event);
    }
    fn stop_pathfinding(&self) {
        self.ecs.write().write_message(StopPathfindingEvent {
            entity: self.entity,
            force: false,
        });
    }
    fn force_stop_pathfinding(&self) {
        self.ecs.write().write_message(StopPathfindingEvent {
            entity: self.entity,
            force: true,
        });
    }
    async fn wait_until_goto_target_reached(&self) {
        // we do this to make sure the event got handled before we start checking
        // is_goto_target_reached
        self.wait_updates(1).await;

        let mut tick_broadcaster = self.get_tick_broadcaster();
        while !self.is_goto_target_reached() {
            // check every tick
            match tick_broadcaster.recv().await {
                Ok(_) => (),
                Err(RecvError::Closed) => return,
                Err(err) => warn!("{err}"),
            };
        }
    }
    fn is_goto_target_reached(&self) -> bool {
        self.get_component::<Pathfinder>()
            .is_none_or(|p| p.queued_goto_id.is_none() && p.goal.is_none() && !p.is_calculating)
    }
    fn is_executing_path(&self) -> bool {
        self.get_component::<ExecutingPath>().is_some()
    }
    fn is_calculating_path(&self) -> bool {
        self.get_component::<Pathfinder>()
            .is_some_and(|p| p.is_calculating)
    }
}

impl Client {
    /// Synchronously retire every request queued through this client's
    /// pathfinder methods, as well as any calculating or executing request,
    /// then queue a walking-input stop.
    ///
    /// Unlike [`PathfinderClientExt::force_stop_pathfinding`], all pathfinder
    /// generation, state, and component cleanup takes effect while the
    /// client's ECS write lock is held. This is useful when a replacement or
    /// terminal outcome must be linearized before another queued goto can run.
    /// Raw unstamped [`GotoEvent::new`] messages are outside this synchronous
    /// ordering guarantee.
    pub fn force_retire_pathfinding(&self) {
        force_retire_pathfinding_in_world(&mut self.ecs.write(), self.entity);
    }
}

#[derive(Component)]
pub struct ComputePath(Task<Option<PathFoundEvent>>);

#[allow(clippy::type_complexity)]
pub fn goto_listener(
    mut commands: Commands,
    mut events: MessageReader<GotoEvent>,
    mut path_found_events: MessageWriter<PathFoundEvent>,
    mut walk_events: MessageWriter<StartWalkEvent>,
    mut query: Query<(
        &mut Pathfinder,
        Option<&mut ExecutingPath>,
        Option<&SimulatingPathState>,
        Option<&Position>,
        Option<&WorldName>,
        Option<&Inventory>,
        Option<&CustomPathfinderState>,
        Option<&world::PathfinderBlockSource>,
    )>,
    worlds: Res<Worlds>,
) {
    let thread_pool = AsyncComputeTaskPool::get();

    for event in events.read() {
        let Ok((
            mut pathfinder,
            executing_path,
            simulating_path_state,
            position,
            world_name,
            inventory,
            custom_state,
            block_source,
        )) = query.get_mut(event.entity)
        else {
            warn!("got goto event for an entity that can't pathfind");
            continue;
        };
        // 借用不能逃进下面的 async move，先取成自有 Arc。
        let block_source = block_source.map(|source| source.0.clone());

        // this env variable is set from the build.rs
        if env!("OPT_LEVEL") == "0" {
            static WARNED: AtomicBool = AtomicBool::new(false);
            if !WARNED.swap(true, atomic::Ordering::Relaxed) {
                warn!(
                    "Azalea was compiled with no optimizations, which may result in significantly reduced pathfinding performance. Consider following the steps at https://azalea.matdoes.dev/azalea/#optimization for faster performance in debug mode."
                )
            }
        }

        let goto_id_atomic = pathfinder.goto_id.clone();
        let calculation_id = if let Some(calculation_id) = event.calculation_id {
            let current_id = goto_id_atomic.load(atomic::Ordering::SeqCst);
            if calculation_id != current_id {
                debug!(
                    "discarding queued goto from obsolete calculation {calculation_id}; current calculation is {current_id}"
                );
                continue;
            }
            calculation_id
        } else {
            if let Some(queued_id) = pathfinder.queued_goto_id {
                // Messages are consumed in FIFO order. An unstamped message
                // observed while a stamped request is still queued must have
                // been written before that stamped request, so it cannot be
                // allowed to allocate a newer generation and supersede it.
                debug!("discarding unstamped goto queued before stamped calculation {queued_id}");
                continue;
            }
            // Raw GotoEvent::new messages have no generation until they reach
            // this listener. Preserve that public compatibility path while
            // client APIs reserve their generation before queueing.
            next_path_calculation_id(&goto_id_atomic)
        };

        // Clear only the request this listener is about to consume. A stale
        // message must never erase the marker belonging to its replacement.
        if pathfinder.queued_goto_id == Some(calculation_id) {
            pathfinder.queued_goto_id = None;
        }

        let (Some(position), Some(world_name), Some(inventory)) = (position, world_name, inventory)
        else {
            // The message has been consumed and cannot be retried. Leave a
            // terminal state rather than a permanent "queued" marker or an
            // obsolete calculation that can no longer produce a usable path.
            warn!("got goto event for an entity missing position, world name, or inventory");
            pathfinder.goal = None;
            pathfinder.opts = None;
            pathfinder.is_calculating = false;
            commands.entity(event.entity).remove::<ComputePath>();
            commands.entity(event.entity).remove::<ExecutingPath>();
            walk_events.write(StartWalkEvent {
                entity: event.entity,
                direction: WalkDirection::None,
            });
            continue;
        };

        let cur_pos = player_pos_to_block_pos(**position);

        if event.goal.success(cur_pos) {
            // we're already at the goal, nothing to do
            pathfinder.goal = None;
            pathfinder.opts = None;
            pathfinder.is_calculating = false;
            commands.entity(event.entity).remove::<ComputePath>();
            if executing_path.is_some() {
                commands.entity(event.entity).remove::<ExecutingPath>();
                walk_events.write(StartWalkEvent {
                    entity: event.entity,
                    direction: WalkDirection::None,
                });
            }
            debug!("already at goal, not pathfinding");
            continue;
        }

        // we store the goal so it can be recalculated later if necessary
        pathfinder.goal = Some(event.goal.clone());
        pathfinder.opts = Some(event.opts.clone());
        pathfinder.is_calculating = true;

        let world_lock = worlds
            .get(world_name)
            .expect("Entity tried to pathfind but the entity isn't in a valid world");

        let goal = event.goal.clone();
        let entity = event.entity;

        let allow_mining = event.opts.allow_mining;
        let inventory_menu = if allow_mining {
            Some(inventory.inventory_menu.clone())
        } else {
            None
        };

        let custom_state = custom_state.cloned().unwrap_or_default();
        let opts = event.opts.clone();

        // if we're executing a path, this might get replaced with something else
        let mut start = cur_pos;

        if let Some(mut executing_path) = executing_path {
            // first try calculating the path instantly, which allows us to react quickly
            // for easy paths (but we'll fall back to spawning a thread if this fails)

            // first, try starting at the node that we're going to
            let instant_path_start = simulating_path_state
                .and_then(|s| s.as_simulated().map(|s| s.target))
                .unwrap_or_else(|| {
                    executing_path
                        .path
                        .iter()
                        .next()
                        .map(|e| e.movement.target)
                        .unwrap_or(cur_pos)
                });

            let path_found_event = calculate_path_at_generation(
                CalculatePathCtx {
                    entity,
                    start: instant_path_start,
                    goal: goal.clone(),
                    world_lock: world_lock.clone(),
                    goto_id_atomic: goto_id_atomic.clone(),
                    mining_cache: MiningCache::new(inventory_menu.clone()),
                    custom_state: custom_state.clone(),
                    block_source: block_source.clone(),
                    opts: PathfinderOpts {
                        min_timeout: PathfinderTimeout::Nodes(2_000),
                        max_timeout: PathfinderTimeout::Nodes(2_000),
                        ..opts
                    },
                },
                calculation_id,
            );

            if let Some(path_found_event) = path_found_event
                && !path_found_event.is_partial
            {
                debug!("Found path instantly!");

                // instant_path_start needs to be equal to executing_path.path.back() for the
                // path merging in path_found_listener to work correctly
                let instant_path_start_index = executing_path
                    .path
                    .iter()
                    .position(|e| e.movement.target == instant_path_start);
                if let Some(instant_path_start_index) = instant_path_start_index {
                    let truncate_to_len = instant_path_start_index + 1;
                    debug!("truncating to {truncate_to_len} for instant path");
                    executing_path.path.truncate(truncate_to_len);

                    path_found_events.write(path_found_event);

                    // we found the path instantly, so we're done here :)
                    continue;
                } else {
                    warn!(
                        "we just calculated an instant path, but the start of it isn't in the current path? instant_path_start: {instant_path_start:?}, simulating_path_state: {simulating_path_state:?}, executing_path.path: {:?}",
                        executing_path.path
                    )
                }
            }

            if !executing_path.path.is_empty() {
                // if we're currently pathfinding and got a goto event, start a little ahead

                let executing_path_limit = 50;

                // truncate the executing path so we can cleanly combine the two paths later
                executing_path.path.truncate(executing_path_limit);

                start = executing_path
                    .path
                    .back()
                    .expect("path was just checked to not be empty")
                    .movement
                    .target;
            }
        }

        if start == cur_pos {
            info!("got goto {:?}, starting from {start:?}", event.goal);
        } else {
            info!(
                "got goto {:?}, starting from {start:?} (currently at {cur_pos:?})",
                event.goal,
            );
        }

        let mining_cache = MiningCache::new(inventory_menu);
        let task = thread_pool.spawn(async move {
            calculate_path_at_generation(
                CalculatePathCtx {
                    entity,
                    start,
                    goal,
                    world_lock,
                    goto_id_atomic,
                    mining_cache,
                    custom_state,
                    block_source,
                    opts,
                },
                calculation_id,
            )
        });

        commands.entity(event.entity).insert(ComputePath(task));
    }
}

/// Convert a player position to a block position, used internally in the
/// pathfinder.
///
/// This is almost the same as `BlockPos::from(position)`, except that non-full
/// blocks are handled correctly.
#[inline]
pub fn player_pos_to_block_pos(position: Vec3) -> BlockPos {
    // 0.5 to account for non-full blocks
    BlockPos::from(position.up(0.5))
}

// Reserve a generation before dispatching path calculation work. Allocating on
// the caller's thread makes request ordering independent of worker scheduling.
fn next_path_calculation_id(goto_id_atomic: &AtomicUsize) -> usize {
    goto_id_atomic
        .fetch_add(1, atomic::Ordering::SeqCst)
        .wrapping_add(1)
}

/// Reserve and expose the generation of a goto before its message is queued.
fn reserve_queued_goto(pathfinder: &mut Pathfinder) -> usize {
    let calculation_id = next_path_calculation_id(&pathfinder.goto_id);
    pathfinder.queued_goto_id = Some(calculation_id);
    calculation_id
}

pub struct CalculatePathCtx {
    pub entity: Entity,
    pub start: BlockPos,
    pub goal: Arc<dyn Goal>,
    pub world_lock: Arc<RwLock<azalea_world::World>>,
    pub goto_id_atomic: Arc<AtomicUsize>,
    pub mining_cache: MiningCache,
    pub custom_state: CustomPathfinderState,
    /// `None` plans over the loaded world; `Some` plans over that source only.
    pub block_source: Option<Arc<dyn world::BlockSource>>,

    pub opts: PathfinderOpts,
}

/// Calculate the [`PathFoundEvent`] for the given pathfinder options.
///
/// You usually want to just use [`PathfinderClientExt::goto`] or send a
/// [`GotoEvent`] instead of calling this directly.
///
/// You are expected to immediately send the `PathFoundEvent` you received after
/// calling this function. `None` will be returned if the pathfinding was
/// interrupted by another path calculation.
pub fn calculate_path(ctx: CalculatePathCtx) -> Option<PathFoundEvent> {
    let calculation_id = next_path_calculation_id(&ctx.goto_id_atomic);
    calculate_path_at_generation(ctx, calculation_id)
}

fn calculate_path_at_generation(
    ctx: CalculatePathCtx,
    calculation_id: usize,
) -> Option<PathFoundEvent> {
    debug!("start: {}", ctx.start);

    let origin = ctx.start;
    let cached_world = match ctx.block_source {
        Some(source) => CachedWorld::new(ctx.world_lock, origin).with_block_source(source),
        None => CachedWorld::new(ctx.world_lock, origin),
    };
    let successors = |pos: RelBlockPos| {
        call_successors_fn(
            &cached_world,
            &ctx.mining_cache,
            &ctx.custom_state.0.read(),
            ctx.opts.successors_fn,
            pos,
        )
    };

    let start_time = Instant::now();

    let astar::Path {
        movements,
        is_partial,
        cost,
    } = a_star(
        RelBlockPos::get_origin(origin),
        |n| ctx.goal.heuristic(n.apply(origin)),
        successors,
        |n| ctx.goal.success(n.apply(origin)),
        ctx.opts.min_timeout,
        ctx.opts.max_timeout,
    );
    let end_time = Instant::now();
    debug!("partial: {is_partial:?}, cost: {cost}");
    let duration = end_time - start_time;
    if is_partial {
        if movements.is_empty() {
            info!("Pathfinder took {duration:?} (empty path)");
        } else {
            info!("Pathfinder took {duration:?} (incomplete path)");
        }
        // wait a bit so it's not a busy loop
        thread::sleep(Duration::from_millis(100));
    } else {
        info!("Pathfinder took {duration:?}");
    }

    debug!("Path:");
    for movement in &movements {
        debug!("  {}", movement.target.apply(origin));
    }

    let path = movements.into_iter().collect::<VecDeque<_>>();

    let goto_id_now = ctx.goto_id_atomic.load(atomic::Ordering::SeqCst);
    if calculation_id != goto_id_now {
        // we must've done another goto while calculating this path, so throw it away
        warn!("finished calculating a path, but it's outdated");
        return None;
    }

    if path.is_empty() && is_partial {
        debug!("this path is empty, we might be stuck :(");
    }

    let mut mapped_path = VecDeque::with_capacity(path.len());
    let mut current_position = RelBlockPos::get_origin(origin);
    for movement in path {
        let mut found_edge = None;
        for edge in successors(current_position) {
            if edge.movement.target == movement.target {
                found_edge = Some(edge);
                break;
            }
        }

        let found_edge = found_edge.expect(
            "path should always still be possible because we're using the same world cache",
        );
        current_position = found_edge.movement.target;

        // we don't just clone the found_edge because we're using BlockPos instead of
        // RelBlockPos as the target type
        mapped_path.push_back(Edge {
            movement: astar::Movement {
                target: movement.target.apply(origin),
                data: movement.data,
            },
            cost: found_edge.cost,
        });
    }

    Some(PathFoundEvent {
        entity: ctx.entity,
        calculation_id,
        start: ctx.start,
        path: Some(mapped_path),
        is_partial,
        successors_fn: ctx.opts.successors_fn,
        allow_mining: ctx.opts.allow_mining,
    })
}

// poll the tasks and send the PathFoundEvent if they're done
pub fn handle_tasks(
    mut commands: Commands,
    mut transform_tasks: Query<(Entity, &mut ComputePath)>,
    mut path_found_events: MessageWriter<PathFoundEvent>,
) {
    for (entity, mut task) in &mut transform_tasks {
        if let Some(optional_path_found_event) = future::block_on(future::poll_once(&mut task.0)) {
            if let Some(path_found_event) = optional_path_found_event {
                path_found_events.write(path_found_event);
            }

            // Task is complete, so remove task component from entity
            commands.entity(entity).remove::<ComputePath>();
        }
    }
}

// set the path for the target entity when we get the PathFoundEvent
#[allow(clippy::type_complexity)]
pub fn path_found_listener(
    mut events: MessageReader<PathFoundEvent>,
    mut query: Query<(
        &mut Pathfinder,
        Option<&mut ExecutingPath>,
        &WorldName,
        &Inventory,
        Option<&CustomPathfinderState>,
        Option<&world::PathfinderBlockSource>,
    )>,
    worlds: Res<Worlds>,
    mut commands: Commands,
    mut walk_events: MessageWriter<StartWalkEvent>,
) {
    for event in events.read() {
        let Ok((mut pathfinder, executing_path, world_name, inventory, custom_state, block_source)) =
            query.get_mut(event.entity)
        else {
            debug!("got path found event for an entity that can't pathfind");
            continue;
        };
        if event.calculation_id != pathfinder.goto_id.load(atomic::Ordering::SeqCst) {
            debug!(
                "discarding path result from obsolete calculation {}",
                event.calculation_id
            );
            continue;
        }
        if let Some(found_path) = &event.path {
            if found_path.is_empty() {
                debug!("calculated path is empty");
                let should_retry = event.is_partial
                    && pathfinder
                        .opts
                        .as_ref()
                        .is_some_and(|opts| opts.retry_on_no_path);
                if let Some(mut executing_path) = executing_path {
                    // An empty continuation must not queue a copy of the path
                    // that is already being executed. Doing so can leave an
                    // empty ExecutingPath component that never makes progress.
                    executing_path.queued_path = None;
                    executing_path.is_path_partial = should_retry;
                    if executing_path.path.is_empty() {
                        commands.entity(event.entity).remove::<ExecutingPath>();
                        walk_events.write(StartWalkEvent {
                            entity: event.entity,
                            direction: WalkDirection::None,
                        });
                        if !should_retry {
                            pathfinder.goal = None;
                            pathfinder.opts = None;
                        }
                    } else if event.is_partial && !should_retry {
                        // The replacement request exhausted its graph. Finish
                        // only the movement already in flight; continuing the
                        // retained prefix (up to 50 nodes) would execute a path
                        // that no longer belongs to an active goal.
                        pathfinder.goal = None;
                        executing_path.queued_path = Some(VecDeque::new());
                    }
                } else if !should_retry {
                    pathfinder.goal = None;
                    pathfinder.opts = None;
                }
                pathfinder.is_calculating = false;
                continue;
            }
            if let Some(mut executing_path) = executing_path {
                let mut new_path = VecDeque::new();

                // combine the old and new paths if the first node of the new path is a
                // successor of the last node of the old path
                if let Some(last_node_of_current_path) = executing_path.path.back() {
                    let world_lock = worlds
                        .get(world_name)
                        .expect("Entity tried to pathfind but the entity isn't in a valid world");
                    let origin = event.start;
                    let successors_fn: moves::SuccessorsFn = event.successors_fn;
                    let cached_world = match block_source {
                        Some(source) => {
                            CachedWorld::new(world_lock, origin).with_block_source(source.0.clone())
                        }
                        None => CachedWorld::new(world_lock, origin),
                    };
                    let mining_cache = MiningCache::new(if event.allow_mining {
                        Some(inventory.inventory_menu.clone())
                    } else {
                        None
                    });
                    let custom_state = custom_state.cloned().unwrap_or_default();
                    let custom_state_ref = custom_state.0.read();
                    let successors = |pos: RelBlockPos| {
                        call_successors_fn(
                            &cached_world,
                            &mining_cache,
                            &custom_state_ref,
                            successors_fn,
                            pos,
                        )
                    };

                    let first_node_of_new_path = found_path
                        .front()
                        .expect("empty paths are handled before path merging");
                    let last_target_of_current_path =
                        RelBlockPos::from_origin(origin, last_node_of_current_path.movement.target);
                    let first_target_of_new_path =
                        RelBlockPos::from_origin(origin, first_node_of_new_path.movement.target);

                    if successors(last_target_of_current_path)
                        .iter()
                        .any(|edge| edge.movement.target == first_target_of_new_path)
                    {
                        debug!("combining old and new paths");
                        debug!(
                            "old path: {:?}",
                            executing_path.path.iter().collect::<Vec<_>>()
                        );
                        debug!(
                            "new path: {:?}",
                            found_path.iter().take(10).collect::<Vec<_>>()
                        );
                        new_path.extend(executing_path.path.iter().cloned());
                    }
                }

                new_path.extend(found_path.to_owned());

                debug!(
                    "set queued path to {:?}",
                    new_path.iter().take(10).collect::<Vec<_>>()
                );
                executing_path.queued_path = Some(new_path);
                executing_path.is_path_partial = event.is_partial;
            } else {
                commands.entity(event.entity).insert(ExecutingPath {
                    path: found_path.to_owned(),
                    queued_path: None,
                    last_reached_node: event.start,
                    ticks_since_last_node_reached: 0,
                    is_path_partial: event.is_partial,
                });
                debug!(
                    "set path to {:?}",
                    found_path.iter().take(10).collect::<Vec<_>>()
                );
                debug!("partial: {}", event.is_partial);
            }
        } else {
            error!("No path found");
            if let Some(mut executing_path) = executing_path {
                // set the queued path so we don't stop in the middle of a move
                executing_path.queued_path = Some(VecDeque::new());
            } else {
                // wasn't executing a path, don't need to do anything
            }
        }
        pathfinder.is_calculating = false;
    }
}

/// Apply the force-retirement invariant directly to an ECS world.
///
/// Both the synchronous client API and the buffered stop handler use this
/// helper so they cannot drift on generation, component, or movement cleanup.
fn force_retire_pathfinding_in_world(world: &mut bevy_ecs::world::World, entity: Entity) {
    {
        let Ok(mut entity_mut) = world.get_entity_mut(entity) else {
            return;
        };
        if let Some(mut pathfinder) = entity_mut.get_mut::<Pathfinder>() {
            execute::retire_pathfinder_request(&mut pathfinder);
        }
        entity_mut.remove::<ComputePath>();
        entity_mut.remove::<ExecutingPath>();
    }

    world.write_message(StartWalkEvent {
        entity,
        direction: WalkDirection::None,
    });
}

#[derive(Message)]
pub struct StopPathfindingEvent {
    pub entity: Entity,
    /// Whether we should stop moving immediately without waiting for the
    /// current movement to finish.
    ///
    /// This is usually set to false, since it might cause the bot to fall if it
    /// was in the middle of parkouring.
    pub force: bool,
}

pub fn handle_stop_pathfinding_event(
    mut events: MessageReader<StopPathfindingEvent>,
    mut query: Query<(&mut Pathfinder, Option<&mut ExecutingPath>)>,
    mut walk_events: MessageWriter<StartWalkEvent>,
    mut commands: Commands,
) {
    for event in events.read() {
        if event.force {
            let entity = event.entity;
            commands.queue(move |world: &mut bevy_ecs::world::World| {
                force_retire_pathfinding_in_world(world, entity);
            });
            continue;
        }

        // stop computing any path that's being computed
        commands.entity(event.entity).remove::<ComputePath>();

        let Ok((mut pathfinder, executing_path)) = query.get_mut(event.entity) else {
            continue;
        };
        // Invalidate workers as well as the task component. A worker may have
        // completed between its final check and PathFoundEvent consumption.
        next_path_calculation_id(&pathfinder.goto_id);
        pathfinder.queued_goto_id = None;
        pathfinder.goal = None;
        pathfinder.is_calculating = false;

        match executing_path {
            Some(mut executing_path) if !executing_path.path.is_empty() => {
                // Finish the current movement before switching to an empty
                // path. Keep opts until then because local patching still uses
                // them.
                executing_path.queued_path = Some(VecDeque::new());
                executing_path.is_path_partial = false;
            }
            Some(_) => {
                pathfinder.opts = None;
                commands.entity(event.entity).remove::<ExecutingPath>();
                walk_events.write(StartWalkEvent {
                    entity: event.entity,
                    direction: WalkDirection::None,
                });
            }
            None => {
                pathfinder.opts = None;
                // There is no movement left to finish gracefully, so both
                // forceful and graceful stops must clear any latched input.
                walk_events.write(StartWalkEvent {
                    entity: event.entity,
                    direction: WalkDirection::None,
                });
            }
        }
    }
}

#[allow(clippy::type_complexity)]
pub fn stop_pathfinding_on_world_change(
    mut query: Query<
        (
            Entity,
            Ref<WorldName>,
            Option<&ExecutingPath>,
            Option<&ComputePath>,
            &Pathfinder,
        ),
        Changed<WorldName>,
    >,
    mut stop_pathfinding_events: MessageWriter<StopPathfindingEvent>,
) {
    for (entity, world_name, executing_path, compute_path, pathfinder) in &mut query {
        // `Changed` also matches a component's initial insertion. A newly
        // joined player has not crossed dimensions and must not have a goto
        // from the same update canceled as a false world change.
        if world_name.is_added() {
            continue;
        }
        let has_active_navigation = executing_path.is_some()
            || compute_path.is_some()
            || pathfinder.queued_goto_id.is_some()
            || pathfinder.goal.is_some()
            || pathfinder.is_calculating;
        if !has_active_navigation {
            continue;
        }
        debug!("world changed, stopping pathfinding");
        stop_pathfinding_events.write(StopPathfindingEvent {
            entity,
            force: true,
        });
    }
}

pub fn call_successors_fn(
    cached_world: &CachedWorld,
    mining_cache: &MiningCache,
    custom_state: &CustomPathfinderStateRef,
    successors_fn: SuccessorsFn,
    pos: RelBlockPos,
) -> Vec<astar::Edge<RelBlockPos, moves::MoveData>> {
    let mut edges = Vec::with_capacity(16);
    let mut ctx = MovesCtx {
        edges: &mut edges,
        world: cached_world,
        mining_cache,
        custom_state,
    };
    successors_fn(&mut ctx, pos);
    edges
}
