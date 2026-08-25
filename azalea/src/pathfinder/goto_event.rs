use std::{sync::Arc, time::Duration};

use bevy_ecs::prelude::*;

use crate::pathfinder::{
    astar::PathfinderTimeout,
    goals::Goal,
    moves::{self, SuccessorsFn},
};

/// Send this event to start pathfinding to the given goal.
///
/// Also see [`PathfinderClientExt::goto`].
///
/// This event is read by [`goto_listener`]
///
/// [`goto_listener`]: crate::pathfinder::goto_listener
/// [`PathfinderClientExt::goto`]: crate::pathfinder::PathfinderClientExt::goto
#[derive(Message)]
#[non_exhaustive]
pub struct GotoEvent {
    /// The local bot entity that will do the pathfinding and execute the path.
    pub entity: Entity,
    pub goal: Arc<dyn Goal>,
    pub opts: PathfinderOpts,
    /// Generation reserved by high-level entry points before this message is
    /// queued. Direct [`GotoEvent::new`] callers leave this unstamped so the
    /// listener can allocate a generation for backwards compatibility.
    pub(crate) calculation_id: Option<usize>,
}

impl GotoEvent {
    /// Build an unstamped goto message.
    ///
    /// Its generation is allocated when
    /// [`goto_listener`](crate::pathfinder::goto_listener) consumes it.
    /// Prefer the [`Client`](crate::Client) pathfinder methods when request
    /// ordering must already be fixed while messages are still queued.
    pub fn new(entity: Entity, goal: impl Goal + 'static, opts: PathfinderOpts) -> Self {
        Self {
            entity,
            goal: Arc::new(goal),
            opts,
            calculation_id: None,
        }
    }
}

/// Configuration options that the pathfinder will use when calculating and
/// executing a path.
///
/// This can be passed into [`Client::goto_with_opts`] or
/// [`Client::start_goto_with_opts`].
///
/// ```
/// # use azalea::pathfinder::{moves, PathfinderOpts};
/// // example config to disallow mining blocks and to not do parkour
/// let opts = PathfinderOpts::new()
///     .allow_mining(false)
///     .successors_fn(moves::basic::basic_move);
/// ```
///
/// [`Client::goto_with_opts`]: super::PathfinderClientExt::goto_with_opts
/// [`Client::start_goto_with_opts`]: super::PathfinderClientExt::start_goto_with_opts
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct PathfinderOpts {
    pub(crate) successors_fn: SuccessorsFn,
    pub(crate) allow_mining: bool,
    pub(crate) retry_on_no_path: bool,
    pub(crate) recalculate_partial_paths: bool,
    pub(crate) min_timeout: PathfinderTimeout,
    pub(crate) max_timeout: PathfinderTimeout,
}

impl PathfinderOpts {
    pub const fn new() -> Self {
        Self {
            successors_fn: moves::default_move,
            allow_mining: true,
            retry_on_no_path: true,
            recalculate_partial_paths: true,
            min_timeout: PathfinderTimeout::Time(Duration::from_secs(1)),
            max_timeout: PathfinderTimeout::Time(Duration::from_secs(5)),
        }
    }
    /// Set the function that's used for checking what moves are possible.
    ///
    /// Defaults to [`moves::default_move`].
    pub fn successors_fn(mut self, successors_fn: SuccessorsFn) -> Self {
        self.successors_fn = successors_fn;
        self
    }
    /// Set whether the bot is allowed to break blocks while pathfinding.
    ///
    /// Defaults to `true`.
    pub fn allow_mining(mut self, allow_mining: bool) -> Self {
        self.allow_mining = allow_mining;
        self
    }
    /// Whether we should recalculate the path when the pathfinder timed out and
    /// there's no partial path to try.
    ///
    /// Defaults to `true`.
    pub fn retry_on_no_path(mut self, retry_on_no_path: bool) -> Self {
        self.retry_on_no_path = retry_on_no_path;
        self
    }
    /// Set whether a partial path should automatically start another full-goal
    /// calculation near or after the end of the returned segment.
    ///
    /// This does not disable local obstruction and timeout patching while the
    /// segment is being executed. Defaults to `true`.
    pub fn recalculate_partial_paths(mut self, recalculate_partial_paths: bool) -> Self {
        self.recalculate_partial_paths = recalculate_partial_paths;
        self
    }
    /// The minimum amount of time that should pass before the A* pathfinder
    /// function can return a timeout if it finds a path that seems good enough.
    /// It may take up to [`Self::max_timeout`] if it can't immediately find
    /// a usable path.
    ///
    /// Defaults to `PathfinderTimeout::Time(Duration::from_secs(1))`.
    ///
    /// Also see [`PathfinderTimeout::Nodes`]
    pub fn min_timeout(mut self, min_timeout: impl Into<PathfinderTimeout>) -> Self {
        self.min_timeout = min_timeout.into();
        self
    }
    /// The absolute maximum amount of time that the pathfinder function can
    /// take to find a path.
    ///
    /// If it takes this long, it means no usable path was found (so it might be
    /// impossible).
    ///
    /// Defaults to `PathfinderTimeout::Time(Duration::from_secs(5))`.
    pub fn max_timeout(mut self, max_timeout: impl Into<PathfinderTimeout>) -> Self {
        self.max_timeout = max_timeout.into();
        self
    }
}
impl Default for PathfinderOpts {
    fn default() -> Self {
        Self::new()
    }
}
