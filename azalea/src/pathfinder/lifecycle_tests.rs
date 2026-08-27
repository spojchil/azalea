use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use azalea_client::StartWalkEvent;
use azalea_core::{position::BlockPos, registry_holder::RegistryHolder};
use azalea_entity::{Position, inventory::Inventory};
use azalea_world::{WorldName, Worlds};
use bevy_app::{App, Update};
use bevy_ecs::prelude::*;
use bevy_tasks::{AsyncComputeTaskPool, TaskPoolBuilder};
use parking_lot::RwLock;

use super::{
    ComputePath, ExecutingPath, PathFoundEvent, Pathfinder, PathfinderClientExt, PathfinderOpts,
    StopPathfindingEvent,
    astar::{Edge, Movement},
    execute::{
        finish_timeout_patch, graceful_stop_pending,
        patching::{PatchOutcome, check_for_path_obstruction},
        recalculate_if_has_goal_but_no_path, recalculate_near_end_of_path, restore_graceful_stop,
        retire_pathfinder_request, timeout_movement,
    },
    goals::{BlockPosGoal, Goal},
    goto_listener, handle_stop_pathfinding_event, next_path_calculation_id, path_found_listener,
    reserve_queued_goto, stop_pathfinding_on_world_change,
};
use crate::{
    Client, WalkDirection,
    pathfinder::moves::{
        self, MoveData,
        basic::{descend_is_reached, execute_descend_move},
    },
};

#[derive(Debug)]
struct CountingGoal {
    target: BlockPos,
    success_calls: Arc<AtomicUsize>,
}

impl Goal for CountingGoal {
    fn heuristic(&self, n: BlockPos) -> f32 {
        BlockPosGoal(self.target).heuristic(n)
    }

    fn success(&self, n: BlockPos) -> bool {
        self.success_calls.fetch_add(1, Ordering::SeqCst);
        n == self.target
    }
}

fn queued_goto_listener_app() -> (App, Entity, BlockPos) {
    AsyncComputeTaskPool::get_or_init(|| {
        TaskPoolBuilder::default()
            .num_threads(1)
            .thread_name("Pathfinder queued-goto test pool".to_owned())
            .build()
    });

    let current = BlockPos::new(0, 64, 0);
    let mut app = App::new();
    app.add_message::<super::GotoEvent>()
        .add_message::<PathFoundEvent>()
        .add_message::<StartWalkEvent>()
        .init_resource::<Worlds>()
        .add_systems(Update, goto_listener);
    let entity = app
        .world_mut()
        .spawn((
            Pathfinder::default(),
            Position::new(current.center_bottom()),
            WorldName::new("minecraft:overworld"),
            Inventory::default(),
        ))
        .id();
    (app, entity, current)
}

fn retain_overworld(app: &mut App) -> Arc<RwLock<azalea_world::World>> {
    app.world_mut().resource_mut::<Worlds>().get_or_insert(
        WorldName::new("minecraft:overworld"),
        384,
        -64,
        &RegistryHolder::default(),
    )
}

fn detach_app_world(app: &mut App) -> Arc<RwLock<bevy_ecs::world::World>> {
    Arc::new(RwLock::new(std::mem::take(app.world_mut())))
}

fn restore_app_world(app: &mut App, ecs: Arc<RwLock<bevy_ecs::world::World>>) {
    let ecs = match Arc::try_unwrap(ecs) {
        Ok(ecs) => ecs,
        Err(_) => panic!("test client must release its only ECS handle"),
    };
    *app.world_mut() = ecs.into_inner();
}

fn active_pathfinder() -> Pathfinder {
    Pathfinder {
        goal: Some(Arc::new(BlockPosGoal(BlockPos::new(8, 64, 0)))),
        opts: Some(PathfinderOpts::new()),
        is_calculating: true,
        ..Default::default()
    }
}

fn one_edge_path() -> VecDeque<Edge<BlockPos, MoveData>> {
    VecDeque::from([Edge {
        movement: Movement {
            target: BlockPos::new(1, 64, 0),
            data: MoveData::new(&execute_descend_move, &descend_is_reached),
        },
        cost: 1.0,
    }])
}

fn straight_path(len: usize) -> VecDeque<Edge<BlockPos, MoveData>> {
    (1..=len)
        .map(|x| Edge {
            movement: Movement {
                target: BlockPos::new(x as i32, 64, 0),
                data: MoveData::new(&execute_descend_move, &descend_is_reached),
            },
            cost: 1.0,
        })
        .collect()
}

#[derive(Resource, Default)]
struct ObservedMessages {
    walk_stops: usize,
    gotos: usize,
}

fn observe_messages(
    mut walks: MessageReader<StartWalkEvent>,
    mut gotos: MessageReader<super::GotoEvent>,
    mut observed: ResMut<ObservedMessages>,
) {
    observed.walk_stops += walks
        .read()
        .filter(|event| event.direction == WalkDirection::None)
        .count();
    observed.gotos += gotos.read().count();
}

#[test]
fn stop_retires_a_calculation_without_an_executing_path() {
    let mut app = App::new();
    app.add_message::<StopPathfindingEvent>()
        .add_message::<StartWalkEvent>()
        .add_message::<super::GotoEvent>()
        .init_resource::<ObservedMessages>()
        .add_systems(
            Update,
            (handle_stop_pathfinding_event, observe_messages).chain(),
        );
    let entity = app.world_mut().spawn(active_pathfinder()).id();
    let before = app
        .world()
        .get::<Pathfinder>(entity)
        .unwrap()
        .goto_id
        .load(Ordering::SeqCst);
    app.world_mut().write_message(StopPathfindingEvent {
        entity,
        force: true,
    });

    app.update();

    let pathfinder = app.world().get::<Pathfinder>(entity).unwrap();
    assert!(pathfinder.goal.is_none());
    assert!(pathfinder.opts.is_none());
    assert!(!pathfinder.is_calculating);
    assert_eq!(pathfinder.goto_id.load(Ordering::SeqCst), before + 1);
    assert_eq!(app.world().resource::<ObservedMessages>().walk_stops, 1);
}

#[test]
fn graceful_stop_without_an_executing_path_stops_latched_movement() {
    let mut app = App::new();
    app.add_message::<StopPathfindingEvent>()
        .add_message::<StartWalkEvent>()
        .add_message::<super::GotoEvent>()
        .init_resource::<ObservedMessages>()
        .add_systems(
            Update,
            (handle_stop_pathfinding_event, observe_messages).chain(),
        );
    let entity = app.world_mut().spawn(active_pathfinder()).id();
    app.world_mut().write_message(StopPathfindingEvent {
        entity,
        force: false,
    });

    app.update();

    let pathfinder = app.world().get::<Pathfinder>(entity).unwrap();
    assert!(pathfinder.goal.is_none());
    assert!(pathfinder.opts.is_none());
    assert!(!pathfinder.is_calculating);
    assert_eq!(app.world().resource::<ObservedMessages>().walk_stops, 1);
}

#[test]
fn stale_path_result_cannot_revive_a_stopped_or_replaced_request() {
    let mut app = App::new();
    app.add_message::<PathFoundEvent>()
        .add_message::<StartWalkEvent>()
        .init_resource::<Worlds>()
        .add_systems(Update, path_found_listener);
    let pathfinder = active_pathfinder();
    pathfinder.goto_id.store(2, Ordering::SeqCst);
    let entity = app
        .world_mut()
        .spawn((
            pathfinder,
            WorldName::new("minecraft:overworld"),
            Inventory::default(),
        ))
        .id();
    app.world_mut().write_message(PathFoundEvent {
        entity,
        calculation_id: 1,
        start: BlockPos::new(0, 64, 0),
        path: Some(one_edge_path()),
        is_partial: false,
        successors_fn: moves::default_move,
        allow_mining: false,
    });

    app.update();

    let pathfinder = app.world().get::<Pathfinder>(entity).unwrap();
    assert!(pathfinder.goal.is_some());
    assert!(pathfinder.opts.is_some());
    assert!(pathfinder.is_calculating);
    assert!(app.world().get::<ExecutingPath>(entity).is_none());
}

#[test]
fn reaching_a_goal_invalidates_an_in_flight_result() {
    let mut pathfinder = active_pathfinder();
    let calculation_id = next_path_calculation_id(&pathfinder.goto_id);

    retire_pathfinder_request(&mut pathfinder);

    assert!(pathfinder.goal.is_none());
    assert!(pathfinder.opts.is_none());
    assert!(!pathfinder.is_calculating);
    assert_ne!(
        pathfinder.goto_id.load(Ordering::SeqCst),
        calculation_id,
        "a result emitted by the in-flight worker must now be stale"
    );
}

#[test]
fn calculation_ids_are_reserved_in_request_order() {
    let pathfinder = Pathfinder::default();
    let older = next_path_calculation_id(&pathfinder.goto_id);
    let newer = next_path_calculation_id(&pathfinder.goto_id);

    assert!(older < newer);
    assert_eq!(pathfinder.goto_id.load(Ordering::SeqCst), newer);
}

#[test]
fn an_already_satisfied_client_goto_clears_its_queued_marker() {
    let (mut app, entity, current) = queued_goto_listener_app();
    let ecs = detach_app_world(&mut app);
    let client = Client::new(entity, ecs.clone());
    let success_calls = Arc::new(AtomicUsize::new(0));

    client.start_goto_with_opts(
        CountingGoal {
            target: current,
            success_calls: success_calls.clone(),
        },
        PathfinderOpts::new(),
    );
    assert!(
        ecs.read()
            .get::<Pathfinder>(entity)
            .unwrap()
            .queued_goto_id
            .is_some()
    );
    drop(client);
    restore_app_world(&mut app, ecs);

    app.update();

    assert_eq!(success_calls.load(Ordering::SeqCst), 1);
    let pathfinder = app.world().get::<Pathfinder>(entity).unwrap();
    assert!(pathfinder.queued_goto_id.is_none());
    assert!(pathfinder.goal.is_none());
    assert!(pathfinder.opts.is_none());
    assert!(!pathfinder.is_calculating);
    assert!(app.world().get::<ComputePath>(entity).is_none());
    assert!(app.world().get::<ExecutingPath>(entity).is_none());
}

#[test]
fn a_stale_goto_does_not_clear_the_queued_marker_of_its_replacement() {
    let (mut app, entity, current) = queued_goto_listener_app();
    let success_calls = Arc::new(AtomicUsize::new(0));
    let older_id = {
        let mut pathfinder = app.world_mut().get_mut::<Pathfinder>(entity).unwrap();
        reserve_queued_goto(&mut pathfinder)
    };
    app.world_mut().write_message(super::GotoEvent {
        entity,
        goal: Arc::new(CountingGoal {
            target: current,
            success_calls: success_calls.clone(),
        }),
        opts: PathfinderOpts::new(),
        calculation_id: Some(older_id),
    });
    let replacement_id = {
        let mut pathfinder = app.world_mut().get_mut::<Pathfinder>(entity).unwrap();
        reserve_queued_goto(&mut pathfinder)
    };

    app.update();

    assert_eq!(success_calls.load(Ordering::SeqCst), 0);
    let pathfinder = app.world().get::<Pathfinder>(entity).unwrap();
    assert_eq!(pathfinder.goto_id.load(Ordering::SeqCst), replacement_id);
    assert_eq!(pathfinder.queued_goto_id, Some(replacement_id));
    assert!(pathfinder.goal.is_none());
    assert!(!pathfinder.is_calculating);
}

#[test]
fn two_client_gotos_queued_before_the_listener_only_apply_the_latest() {
    let (mut app, entity, current) = queued_goto_listener_app();
    let ecs = detach_app_world(&mut app);
    let client = Client::new(entity, ecs.clone());
    let older_calls = Arc::new(AtomicUsize::new(0));
    let latest_calls = Arc::new(AtomicUsize::new(0));

    client.start_goto_with_opts(
        CountingGoal {
            target: current,
            success_calls: older_calls.clone(),
        },
        PathfinderOpts::new(),
    );
    client.start_goto_with_opts(
        CountingGoal {
            target: current,
            success_calls: latest_calls.clone(),
        },
        PathfinderOpts::new(),
    );
    let reserved_id = {
        let ecs = ecs.read();
        let pathfinder = ecs.get::<Pathfinder>(entity).unwrap();
        let reserved_id = pathfinder.goto_id.load(Ordering::SeqCst);
        assert_eq!(pathfinder.queued_goto_id, Some(reserved_id));
        reserved_id
    };
    assert!(
        !client.is_goto_target_reached(),
        "a queued request is observable before the listener arms goal/calculation state"
    );
    drop(client);
    restore_app_world(&mut app, ecs);

    app.update();

    assert_eq!(older_calls.load(Ordering::SeqCst), 0);
    assert_eq!(latest_calls.load(Ordering::SeqCst), 1);
    let pathfinder = app.world().get::<Pathfinder>(entity).unwrap();
    assert_eq!(pathfinder.goto_id.load(Ordering::SeqCst), reserved_id);
    assert!(pathfinder.queued_goto_id.is_none());
    assert!(pathfinder.goal.is_none());
    assert!(pathfinder.opts.is_none());
    assert!(!pathfinder.is_calculating);
    assert!(app.world().get::<ComputePath>(entity).is_none());
    assert!(app.world().get::<ExecutingPath>(entity).is_none());
}

#[test]
fn timeout_does_not_patch_an_old_leg_after_replacement_listener_consumption() {
    let (mut app, entity, _) = queued_goto_listener_app();
    app.add_systems(Update, timeout_movement.after(goto_listener));
    let _world_lock = retain_overworld(&mut app);
    let far_position = BlockPos::new(100, 64, 100);
    *app.world_mut().get_mut::<Position>(entity).unwrap() =
        Position::new(far_position.center_bottom());
    app.world_mut().entity_mut(entity).insert(ExecutingPath {
        path: straight_path(10),
        queued_path: None,
        last_reached_node: BlockPos::new(0, 64, 0),
        ticks_since_last_node_reached: 41,
        is_path_partial: false,
        allow_mining: false,
    });
    let ecs = detach_app_world(&mut app);
    let client = Client::new(entity, ecs.clone());
    client.start_goto_with_opts(
        BlockPosGoal(BlockPos::new(110, 64, 100)),
        PathfinderOpts::new().allow_mining(false),
    );
    let replacement_id = ecs
        .read()
        .get::<Pathfinder>(entity)
        .unwrap()
        .goto_id
        .load(Ordering::SeqCst);
    drop(client);
    restore_app_world(&mut app, ecs);

    app.update();

    let pathfinder = app.world().get::<Pathfinder>(entity).unwrap();
    assert!(pathfinder.queued_goto_id.is_none());
    assert!(pathfinder.is_calculating);
    assert_eq!(pathfinder.goto_id.load(Ordering::SeqCst), replacement_id);
    assert!(app.world().get::<ComputePath>(entity).is_some());
    let executing = app.world().get::<ExecutingPath>(entity).unwrap();
    assert_eq!(executing.path.len(), 10);
    assert_eq!(executing.path.front().unwrap().movement.target.x, 1);
    assert_eq!(executing.ticks_since_last_node_reached, 41);
}

#[test]
fn obstruction_does_not_patch_an_old_leg_after_replacement_listener_consumption() {
    let (mut app, entity, _) = queued_goto_listener_app();
    app.add_systems(Update, check_for_path_obstruction.after(goto_listener));
    let _world_lock = retain_overworld(&mut app);
    app.world_mut().entity_mut(entity).insert(ExecutingPath {
        path: straight_path(10),
        queued_path: None,
        last_reached_node: BlockPos::new(0, 64, 0),
        ticks_since_last_node_reached: 17,
        is_path_partial: false,
        allow_mining: false,
    });
    let ecs = detach_app_world(&mut app);
    let client = Client::new(entity, ecs.clone());
    client.start_goto_with_opts(
        BlockPosGoal(BlockPos::new(20, 64, 0)),
        PathfinderOpts::new().allow_mining(false),
    );
    let replacement_id = ecs
        .read()
        .get::<Pathfinder>(entity)
        .unwrap()
        .goto_id
        .load(Ordering::SeqCst);
    drop(client);
    restore_app_world(&mut app, ecs);

    app.update();

    let pathfinder = app.world().get::<Pathfinder>(entity).unwrap();
    assert!(pathfinder.queued_goto_id.is_none());
    assert!(pathfinder.is_calculating);
    assert_eq!(pathfinder.goto_id.load(Ordering::SeqCst), replacement_id);
    assert!(app.world().get::<ComputePath>(entity).is_some());
    let executing = app.world().get::<ExecutingPath>(entity).unwrap();
    assert_eq!(executing.path.len(), 10);
    assert_eq!(executing.path.front().unwrap().movement.target.x, 1);
    assert_eq!(executing.ticks_since_last_node_reached, 17);
}

#[test]
fn force_retire_synchronously_invalidates_a_queued_client_goto() {
    let (mut app, entity, current) = queued_goto_listener_app();
    *app.world_mut().get_mut::<Pathfinder>(entity).unwrap() = active_pathfinder();
    app.world_mut().entity_mut(entity).insert((
        ExecutingPath {
            path: one_edge_path(),
            queued_path: None,
            last_reached_node: current,
            ticks_since_last_node_reached: 0,
            is_path_partial: false,
            allow_mining: false,
        },
        ComputePath(AsyncComputeTaskPool::get().spawn(async { None })),
    ));
    let ecs = detach_app_world(&mut app);
    let client = Client::new(entity, ecs.clone());
    let success_calls = Arc::new(AtomicUsize::new(0));

    client.start_goto_with_opts(
        CountingGoal {
            target: current,
            success_calls: success_calls.clone(),
        },
        PathfinderOpts::new(),
    );
    let reserved_id = ecs
        .read()
        .get::<Pathfinder>(entity)
        .unwrap()
        .goto_id
        .load(Ordering::SeqCst);
    client.force_retire_pathfinding();
    let retired_id = {
        let ecs = ecs.read();
        let pathfinder = ecs.get::<Pathfinder>(entity).unwrap();
        assert!(pathfinder.queued_goto_id.is_none());
        assert!(pathfinder.goal.is_none());
        assert!(pathfinder.opts.is_none());
        assert!(!pathfinder.is_calculating);
        assert!(ecs.get::<ComputePath>(entity).is_none());
        assert!(ecs.get::<ExecutingPath>(entity).is_none());
        assert_eq!(ecs.resource::<Messages<StartWalkEvent>>().len(), 1);
        pathfinder.goto_id.load(Ordering::SeqCst)
    };
    assert_eq!(retired_id, reserved_id + 1);
    drop(client);
    restore_app_world(&mut app, ecs);

    app.update();

    assert_eq!(success_calls.load(Ordering::SeqCst), 0);
    let pathfinder = app.world().get::<Pathfinder>(entity).unwrap();
    assert_eq!(pathfinder.goto_id.load(Ordering::SeqCst), retired_id);
    assert!(pathfinder.queued_goto_id.is_none());
    assert!(pathfinder.goal.is_none());
    assert!(pathfinder.opts.is_none());
    assert!(!pathfinder.is_calculating);
    assert!(app.world().get::<ComputePath>(entity).is_none());
    assert!(app.world().get::<ExecutingPath>(entity).is_none());
}

#[test]
fn empty_current_path_is_removed_and_stops_walking() {
    let mut app = App::new();
    app.add_message::<PathFoundEvent>()
        .add_message::<StartWalkEvent>()
        .add_message::<super::GotoEvent>()
        .init_resource::<Worlds>()
        .init_resource::<ObservedMessages>()
        .add_systems(Update, (path_found_listener, observe_messages).chain());
    let mut pathfinder = active_pathfinder();
    pathfinder.is_calculating = true;
    let calculation_id = next_path_calculation_id(&pathfinder.goto_id);
    let entity = app
        .world_mut()
        .spawn((
            pathfinder,
            ExecutingPath {
                path: VecDeque::new(),
                queued_path: None,
                last_reached_node: BlockPos::new(0, 64, 0),
                ticks_since_last_node_reached: 0,
                is_path_partial: true,
                allow_mining: false,
            },
            WorldName::new("minecraft:overworld"),
            Inventory::default(),
        ))
        .id();
    app.world_mut().write_message(PathFoundEvent {
        entity,
        calculation_id,
        start: BlockPos::new(0, 64, 0),
        path: Some(VecDeque::new()),
        is_partial: true,
        successors_fn: moves::default_move,
        allow_mining: false,
    });

    app.update();

    assert!(app.world().get::<ExecutingPath>(entity).is_none());
    assert_eq!(app.world().resource::<ObservedMessages>().walk_stops, 1);
}

#[test]
fn empty_partial_result_with_retry_disabled_stops_after_the_current_movement() {
    let mut app = App::new();
    app.add_message::<PathFoundEvent>()
        .add_message::<StartWalkEvent>()
        .init_resource::<Worlds>()
        .add_systems(Update, path_found_listener);
    let mut pathfinder = active_pathfinder();
    pathfinder.opts = Some(PathfinderOpts::new().retry_on_no_path(false));
    let calculation_id = next_path_calculation_id(&pathfinder.goto_id);
    let entity = app
        .world_mut()
        .spawn((
            pathfinder,
            ExecutingPath {
                path: one_edge_path(),
                queued_path: None,
                last_reached_node: BlockPos::new(0, 64, 0),
                ticks_since_last_node_reached: 0,
                is_path_partial: true,
                allow_mining: false,
            },
            WorldName::new("minecraft:overworld"),
            Inventory::default(),
        ))
        .id();
    app.world_mut().write_message(PathFoundEvent {
        entity,
        calculation_id,
        start: BlockPos::new(0, 64, 0),
        path: Some(VecDeque::new()),
        is_partial: true,
        successors_fn: moves::default_move,
        allow_mining: false,
    });

    app.update();

    let pathfinder = app.world().get::<Pathfinder>(entity).unwrap();
    assert!(pathfinder.goal.is_none());
    assert!(
        pathfinder.opts.is_some(),
        "the in-flight movement may still patch"
    );
    assert!(!pathfinder.is_calculating);
    let executing = app.world().get::<ExecutingPath>(entity).unwrap();
    assert!(
        executing
            .queued_path
            .as_ref()
            .is_some_and(VecDeque::is_empty)
    );
    assert!(!executing.is_path_partial);
}

#[test]
fn partial_recalculation_can_be_disabled_without_dropping_patch_opts() {
    let mut app = App::new();
    app.add_message::<super::GotoEvent>()
        .add_message::<StartWalkEvent>()
        .init_resource::<ObservedMessages>()
        .add_systems(
            Update,
            (recalculate_near_end_of_path, observe_messages).chain(),
        );
    let mut pathfinder = active_pathfinder();
    pathfinder.is_calculating = false;
    pathfinder.opts = Some(PathfinderOpts::new().recalculate_partial_paths(false));
    let entity = app
        .world_mut()
        .spawn((
            pathfinder,
            ExecutingPath {
                path: one_edge_path(),
                queued_path: None,
                last_reached_node: BlockPos::new(0, 64, 0),
                ticks_since_last_node_reached: 0,
                is_path_partial: true,
                allow_mining: false,
            },
        ))
        .id();

    app.update();

    let pathfinder = app.world().get::<Pathfinder>(entity).unwrap();
    assert!(pathfinder.goal.is_some());
    assert!(pathfinder.opts.is_some(), "local patching still needs opts");
    assert!(!pathfinder.is_calculating);
    assert!(app.world().get::<ExecutingPath>(entity).is_some());
    assert_eq!(app.world().resource::<ObservedMessages>().gotos, 0);
}

#[test]
fn an_empty_partial_segment_is_retired_when_recalculation_is_disabled() {
    let mut app = App::new();
    app.add_message::<super::GotoEvent>()
        .add_message::<StartWalkEvent>()
        .init_resource::<ObservedMessages>()
        .add_systems(
            Update,
            (recalculate_near_end_of_path, observe_messages).chain(),
        );
    let mut pathfinder = active_pathfinder();
    pathfinder.is_calculating = false;
    pathfinder.opts = Some(PathfinderOpts::new().recalculate_partial_paths(false));
    let entity = app
        .world_mut()
        .spawn((
            pathfinder,
            ExecutingPath {
                path: VecDeque::new(),
                queued_path: None,
                last_reached_node: BlockPos::new(0, 64, 0),
                ticks_since_last_node_reached: 0,
                is_path_partial: true,
                allow_mining: false,
            },
        ))
        .id();

    app.update();

    let pathfinder = app.world().get::<Pathfinder>(entity).unwrap();
    assert!(pathfinder.goal.is_none());
    assert!(pathfinder.opts.is_none());
    assert!(!pathfinder.is_calculating);
    assert!(app.world().get::<ExecutingPath>(entity).is_none());
    assert_eq!(app.world().resource::<ObservedMessages>().walk_stops, 1);
    assert_eq!(app.world().resource::<ObservedMessages>().gotos, 0);
}

#[test]
fn an_empty_old_segment_does_not_clear_a_replacement_calculation() {
    let mut app = App::new();
    app.add_message::<super::GotoEvent>()
        .add_message::<StartWalkEvent>()
        .init_resource::<ObservedMessages>()
        .add_systems(
            Update,
            (recalculate_near_end_of_path, observe_messages).chain(),
        );
    let mut pathfinder = active_pathfinder();
    pathfinder.opts = Some(PathfinderOpts::new().recalculate_partial_paths(false));
    let entity = app
        .world_mut()
        .spawn((
            pathfinder,
            ExecutingPath {
                path: VecDeque::new(),
                queued_path: None,
                last_reached_node: BlockPos::new(0, 64, 0),
                ticks_since_last_node_reached: 0,
                is_path_partial: true,
                allow_mining: false,
            },
        ))
        .id();

    app.update();

    let pathfinder = app.world().get::<Pathfinder>(entity).unwrap();
    assert!(pathfinder.goal.is_some());
    assert!(pathfinder.opts.is_some());
    assert!(pathfinder.is_calculating);
    assert!(app.world().get::<ExecutingPath>(entity).is_none());
    assert_eq!(app.world().resource::<ObservedMessages>().walk_stops, 1);
}

#[test]
fn an_empty_old_segment_does_not_retire_a_queued_replacement() {
    let mut app = App::new();
    app.add_message::<super::GotoEvent>()
        .add_message::<StartWalkEvent>()
        .init_resource::<ObservedMessages>()
        .add_systems(
            Update,
            (recalculate_near_end_of_path, observe_messages).chain(),
        );
    let mut pathfinder = active_pathfinder();
    pathfinder.is_calculating = false;
    pathfinder.opts = Some(PathfinderOpts::new().recalculate_partial_paths(false));
    let replacement_id = reserve_queued_goto(&mut pathfinder);
    let entity = app
        .world_mut()
        .spawn((
            pathfinder,
            ExecutingPath {
                path: VecDeque::new(),
                queued_path: None,
                last_reached_node: BlockPos::new(0, 64, 0),
                ticks_since_last_node_reached: 0,
                is_path_partial: true,
                allow_mining: false,
            },
        ))
        .id();

    app.update();

    let pathfinder = app.world().get::<Pathfinder>(entity).unwrap();
    assert_eq!(pathfinder.goto_id.load(Ordering::SeqCst), replacement_id);
    assert_eq!(pathfinder.queued_goto_id, Some(replacement_id));
    assert!(pathfinder.goal.is_some());
    assert!(pathfinder.opts.is_some());
    assert!(!pathfinder.is_calculating);
    assert!(app.world().get::<ExecutingPath>(entity).is_none());
    assert_eq!(app.world().resource::<ObservedMessages>().walk_stops, 1);
    assert_eq!(app.world().resource::<ObservedMessages>().gotos, 0);
}

#[test]
fn timeout_patching_preserves_a_graceful_stop_sentinel() {
    let mut pathfinder = active_pathfinder();
    pathfinder.goal = None;
    let mut executing = ExecutingPath {
        path: one_edge_path(),
        queued_path: Some(VecDeque::new()),
        last_reached_node: BlockPos::new(0, 64, 0),
        ticks_since_last_node_reached: 41,
        is_path_partial: true,
        allow_mining: false,
    };

    let pending = graceful_stop_pending(&pathfinder, &executing);
    executing.queued_path = None; // what patching does while replacing the path
    restore_graceful_stop(&mut executing, pending);

    assert!(
        executing
            .queued_path
            .as_ref()
            .is_some_and(VecDeque::is_empty)
    );
    assert!(!executing.is_path_partial);
}

#[test]
fn an_empty_timeout_patch_completes_a_graceful_stop() {
    let mut app = App::new();
    app.add_message::<super::GotoEvent>()
        .add_message::<StartWalkEvent>()
        .init_resource::<ObservedMessages>()
        .add_systems(
            Update,
            (recalculate_near_end_of_path, observe_messages).chain(),
        );
    let mut pathfinder = active_pathfinder();
    pathfinder.goal = None;
    pathfinder.is_calculating = false;
    let mut executing = ExecutingPath {
        path: one_edge_path(),
        queued_path: Some(VecDeque::new()),
        last_reached_node: BlockPos::new(0, 64, 0),
        ticks_since_last_node_reached: 41,
        is_path_partial: true,
        allow_mining: false,
    };
    let pending = graceful_stop_pending(&pathfinder, &executing);
    executing.path.clear(); // the timeout patch found no safe replacement
    executing.queued_path = None;
    restore_graceful_stop(&mut executing, pending);
    let entity = app.world_mut().spawn((pathfinder, executing)).id();

    app.update();

    let pathfinder = app.world().get::<Pathfinder>(entity).unwrap();
    assert!(pathfinder.goal.is_none());
    assert!(pathfinder.opts.is_none());
    assert!(!pathfinder.is_calculating);
    assert!(app.world().get::<ExecutingPath>(entity).is_none());
    assert_eq!(app.world().resource::<ObservedMessages>().walk_stops, 1);
}

#[test]
fn an_empty_timeout_patch_does_not_reset_the_stall_deadline() {
    let mut executing = ExecutingPath {
        path: VecDeque::new(),
        queued_path: None,
        last_reached_node: BlockPos::new(0, 64, 0),
        ticks_since_last_node_reached: 41,
        is_path_partial: true,
        allow_mining: false,
    };

    finish_timeout_patch(&mut executing, false, PatchOutcome::NoPath);

    assert_eq!(executing.ticks_since_last_node_reached, 41);
    assert!(executing.path.is_empty());
}

#[test]
fn a_usable_timeout_patch_starts_a_new_movement_deadline() {
    let mut executing = ExecutingPath {
        path: one_edge_path(),
        queued_path: None,
        last_reached_node: BlockPos::new(0, 64, 0),
        ticks_since_last_node_reached: 41,
        is_path_partial: true,
        allow_mining: false,
    };

    finish_timeout_patch(&mut executing, false, PatchOutcome::Applied);

    assert_eq!(executing.ticks_since_last_node_reached, 0);
    assert!(!executing.path.is_empty());
}

#[test]
fn retry_false_retires_a_goal_after_its_path_is_gone() {
    let mut app = App::new();
    app.add_message::<super::GotoEvent>()
        .add_message::<StartWalkEvent>()
        .init_resource::<ObservedMessages>()
        .add_systems(
            Update,
            (recalculate_if_has_goal_but_no_path, observe_messages).chain(),
        );
    let mut pathfinder = active_pathfinder();
    pathfinder.is_calculating = false;
    pathfinder.opts = Some(PathfinderOpts::new().retry_on_no_path(false));
    let entity = app.world_mut().spawn(pathfinder).id();

    app.update();

    let pathfinder = app.world().get::<Pathfinder>(entity).unwrap();
    assert!(pathfinder.goal.is_none());
    assert!(pathfinder.opts.is_none());
    assert!(!pathfinder.is_calculating);
    assert_eq!(app.world().resource::<ObservedMessages>().gotos, 0);
}

#[test]
fn an_internal_retry_exposes_its_stamped_queued_generation() {
    let mut app = App::new();
    app.add_message::<super::GotoEvent>()
        .add_message::<StartWalkEvent>()
        .init_resource::<ObservedMessages>()
        .add_systems(
            Update,
            (recalculate_if_has_goal_but_no_path, observe_messages).chain(),
        );
    let mut pathfinder = active_pathfinder();
    pathfinder.is_calculating = false;
    let before = pathfinder.goto_id.load(Ordering::SeqCst);
    let entity = app.world_mut().spawn(pathfinder).id();

    app.update();

    let pathfinder = app.world().get::<Pathfinder>(entity).unwrap();
    let queued_id = pathfinder
        .queued_goto_id
        .expect("the stamped retry must remain observable until goto_listener consumes it");
    assert_eq!(queued_id, before + 1);
    assert_eq!(pathfinder.goto_id.load(Ordering::SeqCst), queued_id);
    assert!(pathfinder.is_calculating);
    assert_eq!(app.world().resource::<ObservedMessages>().gotos, 1);
}

#[test]
fn changing_world_stops_a_calculation_without_an_executing_path() {
    let mut app = App::new();
    app.add_message::<StopPathfindingEvent>()
        .add_message::<StartWalkEvent>()
        .add_systems(
            Update,
            (
                stop_pathfinding_on_world_change,
                handle_stop_pathfinding_event,
            )
                .chain(),
        );
    let entity = app
        .world_mut()
        .spawn((active_pathfinder(), WorldName::new("minecraft:overworld")))
        .id();

    app.update();
    assert!(
        app.world()
            .get::<Pathfinder>(entity)
            .unwrap()
            .goal
            .is_some(),
        "initial WorldName insertion is not a dimension change"
    );
    *app.world_mut().get_mut::<WorldName>(entity).unwrap() = WorldName::new("minecraft:the_nether");
    app.update();

    let pathfinder = app.world().get::<Pathfinder>(entity).unwrap();
    assert!(pathfinder.goal.is_none());
    assert!(pathfinder.opts.is_none());
    assert!(!pathfinder.is_calculating);
}

fn mining_permission_after_path_found(allow_mining: bool) -> bool {
    let mut app = App::new();
    app.add_message::<PathFoundEvent>()
        .add_message::<StartWalkEvent>()
        .init_resource::<Worlds>()
        .add_systems(Update, path_found_listener);
    let pathfinder = active_pathfinder();
    let calculation_id = next_path_calculation_id(&pathfinder.goto_id);
    let entity = app
        .world_mut()
        .spawn((
            pathfinder,
            WorldName::new("minecraft:overworld"),
            Inventory::default(),
        ))
        .id();
    app.world_mut().write_message(PathFoundEvent {
        entity,
        calculation_id,
        start: BlockPos::new(0, 64, 0),
        path: Some(one_edge_path()),
        is_partial: false,
        successors_fn: moves::default_move,
        allow_mining,
    });

    app.update();

    app.world()
        .get::<ExecutingPath>(entity)
        .expect("a non-empty path must start executing")
        .allow_mining
}

/// The executor must never be more permissive than the plan that produced the
/// path. A path computed with mining forbidden can still walk into a block the
/// planner believed was air; breaking it would be an action nobody authorized.
#[test]
fn executing_path_carries_the_plans_mining_permission() {
    assert!(
        !mining_permission_after_path_found(false),
        "a path planned without mining must not let the executor mine"
    );
    assert!(
        mining_permission_after_path_found(true),
        "the permission must be read from the plan, not pinned to one value"
    );
}
