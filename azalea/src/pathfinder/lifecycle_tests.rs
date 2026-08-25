use std::{
    collections::VecDeque,
    sync::{Arc, atomic::Ordering},
};

use azalea_client::StartWalkEvent;
use azalea_core::position::BlockPos;
use azalea_entity::inventory::Inventory;
use azalea_world::{WorldName, Worlds};
use bevy_app::{App, Update};
use bevy_ecs::prelude::*;

use super::{
    ExecutingPath, PathFoundEvent, Pathfinder, PathfinderOpts, StopPathfindingEvent,
    astar::{Edge, Movement},
    execute::{
        graceful_stop_pending, recalculate_if_has_goal_but_no_path, recalculate_near_end_of_path,
        restore_graceful_stop, retire_pathfinder_request,
    },
    goals::BlockPosGoal,
    handle_stop_pathfinding_event, next_path_calculation_id, path_found_listener,
    stop_pathfinding_on_world_change,
};
use crate::{
    WalkDirection,
    pathfinder::moves::{
        self, MoveData,
        basic::{descend_is_reached, execute_descend_move},
    },
};

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
            data: MoveData {
                execute: &execute_descend_move,
                is_reached: &descend_is_reached,
            },
        },
        cost: 1.0,
    }])
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
fn timeout_patching_preserves_a_graceful_stop_sentinel() {
    let mut pathfinder = active_pathfinder();
    pathfinder.goal = None;
    let mut executing = ExecutingPath {
        path: one_edge_path(),
        queued_path: Some(VecDeque::new()),
        last_reached_node: BlockPos::new(0, 64, 0),
        ticks_since_last_node_reached: 41,
        is_path_partial: true,
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
