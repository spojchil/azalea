//! Regression test for a re-entrant ECS lock in the swarm client handler.
//!
//! `SwarmBuilder::start` runs a `client_handler_task` that holds a write guard
//! on the shared ECS (`ecs_mutex.write()`) while it looks up each bot's state
//! component. When that lookup fails it logs an error whose message includes
//! `Client::username()` — and `username()` takes a *read* guard on the very
//! same `parking_lot::RwLock`. parking_lot's locks are not re-entrant, so the
//! handler thread parks forever: no panic, no log line, the process simply
//! never finishes shutting down.
//!
//! The lookup fails exactly when a bot's entity has already lost its state
//! component while its events are still queued in `bots_rx`, which is what
//! happens during shutdown.
//!
//! These tests reproduce the statement sequence of `swarm/builder.rs` rather
//! than driving a real `SwarmBuilder` (that would need a live server), so they
//! pin the *shape* of that code: hold the write guard, then reach for the
//! username through a path that re-locks.

use std::{
    sync::{Arc, mpsc},
    thread,
    time::Duration,
};

use azalea::{Client, auth::game_profile::GameProfile, player::GameProfileComponent};
use bevy_ecs::{component::Component, world::World};
use parking_lot::RwLock;
use uuid::Uuid;

/// Stands in for the `S` state component of `SwarmBuilder<S, ..>`. The tests
/// deliberately never insert it, which is what sends the handler down its
/// error path.
#[derive(Component, Clone)]
struct BotState;

const USERNAME: &str = "Bot";

/// Build a one-entity ECS plus a `Client` handle for it, mirroring what the
/// handler receives from `bots_rx`. The handle is built *before* any write
/// guard is taken because `Client::new` reads the ECS itself.
fn client_with_profile() -> (Arc<RwLock<World>>, Client) {
    let ecs = Arc::new(RwLock::new(World::new()));
    let entity = ecs
        .write()
        .spawn(GameProfileComponent(GameProfile::new(
            Uuid::nil(),
            USERNAME.to_owned(),
        )))
        .id();
    let client = Client::new(entity, ecs.clone());
    (ecs, client)
}

/// The error path of `client_handler_task` must complete instead of parking on
/// the lock it already holds.
#[test]
fn swarm_handler_error_path_does_not_relock_the_ecs() {
    let (_ecs, first_bot) = client_with_profile();

    let (done_tx, done_rx) = mpsc::channel();
    thread::Builder::new()
        .name("swarm-client-handler".to_owned())
        .spawn(move || {
            // Statement sequence copied from `swarm/builder.rs`.
            let ecs_mutex = first_bot.ecs.clone();
            let mut ecs = ecs_mutex.write();
            let mut query = ecs.query::<Option<&BotState>>();
            let Ok(Some(_first_bot_state)) = query.get(&ecs, first_bot.entity) else {
                // What the `error!` call needs. Taking it off the `Client`
                // re-locks the guard we are still holding.
                let name = first_bot.username();
                drop(ecs);
                let _ = done_tx.send(name);
                return;
            };
            unreachable!("the test never inserts BotState, so the query must miss");
        })
        .expect("spawn handler thread");

    match done_rx.recv_timeout(Duration::from_secs(10)) {
        Ok(name) => assert_eq!(name, USERNAME),
        Err(_) => panic!(
            "the swarm client handler deadlocked: it asked for the bot's username while \
             holding the ECS write guard, and parking_lot's RwLock is not re-entrant"
        ),
    }
}

/// The write guard already grants access to the component, so the error path
/// can name the bot without going back through the lock. This is the shape the
/// fix relies on.
#[test]
fn username_is_reachable_through_the_held_write_guard() {
    let (ecs, client) = client_with_profile();

    let guard = ecs.write();
    let name = guard
        .get::<GameProfileComponent>(client.entity)
        .map(|profile| profile.name.clone());

    assert_eq!(name.as_deref(), Some(USERNAME));
}
