//! The negative half of the precondition pinned by
//! `swarm_handler_ecs_lock.rs`: with **no** `tracing` subscriber installed,
//! `error!` never evaluates its arguments, so the re-entrant
//! `Client::username()` call at `azalea/src/swarm/builder.rs:560` is never
//! made and the handler does *not* deadlock.
//!
//! This lives in its own integration-test binary because a subscriber is
//! process-global once installed — it cannot be un-installed for one test.
//!
//! The practical consequence: the bug is real but conditional. A consumer that
//! never installs a subscriber with `ERROR` enabled will not hit it. The live
//! MineIntent runs did install one, which is why their thread dumps show the
//! process parked inside `Client::username`.

use std::{
    sync::{Arc, mpsc},
    thread,
    time::Duration,
};

use azalea::{Client, auth::game_profile::GameProfile, player::GameProfileComponent};
use bevy_ecs::{component::Component, world::World};
use parking_lot::RwLock;
use tracing::{Level, error};
use uuid::Uuid;

#[derive(Component, Clone)]
struct BotState;

#[test]
fn without_a_subscriber_the_error_path_does_not_deadlock() {
    assert!(
        !tracing::enabled!(Level::ERROR),
        "this test binary must not install a tracing subscriber"
    );

    let ecs = Arc::new(RwLock::new(World::new()));
    let entity = ecs
        .write()
        .spawn(GameProfileComponent(GameProfile::new(
            Uuid::nil(),
            "Bot".to_owned(),
        )))
        .id();
    let first_bot = Client::new(entity, ecs.clone());

    let (done_tx, done_rx) = mpsc::channel();
    thread::Builder::new()
        .name("swarm-client-handler-nosub".to_owned())
        .spawn(move || {
            let ecs_mutex = first_bot.ecs.clone();
            let mut ecs = ecs_mutex.write();
            let mut query = ecs.query::<Option<&BotState>>();
            let Ok(Some(_first_bot_state)) = query.get(&ecs, first_bot.entity) else {
                error!(
                    "the first bot ({} / {}) is missing the required state component! none of the client handler functions will be called.",
                    first_bot.username(),
                    first_bot.entity
                );
                let _ = done_tx.send(());
                return;
            };
            unreachable!("the test never inserts BotState, so the query must miss");
        })
        .expect("spawn handler thread");

    done_rx
        .recv_timeout(Duration::from_secs(30))
        .expect("with no subscriber the error! arguments are never evaluated, so the handler must finish");
}
