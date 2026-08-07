//! Regression test for a re-entrant ECS lock in the swarm client handler.
//!
//! `SwarmBuilder::start` runs a `client_handler_task` that holds a write guard
//! on the shared ECS (`ecs_mutex.write()`, `azalea/src/swarm/builder.rs:555`)
//! while it looks up each bot's state component. When that lookup fails it logs
//! an error whose *arguments* include `Client::username()` (`:560` for the
//! first bot, `:580` for any subsequent one) — and `username()` takes a *read*
//! guard on the very same `parking_lot::RwLock`
//! (`azalea/src/client_impl/entity_query.rs:313`). parking_lot's locks are not
//! re-entrant, so the handler thread parks forever: no panic, no log line, the
//! process simply never finishes shutting down.
//!
//! Three preconditions have to line up for that to fire, and each one gets its
//! own test here:
//!
//! 1. the state lookup has to miss — see
//!    [`clear_all_makes_the_state_lookup_miss`], which shows that the
//!    `World::clear_all()` the ECS runner performs on `AppExit`
//!    (`azalea-client/src/client.rs:216`) is enough to cause it;
//! 2. `tracing` has to actually evaluate the `error!` arguments, which it only
//!    does when a subscriber has `ERROR` enabled — see
//!    [`error_macro_evaluates_its_arguments_when_a_subscriber_is_installed`]
//!    here and the negative case in `swarm_handler_error_macro_gate.rs`;
//! 3. the write guard has to still be alive inside the `let ... else` block,
//!    which is what [`error_path_parks_on_the_guard_it_already_holds`] pins
//!    down.
//!
//! These tests reproduce the *statement sequence* of `swarm/builder.rs` rather
//! than driving a real `SwarmBuilder` (that would need a live server, and
//! `Swarm::bots_tx` is `pub(crate)` so an integration test cannot feed it), so
//! what they establish is the shape and its preconditions — not that
//! production reached that line. For that, see the live-run thread dumps
//! discussed in the accompanying report.

use std::{
    any::Any,
    panic::{self, AssertUnwindSafe},
    sync::{Arc, Once, mpsc},
    thread,
    time::{Duration, Instant},
};

use azalea::{Client, auth::game_profile::GameProfile, player::GameProfileComponent};
use bevy_ecs::{component::Component, world::World};
use bevy_log::tracing_subscriber::{
    filter::LevelFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt,
};
use parking_lot::RwLock;
use tracing::{Level, error};
use uuid::Uuid;

/// Stands in for the `S` state component of `SwarmBuilder<S, ..>`. Most of the
/// tests deliberately never insert it, which is what sends the handler down its
/// error path.
#[derive(Component, Clone)]
struct BotState;

const USERNAME: &str = "Bot";

/// How long to wait before calling a stuck handler thread deadlocked. The
/// original run of this test used 10s; 60s is here to rule out "it was merely
/// slow". Override with `AZALEA_DEADLOCK_TIMEOUT_SECS` when iterating.
fn timeout() -> Duration {
    let secs = std::env::var("AZALEA_DEADLOCK_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60);
    Duration::from_secs(secs)
}

/// `error!` only evaluates its arguments if some subscriber has `ERROR`
/// enabled, so every test that goes through the macro needs one installed.
fn install_error_subscriber() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let _ = bevy_log::tracing_subscriber::registry()
            .with(fmt::layer())
            .with(LevelFilter::ERROR)
            .try_init();
    });
}

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

/// What became of the handler thread. Keeping "panicked" separate from "never
/// answered" matters: a panic inside the thread also drops the sender, and
/// reading that as a deadlock would be wrong.
enum Outcome {
    Finished,
    Panicked(String),
}

fn panic_message(payload: &Box<dyn Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_owned()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_owned()
    }
}

/// Runs `body` on a named thread and reports whether it finished, panicked, or
/// never came back within [`timeout`].
fn run_with_timeout(
    name: &str,
    body: impl FnOnce() + Send + 'static,
) -> Result<Outcome, Duration> {
    let (done_tx, done_rx) = mpsc::channel();
    thread::Builder::new()
        .name(name.to_owned())
        .spawn(move || {
            let result = panic::catch_unwind(AssertUnwindSafe(body));
            let outcome = match result {
                Ok(()) => Outcome::Finished,
                Err(payload) => Outcome::Panicked(panic_message(&payload)),
            };
            let _ = done_tx.send(outcome);
        })
        .expect("spawn handler thread");

    let waited = timeout();
    let started = Instant::now();
    done_rx.recv_timeout(waited).map_err(|_| started.elapsed())
}

/// The error path of `client_handler_task` must complete instead of parking on
/// the lock it already holds.
///
/// This is the reproduction: the statement sequence below, including the
/// `error!` call, is copied from `azalea/src/swarm/builder.rs:553-564`.
#[test]
fn error_path_parks_on_the_guard_it_already_holds() {
    install_error_subscriber();
    assert!(
        tracing::enabled!(Level::ERROR),
        "the reproduction needs ERROR to be enabled, otherwise `error!` never \
         evaluates `username()` and there is nothing to deadlock on"
    );

    let (_ecs, first_bot) = client_with_profile();

    let outcome = run_with_timeout("swarm-client-handler", move || {
        // Statement sequence copied from `swarm/builder.rs`.
        let ecs_mutex = first_bot.ecs.clone();
        let mut ecs = ecs_mutex.write();
        let mut query = ecs.query::<Option<&BotState>>();
        let Ok(Some(_first_bot_state)) = query.get(&ecs, first_bot.entity) else {
            error!(
                "the first bot ({} / {}) is missing the required state component! none of the client handler functions will be called.",
                first_bot.username(),
                first_bot.entity
            );
            return;
        };
        unreachable!("the test never inserts BotState, so the query must miss");
    });

    match outcome {
        Ok(Outcome::Finished) => {
            panic!("expected the handler to deadlock, but it completed")
        }
        Ok(Outcome::Panicked(msg)) => panic!(
            "the handler panicked instead of deadlocking, so this run proves \
             nothing about the lock: {msg}"
        ),
        Err(waited) => {
            // This is the bug. Reported as a failure until it is fixed.
            panic!(
                "the swarm client handler deadlocked after {waited:?}: it asked for the \
                 bot's username while holding the ECS write guard, and parking_lot's \
                 RwLock is not re-entrant"
            )
        }
    }
}

/// Reverse control for [`error_path_parks_on_the_guard_it_already_holds`]: with
/// the state component present the query succeeds, `username()` is never
/// reached, and the same sequence completes immediately.
///
/// If this one hung too, the deadlock would have nothing to do with the error
/// branch and the whole attribution would be wrong.
#[test]
fn normal_path_with_the_state_component_present_completes() {
    install_error_subscriber();

    let (ecs, first_bot) = client_with_profile();
    ecs.write().entity_mut(first_bot.entity).insert(BotState);

    let outcome = run_with_timeout("swarm-client-handler-ok", move || {
        let ecs_mutex = first_bot.ecs.clone();
        let mut ecs = ecs_mutex.write();
        let mut query = ecs.query::<Option<&BotState>>();
        let Ok(Some(_first_bot_state)) = query.get(&ecs, first_bot.entity) else {
            error!(
                "the first bot ({} / {}) is missing the required state component! none of the client handler functions will be called.",
                first_bot.username(),
                first_bot.entity
            );
            return;
        };
        // The real handler goes on to clone the state and spawn the handler
        // task; what matters here is that it got past the query.
    });

    match outcome {
        Ok(Outcome::Finished) => {}
        Ok(Outcome::Panicked(msg)) => panic!("the normal path panicked: {msg}"),
        Err(waited) => panic!(
            "the normal path hung for {waited:?} as well, so the deadlock is not \
             specific to the error branch"
        ),
    }
}

/// Second reverse control: `username()` on its own is fine. Dropping the write
/// guard before asking for the name lets the exact same error path finish, so
/// it is the re-entrancy — not `username()` and not the missing component —
/// that parks the thread.
#[test]
fn error_path_completes_once_the_write_guard_is_released() {
    install_error_subscriber();

    let (_ecs, first_bot) = client_with_profile();

    let outcome = run_with_timeout("swarm-client-handler-released", move || {
        let ecs_mutex = first_bot.ecs.clone();
        let mut ecs = ecs_mutex.write();
        let mut query = ecs.query::<Option<&BotState>>();
        let Ok(Some(_first_bot_state)) = query.get(&ecs, first_bot.entity) else {
            drop(ecs);
            error!(
                "the first bot ({} / {}) is missing the required state component! none of the client handler functions will be called.",
                first_bot.username(),
                first_bot.entity
            );
            return;
        };
        unreachable!("the test never inserts BotState, so the query must miss");
    });

    match outcome {
        Ok(Outcome::Finished) => {}
        Ok(Outcome::Panicked(msg)) => panic!("the released-guard path panicked: {msg}"),
        Err(waited) => panic!(
            "the error path hung for {waited:?} even with the guard dropped, so the \
             re-entrancy is not what parks it"
        ),
    }
}

/// The trigger condition, without hand-waving: the ECS runner calls
/// `World::clear_all()` when it sees `AppExit`
/// (`azalea-client/src/client.rs:216`). After that, the state lookup the
/// handler performs takes the error branch for a bot whose events are still
/// queued in `bots_rx`.
#[test]
fn clear_all_makes_the_state_lookup_miss() {
    let mut world = World::new();
    let entity = world.spawn(BotState).id();

    let mut query = world.query::<Option<&BotState>>();
    assert!(
        matches!(query.get(&world, entity), Ok(Some(_))),
        "before shutdown the bot has its state component"
    );

    // What `run_schedule_loop` does on AppExit.
    world.clear_all();

    let mut query = world.query::<Option<&BotState>>();
    assert!(
        query.get(&world, entity).is_err(),
        "after clear_all the entity is gone, so the handler's `let ... else` takes \
         the error branch"
    );
}

/// `error!` is not an unconditional call: `tracing` skips evaluating the
/// arguments unless a subscriber has the level enabled. That makes an
/// ERROR-enabled subscriber a precondition of the whole bug, so pin it.
#[test]
fn error_macro_evaluates_its_arguments_when_a_subscriber_is_installed() {
    use std::sync::atomic::{AtomicBool, Ordering};

    install_error_subscriber();
    assert!(tracing::enabled!(Level::ERROR));

    static EVALUATED: AtomicBool = AtomicBool::new(false);
    fn probe() -> &'static str {
        EVALUATED.store(true, Ordering::SeqCst);
        "probe"
    }

    error!("argument evaluation probe: {}", probe());
    assert!(
        EVALUATED.load(Ordering::SeqCst),
        "with an ERROR-enabled subscriber the macro must evaluate its arguments"
    );
}

/// The write guard already grants access to the component, so the error path
/// can name the bot without going back through the lock. This is the shape a
/// fix would rely on.
#[test]
fn username_is_reachable_through_the_held_write_guard() {
    let (ecs, client) = client_with_profile();

    let guard = ecs.write();
    let name = guard
        .get::<GameProfileComponent>(client.entity)
        .map(|profile| profile.name.clone());

    assert_eq!(name.as_deref(), Some(USERNAME));
}

/// A second, independent criterion for "this is a lock cycle and not just
/// slowness": `parking_lot`'s own deadlock detector, which azalea already
/// enables via `azalea/Cargo.toml`'s dev-dependency on
/// `parking_lot = { features = ["deadlock_detection"] }`.
///
/// This test asserts nothing about the detector's verdict — whether it can see
/// a *same-thread* write→read re-entry is exactly the open question — it only
/// records what it reports. Run with `--nocapture` to see it.
#[test]
fn parking_lot_deadlock_detector_verdict() {
    install_error_subscriber();

    let (_ecs, first_bot) = client_with_profile();
    let entity = first_bot.entity;

    thread::Builder::new()
        .name("deadlock-detector-subject".to_owned())
        .spawn(move || {
            let ecs_mutex = first_bot.ecs.clone();
            let mut ecs = ecs_mutex.write();
            let mut query = ecs.query::<Option<&BotState>>();
            assert!(query.get(&ecs, entity).ok().flatten().is_none());
            // Same re-entry as the handler, minus the macro.
            let _ = first_bot.username();
        })
        .expect("spawn subject thread");

    // Give the subject thread time to park on the read lock.
    thread::sleep(Duration::from_secs(2));

    let cycles = parking_lot::deadlock::check_deadlock();
    eprintln!(
        "parking_lot::deadlock::check_deadlock() reported {} cycle(s) while a \
         same-thread write->read re-entry was parked",
        cycles.len()
    );
    for (i, cycle) in cycles.iter().enumerate() {
        eprintln!("  cycle {i}: {} thread(s)", cycle.len());
    }
}
