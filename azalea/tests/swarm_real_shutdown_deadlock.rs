//! R4: drive the **real** `SwarmBuilder` through a real shutdown and see
//! whether the process comes back.
//!
//! Everything else in this repo's deadlock tests reproduces the *statement
//! sequence* of `azalea/src/swarm/builder.rs:553-564`. This one does not
//! reproduce anything — it runs the actual `SwarmBuilder::start`, against a
//! real server, with a real bot producing a real event stream, and then asks
//! for `AppExit`. The chain under test is entirely inside azalea:
//!
//! ```text
//! bot connected, Event::Tick/Packet queueing into bots_rx
//!   -> AppExit::Success written
//!     -> run_schedule_loop sees it, calls World::clear_all()
//!        (azalea-client/src/client.rs:216) and returns
//!       -> client_handler_task polls a still-queued event
//!         -> query.get::<Option<&S>> misses (the entity is gone)
//!           -> error!(.., first_bot.username(), ..)  [builder.rs:560]
//!             -> Client::username -> ecs.read() while ecs.write() is held
//! ```
//!
//! Ignored by default because it needs a server. To run it:
//!
//! ```text
//! # a vanilla/Paper server on 127.0.0.1:25565 with online-mode=false
//! cargo test -p azalea --test swarm_real_shutdown_deadlock -- --ignored --nocapture
//! ```
//!
//! `AZALEA_PROBE_SERVER` overrides the address. `AZALEA_PROBE_SUBSCRIBER=0`
//! skips installing the tracing subscriber, which is the control: without an
//! `ERROR`-enabled subscriber `tracing` never evaluates the `error!`
//! arguments, so `username()` is never called and the shutdown should be
//! clean. Because a subscriber is process-global, run the two directions as
//! two separate `--exact` invocations.

use std::{
    sync::{
        Once,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use azalea::{app::PluginGroup, bot::DefaultBotPlugins, prelude::*, swarm::prelude::*};
use bevy_log::tracing_subscriber::{
    filter::LevelFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt,
};

#[derive(Clone, Component, Default)]
struct State;

#[derive(Clone, Default, Resource)]
struct SwarmState;

/// Events seen by the handler so far. Used only to wait until the event stream
/// is dense before asking to exit — the whole point is for `bots_rx` to be
/// non-empty at the moment the world is cleared.
static EVENTS: AtomicUsize = AtomicUsize::new(0);
static EXIT_REQUESTED: AtomicBool = AtomicBool::new(false);
static SPAWNED: AtomicBool = AtomicBool::new(false);

/// How many events to let through after spawning before asking for AppExit.
const EVENTS_BEFORE_EXIT: usize = 200;

/// Milliseconds of *blocking* work per event. The whole bug is conditional on
/// `bots_rx` being non-empty at the moment the world is cleared, and the
/// handler task is what starves `client_handler_task` of the LocalSet thread.
/// A real consumer gets this for free (MineIntent runs agent logic per event;
/// azalea's own `LogPlugin` gets it from synchronous INFO-level stderr writes).
/// With zero delay and no log traffic the queue is usually empty at shutdown
/// and the error branch is never reached.
fn handler_block() -> Duration {
    Duration::from_millis(
        std::env::var("AZALEA_PROBE_HANDLER_BLOCK_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
    )
}

async fn handle(bot: Client, event: Event, _state: State) -> eyre::Result<()> {
    let block = handler_block();
    if !block.is_zero() {
        thread::sleep(block);
    }
    match event {
        Event::Spawn => {
            SPAWNED.store(true, Ordering::SeqCst);
            eprintln!("[probe] bot spawned into the world");
        }
        Event::Tick => {
            let n = EVENTS.fetch_add(1, Ordering::SeqCst);
            if SPAWNED.load(Ordering::SeqCst)
                && n >= EVENTS_BEFORE_EXIT
                && !EXIT_REQUESTED.swap(true, Ordering::SeqCst)
            {
                eprintln!("[probe] {n} ticks in, requesting AppExit");
                bot.exit();
            }
        }
        _ => {
            EVENTS.fetch_add(1, Ordering::SeqCst);
        }
    }
    Ok(())
}

async fn swarm_handle(
    _swarm: Swarm,
    event: SwarmEvent,
    _state: SwarmState,
) -> eyre::Result<()> {
    if let SwarmEvent::Disconnect(account, ..) = &event {
        eprintln!("[probe] swarm disconnect for {}", account.username());
    }
    Ok(())
}

fn maybe_install_subscriber() -> bool {
    let want = std::env::var("AZALEA_PROBE_SUBSCRIBER").as_deref() != Ok("0");
    if want {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            let _ = bevy_log::tracing_subscriber::registry()
                .with(fmt::layer())
                .with(LevelFilter::ERROR)
                .try_init();
        });
    }
    want
}

#[test]
#[ignore = "needs a Minecraft server on 127.0.0.1:25565 with online-mode=false"]
fn real_swarm_shutdown() {
    let with_subscriber = maybe_install_subscriber();
    let enabled = tracing::enabled!(tracing::Level::ERROR);
    eprintln!("[probe] subscriber installed: {with_subscriber}, ERROR enabled: {enabled}");
    assert_eq!(
        with_subscriber, enabled,
        "the ERROR level should be enabled exactly when we installed a subscriber"
    );

    let address =
        std::env::var("AZALEA_PROBE_SERVER").unwrap_or_else(|_| "127.0.0.1:25565".to_owned());
    eprintln!("[probe] connecting to {address}");

    let (done_tx, done_rx) = mpsc::channel();
    thread::Builder::new()
        .name("swarm-runtime".to_owned())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("build current-thread runtime");
            let exit = rt.block_on(async move {
                // `SwarmBuilder::new()` would pull in `bevy_log::LogPlugin`, which
                // installs an INFO-level subscriber of its own — that would make the
                // `AZALEA_PROBE_SUBSCRIBER=0` control meaningless. Assemble the same
                // plugin set minus LogPlugin so the subscriber is the only variable.
                // This is also what MineIntent does (`new_without_plugins`).
                SwarmBuilder::new_without_plugins()
                    .add_plugins((
                        DefaultBotPlugins,
                        azalea::swarm::DefaultSwarmPlugins,
                        azalea::DefaultPlugins
                            .build()
                            .disable::<bevy_log::LogPlugin>(),
                    ))
                    .set_handler(handle)
                    .set_swarm_handler(swarm_handle)
                    .set_swarm_state(SwarmState)
                    .add_account_with_state(Account::offline("DeadlockProbe"), State)
                    .reconnect_after(None)
                    .start(address.as_str())
                    .await
            });
            let _ = done_tx.send(exit);
        })
        .expect("spawn swarm runtime thread");

    let started = Instant::now();
    match done_rx.recv_timeout(Duration::from_secs(180)) {
        Ok(exit) => {
            eprintln!(
                "[probe] SwarmBuilder::start returned {exit:?} after {:?}",
                started.elapsed()
            );
            assert!(
                EXIT_REQUESTED.load(Ordering::SeqCst),
                "the run ended before we ever asked for AppExit, so this proves nothing \
                 about shutdown — check that the bot actually connected and spawned"
            );
        }
        Err(_) => {
            panic!(
                "SwarmBuilder::start never returned after {:?} (spawned: {}, exit requested: {}, \
                 events: {}). The client handler is parked on the ECS write guard it already \
                 holds; see azalea/src/swarm/builder.rs:555-562.",
                started.elapsed(),
                SPAWNED.load(Ordering::SeqCst),
                EXIT_REQUESTED.load(Ordering::SeqCst),
                EVENTS.load(Ordering::SeqCst),
            )
        }
    }
}
