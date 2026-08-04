use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use azalea_entity::{LocalEntity, indexing::EntityUuidIndex};
use azalea_protocol::{
    address::ResolvedAddr,
    common::client_information::ClientInformation,
    connect::{Connection, ConnectionError, Proxy},
    packets::{
        ClientIntention, ConnectionProtocol, PROTOCOL_VERSION,
        handshake::ServerboundIntention,
        login::{ClientboundLoginPacket, ServerboundHello, ServerboundLoginPacket},
    },
};
use azalea_world::World;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_tasks::{IoTaskPool, Task, futures_lite::future};
use parking_lot::RwLock;
use tokio::sync::mpsc;
use tracing::{debug, warn};

use crate::{
    LocalPlayerBundle,
    account::Account,
    connection::RawConnection,
    local_player::WorldHolder,
    packet::login::{InLoginState, SendLoginPacketEvent},
};

/// A plugin that allows bots to join servers.
pub struct JoinPlugin;
impl Plugin for JoinPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<StartJoinServerEvent>()
            .add_message::<ConnectionFailedEvent>()
            .add_message::<CancelConnectionTaskEvent>()
            .add_systems(
                Update,
                (
                    handle_start_join_server_event.before(super::login::poll_auth_task),
                    cancel_create_connection_task,
                    poll_create_connection_task,
                )
                    .chain(),
            );
    }
}

/// A process-unique, immutable identity for one join attempt.
///
/// A new token is minted exactly once before a `StartJoinServerEvent` enters
/// the message flow (see `Client::start_client` and the auto-reconnect
/// system), and is carried unchanged through the resolved
/// `(Entity, AttemptToken)` handoff, the [`CreateConnectionTask`], the
/// [`RawConnection`](crate::connection::RawConnection), and every event that
/// the attempt produces. Reusing an entity does not reuse the token: every
/// attempt (including automatic reconnects) mints a fresh one.
#[derive(Clone, Copy, Debug, Component, Eq, Hash, PartialEq)]
pub struct AttemptToken(u64);

impl AttemptToken {
    /// Mint a fresh process-unique attempt token.
    ///
    /// This is intentionally only called at the join-attempt entry points
    /// (`Client::start_client` and `rejoin_after_delay`); handlers and event
    /// producers must never re-mint.
    pub fn mint() -> Self {
        static NEXT_ATTEMPT_TOKEN: AtomicU64 = AtomicU64::new(1);
        Self::mint_from(&NEXT_ATTEMPT_TOKEN).expect("process attempt token space exhausted")
    }

    /// Checked mint from an explicit counter: returns the unique value that
    /// was allocated for this token (the counter value before the update) and
    /// advances the counter by one. Once the counter has reached `u64::MAX`,
    /// returns `None` without changing it. It never wraps and never re-issues
    /// a token.
    pub(crate) fn mint_from(counter: &AtomicU64) -> Option<Self> {
        counter
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .ok()
            .map(Self)
    }
}

/// An event to make a client join the server and be added to our swarm.
///
/// This won't do anything if a client with the Account UUID is already
/// connected to the server.
#[derive(Debug, Message)]
pub struct StartJoinServerEvent {
    pub account: Account,
    pub connect_opts: ConnectOpts,

    // this is mpsc instead of oneshot so it can be cloned (since it's sent in an event)
    pub start_join_callback_tx: Option<mpsc::UnboundedSender<(Entity, AttemptToken)>>,
    /// The immutable identity of this join attempt. Handlers must carry it
    /// through, never mint their own.
    pub attempt_token: AttemptToken,
}

/// Options for how the connection to the server will be made.
///
/// These are persisted on reconnects. This is inserted as a component on
/// clients to make auto-reconnecting work.
#[derive(Clone, Component, Debug)]
pub struct ConnectOpts {
    pub address: ResolvedAddr,
    /// The SOCKS5 proxy used for connecting to the Minecraft server.
    pub server_proxy: Option<Proxy>,
    /// The SOCKS5 proxy that will be used when authenticating our server join
    /// with Mojang.
    ///
    /// This should typically be either the same as [`Self::server_proxy`], or
    /// `None`.
    ///
    /// This is useful to set if a server has `prevent-proxy-connections`
    /// enabled.
    pub sessionserver_proxy: Option<Proxy>,
}

/// An event that's sent when creating the TCP connection and sending the first
/// packet fails.
///
/// This isn't sent if we're kicked later, see [`DisconnectEvent`].
///
/// [`DisconnectEvent`]: crate::disconnect::DisconnectEvent
#[derive(Message)]
pub struct ConnectionFailedEvent {
    pub entity: Entity,
    // wrap it in Arc so it can be cloned
    pub error: Arc<ConnectionError>,
    /// The join attempt that produced this failure.
    pub attempt_token: AttemptToken,
}

/// A message that cancels a pending pre-Init [`CreateConnectionTask`].
///
/// The cancellation only matches when the entity's *current* attempt token
/// equals the token in the message. Cancelling an attempt that already
/// completed, or one that has already been replaced by a newer attempt on the
/// same entity, is a safe no-op. Removing the task component drops the
/// underlying `Task`, which cancels and drops the connection future.
#[derive(Debug, Message)]
pub struct CancelConnectionTaskEvent {
    pub entity: Entity,
    pub attempt_token: AttemptToken,
}

pub fn handle_start_join_server_event(
    mut commands: Commands,
    mut events: MessageReader<StartJoinServerEvent>,
    mut entity_uuid_index: ResMut<EntityUuidIndex>,
    connection_query: Query<&RawConnection>,
) {
    for event in events.read() {
        let uuid = event.account.uuid();
        let attempt_token = event.attempt_token;
        let entity = if let Some(entity) = entity_uuid_index.get(&uuid) {
            debug!("Reusing entity {entity:?} for client");

            // check if it's already connected
            if let Ok(conn) = connection_query.get(entity)
                && conn.is_alive()
            {
                if let Some(start_join_callback_tx) = &event.start_join_callback_tx {
                    warn!(
                        "Received StartJoinServerEvent for {entity:?} but it's already connected. Ignoring the event but replying with Ok."
                    );
                    // Reply with the identity of the connection that is
                    // actually alive, not the token of the ignored new event.
                    let _ = start_join_callback_tx.send((entity, conn.attempt_token()));
                } else {
                    warn!(
                        "Received StartJoinServerEvent for {entity:?} but it's already connected. Ignoring the event."
                    );
                }
                return;
            }

            entity
        } else {
            let entity = commands.spawn_empty().id();
            debug!("Created new entity {entity:?} for client");
            // add to the uuid index
            entity_uuid_index.insert(uuid, entity);
            entity
        };

        if let Some(start_join_callback) = &event.start_join_callback_tx {
            let _ = start_join_callback.send((entity, attempt_token));
        }

        let mut entity_mut = commands.entity(entity);

        entity_mut.insert((
            // add the Account to the entity now so plugins can access it earlier
            event.account.to_owned(),
            // the immutable identity of this attempt, kept on the entity until
            // the next attempt replaces it
            attempt_token,
            // localentity is always present for our clients, even if we're not actually logged
            // in
            LocalEntity,
            // this is inserted early so the user can always access and modify it
            ClientInformation::default(),
            // ConnectOpts is inserted as a component here
            event.connect_opts.clone(),
            // we don't insert InLoginState until we actually create the connection. note that
            // there's no InHandshakeState component since we switch off of the handshake state
            // immediately when the connection is created
        ));

        let task_pool = IoTaskPool::get();
        let connect_opts = event.connect_opts.clone();
        let task = task_pool.spawn(async_compat::Compat::new(
            create_conn_and_send_intention_packet(connect_opts),
        ));

        entity_mut.insert(CreateConnectionTask {
            task,
            attempt_token,
        });
    }
}

async fn create_conn_and_send_intention_packet(
    opts: ConnectOpts,
) -> Result<LoginConn, ConnectionError> {
    let mut conn = if let Some(proxy) = opts.server_proxy {
        Connection::new_with_proxy(&opts.address.socket, proxy).await?
    } else {
        Connection::new(&opts.address.socket).await?
    };

    conn.write(ServerboundIntention {
        protocol_version: PROTOCOL_VERSION,
        hostname: opts.address.server.host.clone(),
        port: opts.address.server.port,
        intention: ClientIntention::Login,
    })
    .await?;

    let conn = conn.login();

    Ok(conn)
}

type LoginConn = Connection<ClientboundLoginPacket, ServerboundLoginPacket>;

#[derive(Component)]
pub struct CreateConnectionTask {
    pub task: Task<Result<LoginConn, ConnectionError>>,
    /// The join attempt this task belongs to. The task may only install its
    /// connection while this is still the entity's current attempt token.
    pub attempt_token: AttemptToken,
}

/// Cancels a pending [`CreateConnectionTask`] by `(Entity, AttemptToken)`.
///
/// Runs before [`poll_create_connection_task`], so a task cancelled in the
/// same tick can never be polled into installing its connection afterwards.
pub fn cancel_create_connection_task(
    mut commands: Commands,
    mut events: MessageReader<CancelConnectionTaskEvent>,
    query: Query<(Entity, &CreateConnectionTask, &AttemptToken)>,
) {
    for cancel in events.read() {
        for (entity, task, current_attempt_token) in &query {
            // Precise cancellation requires all three identities to match:
            // the entity, the entity's current attempt, and the task's own
            // attempt. A residual task from attempt A that somehow survives
            // attempt B's start must be cancelled neither by cancel(B) (task
            // mismatch) nor by cancel(A) (current fence).
            if entity == cancel.entity
                && *current_attempt_token == cancel.attempt_token
                && task.attempt_token == cancel.attempt_token
            {
                // Removing the component drops the `Task`, which cancels and
                // drops the underlying connection future.
                commands.entity(entity).remove::<CreateConnectionTask>();
            }
        }
    }
}

pub fn poll_create_connection_task(
    mut commands: Commands,
    mut query: Query<(Entity, &mut CreateConnectionTask, &Account, &AttemptToken)>,
    mut connection_failed_events: MessageWriter<ConnectionFailedEvent>,
) {
    for (entity, mut task, account, current_attempt_token) in query.iter_mut() {
        if let Some(poll_res) = future::block_on(future::poll_once(&mut task.task)) {
            let attempt_token = task.attempt_token;
            let mut entity_mut = commands.entity(entity);
            entity_mut.remove::<CreateConnectionTask>();
            // A task that completed after its attempt was replaced must not
            // install its connection, remove anything, or emit events over
            // the current attempt.
            if *current_attempt_token != attempt_token {
                warn!("Ignoring completed connection task from a stale attempt");
                continue;
            }
            let conn = match poll_res {
                Ok(conn) => conn,
                Err(error) => {
                    warn!("failed to create connection: {error}");
                    connection_failed_events.write(ConnectionFailedEvent {
                        entity,
                        error: Arc::new(error),
                        attempt_token,
                    });
                    return;
                }
            };

            let (read_conn, write_conn) = conn.into_split();
            let (read_conn, write_conn) = (read_conn.raw, write_conn.raw);

            let world = World::default();
            let world_holder = WorldHolder::new(
                entity,
                // default to an empty world, it'll be set correctly later when we
                // get the login packet
                Arc::new(RwLock::new(world)),
            );

            entity_mut.insert((
                // these stay when we switch to the game state
                LocalPlayerBundle {
                    raw_connection: RawConnection::new(
                        read_conn,
                        write_conn,
                        ConnectionProtocol::Login,
                        attempt_token,
                    ),
                    world_holder,
                    metadata: azalea_entity::metadata::PlayerMetadataBundle::default(),
                },
                InLoginState,
            ));

            commands.trigger(SendLoginPacketEvent::new(
                entity,
                ServerboundHello {
                    name: account.username().to_owned(),
                    profile_id: account.uuid(),
                },
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::atomic::AtomicBool,
        time::{Duration, Instant},
    };

    use azalea_protocol::address::{ResolvedAddr, ServerAddr};
    use bevy_ecs::message::Messages;

    fn test_app() -> App {
        let mut app = App::new();
        let mut plugins = bevy_app::PluginGroup::build(crate::DefaultPlugins);
        #[cfg(feature = "log")]
        {
            plugins = plugins.disable::<bevy_log::LogPlugin>();
        }
        app.add_plugins(plugins);
        app.edit_schedule(bevy_app::Main, |schedule| {
            schedule.set_executor_kind(bevy_ecs::schedule::ExecutorKind::SingleThreaded);
        });
        app
    }

    fn probe_address() -> ResolvedAddr {
        ResolvedAddr {
            server: ServerAddr::try_from("localhost:25565").unwrap(),
            socket: "127.0.0.1:1".parse().unwrap(),
        }
    }

    fn spawn_pending_probe(dropped: Arc<AtomicBool>) -> Task<Result<LoginConn, ConnectionError>> {
        let probe = async move {
            struct DropProbe(Arc<AtomicBool>);
            impl Drop for DropProbe {
                fn drop(&mut self) {
                    self.0.store(true, Ordering::SeqCst);
                }
            }
            let _guard = DropProbe(dropped);
            std::future::pending::<()>().await;
            unreachable!("probe future never completes");
        };
        IoTaskPool::get().spawn(async_compat::Compat::new(probe))
    }

    fn spawn_immediate_error_probe() -> Task<Result<LoginConn, ConnectionError>> {
        let probe = async {
            Err(ConnectionError::Io(std::io::Error::other(
                "controlled probe failure",
            )))
        };
        IoTaskPool::get().spawn(async_compat::Compat::new(probe))
    }

    fn wait_until_dropped(dropped: &AtomicBool, what: &str) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !dropped.load(Ordering::SeqCst) {
            assert!(
                Instant::now() < deadline,
                "{what} was not dropped within the bounded window"
            );
            std::thread::yield_now();
        }
    }

    #[test]
    fn checked_mint_never_wraps_or_reissues() {
        let max_counter = AtomicU64::new(u64::MAX);
        assert_eq!(AttemptToken::mint_from(&max_counter), None);
        assert_eq!(max_counter.load(Ordering::Relaxed), u64::MAX);

        let near_max = AtomicU64::new(u64::MAX - 1);
        assert_eq!(
            AttemptToken::mint_from(&near_max),
            Some(AttemptToken(u64::MAX - 1))
        );
        assert_eq!(near_max.load(Ordering::Relaxed), u64::MAX);
        assert_eq!(AttemptToken::mint_from(&near_max), None);

        let counter = AtomicU64::new(1);
        let first = AttemptToken::mint_from(&counter).unwrap();
        let second = AttemptToken::mint_from(&counter).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn cancel_requires_task_owned_and_current_token_match() {
        let mut app = test_app();
        app.update();

        let entity = app.world_mut().spawn_empty().id();
        // Interleaved state: the entity's current attempt is B, but a
        // residual task from attempt A is still attached.
        let token_a = AttemptToken::mint();
        let token_b = AttemptToken::mint();
        let dropped_a = Arc::new(AtomicBool::new(false));
        app.world_mut().entity_mut(entity).insert((
            Account::offline("probe-mismatch"),
            token_b,
            CreateConnectionTask {
                task: spawn_pending_probe(Arc::clone(&dropped_a)),
                attempt_token: token_a,
            },
        ));

        // cancel(B): the task is owned by A, so it must not be removed.
        app.world_mut().write_message(CancelConnectionTaskEvent {
            entity,
            attempt_token: token_b,
        });
        app.update();
        assert!(
            app.world()
                .entity(entity)
                .contains::<CreateConnectionTask>(),
            "cancel(B) must not remove a residual task owned by A"
        );

        // cancel(A): the entity's current attempt is B, so the current fence
        // must also reject it.
        app.world_mut().write_message(CancelConnectionTaskEvent {
            entity,
            attempt_token: token_a,
        });
        app.update();
        assert!(
            app.world()
                .entity(entity)
                .contains::<CreateConnectionTask>(),
            "cancel(A) must not cross the current attempt fence when current=B"
        );

        // A fully matching cancel still works.
        let token_c = AttemptToken::mint();
        let dropped_c = Arc::new(AtomicBool::new(false));
        app.world_mut().entity_mut(entity).insert((
            token_c,
            CreateConnectionTask {
                task: spawn_pending_probe(Arc::clone(&dropped_c)),
                attempt_token: token_c,
            },
        ));
        app.update();
        app.world_mut().write_message(CancelConnectionTaskEvent {
            entity,
            attempt_token: token_c,
        });
        app.update();
        assert!(
            !app.world()
                .entity(entity)
                .contains::<CreateConnectionTask>()
        );
        wait_until_dropped(&dropped_c, "matching attempt C task");
    }

    #[test]
    fn cancel_matching_task_drops_future_and_stale_cancel_is_noop() {
        let mut app = test_app();
        app.update();

        let entity = app.world_mut().spawn_empty().id();
        app.world_mut()
            .entity_mut(entity)
            .insert((Account::offline("probe-a"), AttemptToken::mint()));

        let token_a = app
            .world()
            .entity(entity)
            .get::<AttemptToken>()
            .copied()
            .unwrap();
        let dropped_a = Arc::new(AtomicBool::new(false));
        app.world_mut()
            .entity_mut(entity)
            .insert(CreateConnectionTask {
                task: spawn_pending_probe(Arc::clone(&dropped_a)),
                attempt_token: token_a,
            });
        // Poll A's task once so the drop guard exists while it is pending.
        app.update();
        assert!(
            app.world()
                .entity(entity)
                .contains::<CreateConnectionTask>()
        );

        // Matching cancel: the component is removed, which drops the `Task`
        // and cancels/drops the pending connection future.
        app.world_mut().write_message(CancelConnectionTaskEvent {
            entity,
            attempt_token: token_a,
        });
        app.update();
        assert!(
            !app.world()
                .entity(entity)
                .contains::<CreateConnectionTask>()
        );
        wait_until_dropped(&dropped_a, "cancelled attempt A task");

        // B reuses the same entity with a fresh token.
        let token_b = AttemptToken::mint();
        let b_flag = Arc::new(AtomicBool::new(false));
        app.world_mut().entity_mut(entity).insert((
            token_b,
            CreateConnectionTask {
                task: spawn_pending_probe(Arc::clone(&b_flag)),
                attempt_token: token_b,
            },
        ));
        // Poll B's task once so its drop guard exists while it is pending.
        app.update();
        assert!(
            app.world()
                .entity(entity)
                .contains::<CreateConnectionTask>()
        );

        // Stale cancel(A) must not remove B's task or drop B's future.
        app.world_mut().write_message(CancelConnectionTaskEvent {
            entity,
            attempt_token: token_a,
        });
        app.update();
        assert!(
            app.world()
                .entity(entity)
                .contains::<CreateConnectionTask>()
        );
        let deadline = Instant::now() + Duration::from_millis(200);
        while Instant::now() < deadline && b_flag.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
        assert!(
            !b_flag.load(Ordering::SeqCst),
            "stale cancel(A) must not drop B's pending future"
        );

        // Matching cancel(B) still works.
        app.world_mut().write_message(CancelConnectionTaskEvent {
            entity,
            attempt_token: token_b,
        });
        app.update();
        assert!(
            !app.world()
                .entity(entity)
                .contains::<CreateConnectionTask>()
        );
        wait_until_dropped(&b_flag, "matching attempt B task");
    }

    #[test]
    fn stale_completed_task_cannot_install_raw_connection_or_emit_connection_failed() {
        let mut app = test_app();
        app.update();

        let entity = app.world_mut().spawn_empty().id();
        // The entity is already owned by attempt B when A's task completes.
        let token_b = AttemptToken::mint();
        let token_a = AttemptToken::mint();
        app.world_mut().entity_mut(entity).insert((
            Account::offline("probe-b"),
            token_b,
            CreateConnectionTask {
                task: spawn_immediate_error_probe(),
                attempt_token: token_a,
            },
        ));

        app.update();

        assert!(
            !app.world()
                .entity(entity)
                .contains::<crate::connection::RawConnection>(),
            "a stale attempt must never install RawConnection over the current attempt"
        );
        assert!(
            !app.world()
                .entity(entity)
                .contains::<CreateConnectionTask>(),
            "the completed stale task must be removed"
        );
        let messages = app.world().resource::<Messages<ConnectionFailedEvent>>();
        let mut cursor = messages.get_cursor();
        assert_eq!(
            cursor.read(messages).count(),
            0,
            "a stale attempt must not emit ConnectionFailedEvent for the current attempt"
        );
        assert_eq!(
            *app.world().entity(entity).get::<AttemptToken>().unwrap(),
            token_b,
            "the current attempt identity must be untouched"
        );
    }

    #[test]
    fn connection_failed_event_stamps_the_attempt_token() {
        let mut app = test_app();
        app.update();

        let entity = app.world_mut().spawn_empty().id();
        let token = AttemptToken::mint();
        app.world_mut().entity_mut(entity).insert((
            Account::offline("probe-fail"),
            token,
            CreateConnectionTask {
                task: spawn_immediate_error_probe(),
                attempt_token: token,
            },
        ));

        app.update();

        assert!(
            !app.world()
                .entity(entity)
                .contains::<CreateConnectionTask>()
        );
        let messages = app.world().resource::<Messages<ConnectionFailedEvent>>();
        let mut cursor = messages.get_cursor();
        let events: Vec<_> = cursor.read(messages).collect();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].entity, entity);
        assert_eq!(events[0].attempt_token, token);
    }

    #[test]
    fn start_join_callback_handoff_carries_entity_and_attempt_token() {
        let mut app = test_app();
        app.update();

        let token = AttemptToken::mint();
        let (callback_tx, mut callback_rx) = mpsc::unbounded_channel();
        app.world_mut().write_message(StartJoinServerEvent {
            account: Account::offline("probe-handoff"),
            connect_opts: ConnectOpts {
                address: probe_address(),
                server_proxy: None,
                sessionserver_proxy: None,
            },
            start_join_callback_tx: Some(callback_tx),
            attempt_token: token,
        });

        app.update();

        let (entity, handoff_token) = callback_rx
            .try_recv()
            .expect("the resolved handoff must carry (Entity, AttemptToken)");
        assert_eq!(handoff_token, token);
        assert_eq!(
            *app.world().entity(entity).get::<AttemptToken>().unwrap(),
            token,
            "the entity must carry the same attempt token"
        );
        assert!(
            app.world()
                .entity(entity)
                .contains::<CreateConnectionTask>()
        );
    }
}
