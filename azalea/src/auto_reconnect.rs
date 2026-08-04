//! Auto-reconnect to the server when the client is kicked.
//!
//! See [`AutoReconnectPlugin`] for more information.

use std::time::{Duration, Instant};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use super::{
    disconnect::DisconnectEvent,
    join::{AttemptToken, ConnectOpts, ConnectionFailedEvent, StartJoinServerEvent},
};
use crate::account::Account;
use crate::events::attempt_matches_current;

/// The default delay that Azalea will use for reconnecting our clients.
///
/// See [`AutoReconnectPlugin`] for more information.
pub const DEFAULT_RECONNECT_DELAY: Duration = Duration::from_secs(5);

/// A default plugin that makes clients automatically rejoin the server when
/// they're disconnected.
///
/// The reconnect delay is configurable globally or per-client with the
/// [`AutoReconnectDelay`] resource/component. Auto reconnecting can be disabled
/// by removing the resource from the ECS.
///
/// The delay defaults to [`DEFAULT_RECONNECT_DELAY`].
pub struct AutoReconnectPlugin;
impl Plugin for AutoReconnectPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(AutoReconnectDelay::new(DEFAULT_RECONNECT_DELAY))
            .add_systems(
                Update,
                (start_rejoin_on_disconnect, rejoin_after_delay)
                    .chain()
                    .before(super::join::handle_start_join_server_event),
            );
    }
}

pub fn start_rejoin_on_disconnect(
    mut commands: Commands,
    mut disconnect_events: MessageReader<DisconnectEvent>,
    mut connection_failed_events: MessageReader<ConnectionFailedEvent>,
    auto_reconnect_delay_res: Option<Res<AutoReconnectDelay>>,
    auto_reconnect_delay_query: Query<&AutoReconnectDelay>,
    current_attempt_query: Query<Option<&AttemptToken>>,
) {
    let disconnect_entries = disconnect_events
        .read()
        .map(|e| (e.entity, e.attempt_token))
        .chain(
            connection_failed_events
                .read()
                .map(|e| (e.entity, Some(e.attempt_token))),
        );
    for (entity, event_attempt) in disconnect_entries {
        // A stale event from a replaced attempt must never schedule a
        // reconnect for the current attempt.
        let Ok(current_attempt) = current_attempt_query.get(entity) else {
            continue;
        };
        if !attempt_matches_current(event_attempt, current_attempt) {
            continue;
        }
        let Some(delay) = get_delay(
            &auto_reconnect_delay_res,
            auto_reconnect_delay_query,
            entity,
        ) else {
            // no auto reconnect
            continue;
        };

        let reconnect_after = Instant::now() + delay;
        commands.entity(entity).insert(InternalReconnectAfter {
            instant: reconnect_after,
            attempt_token: event_attempt,
        });
    }
}

fn get_delay(
    auto_reconnect_delay_res: &Option<Res<AutoReconnectDelay>>,
    auto_reconnect_delay_query: Query<&AutoReconnectDelay>,
    entity: Entity,
) -> Option<Duration> {
    let delay = if let Ok(c) = auto_reconnect_delay_query.get(entity) {
        Some(c.delay)
    } else {
        auto_reconnect_delay_res.as_ref().map(|r| r.delay)
    };

    if delay == Some(Duration::MAX) {
        // if the duration is set to max, treat that as autoreconnect being disabled
        return None;
    }
    delay
}

pub fn rejoin_after_delay(
    mut commands: Commands,
    mut join_events: MessageWriter<StartJoinServerEvent>,
    query: Query<(
        Entity,
        &InternalReconnectAfter,
        &Account,
        &ConnectOpts,
        Option<&AttemptToken>,
    )>,
) {
    for (entity, reconnect_after, account, connect_opts, current_attempt) in query.iter() {
        if Instant::now() >= reconnect_after.instant {
            // don't keep trying to reconnect
            commands.entity(entity).remove::<InternalReconnectAfter>();

            // If the entity has moved on to a newer attempt, this timer is
            // stale: remove it and never start a new join attempt from it.
            if !attempt_matches_current(reconnect_after.attempt_token, current_attempt) {
                continue;
            }

            // our Entity will be reused since the account has the same uuid
            join_events.write(StartJoinServerEvent {
                account: account.clone(),
                connect_opts: connect_opts.clone(),
                start_join_callback_tx: None,
                // Automatic reconnects are brand-new join attempts: they mint
                // a fresh token instead of reusing the old attempt's identity.
                attempt_token: AttemptToken::mint(),
            });
        }
    }
}

/// A resource *and* component that indicates how long to wait before
/// reconnecting when we're kicked.
///
/// Initially, it's a resource in the ECS set to 5 seconds. You can modify
/// the resource to update the global reconnect delay, or insert it as a
/// component to set the individual delay for a single client.
///
/// You can also remove this resource from the ECS to disable the default
/// auto-reconnecting behavior. Inserting the resource/component again will not
/// make clients that were already disconnected automatically reconnect.
#[derive(Clone, Component, Debug, Resource)]
pub struct AutoReconnectDelay {
    pub delay: Duration,
}
impl AutoReconnectDelay {
    pub fn new(delay: Duration) -> Self {
        Self { delay }
    }
}

/// This is inserted when we're disconnected and indicates when we'll reconnect.
///
/// This is set based on [`AutoReconnectDelay`].
#[derive(Clone, Component, Debug)]
pub struct InternalReconnectAfter {
    pub instant: Instant,
    /// The join attempt that scheduled this timer. When the timer fires, the
    /// entity's current attempt must still equal this value; otherwise the
    /// timer is stale and is removed without starting a new attempt.
    pub attempt_token: Option<AttemptToken>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::message::Messages;

    fn reconnect_app() -> App {
        let mut app = App::new();
        app.add_message::<DisconnectEvent>()
            .add_message::<ConnectionFailedEvent>()
            .add_message::<StartJoinServerEvent>()
            .insert_resource(AutoReconnectDelay::new(DEFAULT_RECONNECT_DELAY))
            .add_systems(
                Update,
                (start_rejoin_on_disconnect, rejoin_after_delay).chain(),
            );
        app
    }

    fn connect_opts() -> ConnectOpts {
        ConnectOpts {
            address: azalea_protocol::address::ResolvedAddr {
                server: azalea_protocol::address::ServerAddr::try_from("localhost:25565").unwrap(),
                socket: "127.0.0.1:1".parse().unwrap(),
            },
            server_proxy: None,
            sessionserver_proxy: None,
        }
    }

    fn failed_event(entity: Entity, attempt_token: AttemptToken) -> ConnectionFailedEvent {
        ConnectionFailedEvent {
            entity,
            error: std::sync::Arc::new(azalea_protocol::connect::ConnectionError::Io(
                std::io::Error::other("probe"),
            )),
            attempt_token,
        }
    }

    #[test]
    fn stale_events_do_not_install_reconnect_timer() {
        let mut app = reconnect_app();
        let entity = app.world_mut().spawn_empty().id();
        let current = AttemptToken::mint();
        let stale = AttemptToken::mint();
        app.world_mut().entity_mut(entity).insert(current);

        app.world_mut().write_message(DisconnectEvent {
            entity,
            reason: None,
            attempt_token: Some(stale),
        });
        app.update();
        assert!(
            !app.world()
                .entity(entity)
                .contains::<InternalReconnectAfter>(),
            "a stale A disconnect must not install a reconnect timer for B"
        );

        app.world_mut().write_message(DisconnectEvent {
            entity,
            reason: None,
            attempt_token: Some(current),
        });
        app.update();
        let timer = app
            .world()
            .entity(entity)
            .get::<InternalReconnectAfter>()
            .expect("a matching disconnect installs a timer");
        assert_eq!(timer.attempt_token, Some(current));

        app.world_mut()
            .entity_mut(entity)
            .remove::<InternalReconnectAfter>();
        app.world_mut().write_message(failed_event(entity, stale));
        app.update();
        assert!(
            !app.world()
                .entity(entity)
                .contains::<InternalReconnectAfter>(),
            "a stale A connection failure must not install a reconnect timer for B"
        );

        app.world_mut().write_message(failed_event(entity, current));
        app.update();
        let timer = app
            .world()
            .entity(entity)
            .get::<InternalReconnectAfter>()
            .expect("a matching connection failure installs a timer");
        assert_eq!(timer.attempt_token, Some(current));
    }

    #[test]
    fn stale_reconnect_timer_is_removed_without_starting_attempt() {
        let mut app = reconnect_app();
        let entity = app.world_mut().spawn_empty().id();
        let token_b = AttemptToken::mint();
        let stale_token = AttemptToken::mint();
        app.world_mut().entity_mut(entity).insert((
            Account::offline("probe-bot"),
            connect_opts(),
            token_b,
        ));

        app.world_mut()
            .entity_mut(entity)
            .insert(InternalReconnectAfter {
                instant: Instant::now() - Duration::from_secs(1),
                attempt_token: Some(stale_token),
            });
        app.update();
        assert!(
            !app.world()
                .entity(entity)
                .contains::<InternalReconnectAfter>(),
            "the stale timer must be removed"
        );
        {
            let messages = app.world().resource::<Messages<StartJoinServerEvent>>();
            let mut cursor = messages.get_cursor();
            assert_eq!(
                cursor.read(messages).count(),
                0,
                "a stale A timer must never start a new join attempt"
            );
        }

        app.world_mut()
            .entity_mut(entity)
            .insert(InternalReconnectAfter {
                instant: Instant::now() - Duration::from_secs(1),
                attempt_token: Some(token_b),
            });
        app.update();
        assert!(
            !app.world()
                .entity(entity)
                .contains::<InternalReconnectAfter>()
        );
        let messages = app.world().resource::<Messages<StartJoinServerEvent>>();
        let mut cursor = messages.get_cursor();
        let events: Vec<_> = cursor.read(messages).collect();
        assert_eq!(events.len(), 1);
        assert_ne!(events[0].attempt_token, token_b);
        assert!(events[0].start_join_callback_tx.is_none());
    }
}
