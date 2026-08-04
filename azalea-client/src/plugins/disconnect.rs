//! Disconnect a client from the server.

use azalea_chat::FormattedText;
use azalea_core::entity_id::MinecraftEntityId;
use azalea_entity::{
    EntityBundle, HasClientLoaded, InLoadedChunk, LocalEntity, metadata::PlayerMetadataBundle,
};
use bevy_app::{App, Plugin, PostUpdate};
use bevy_ecs::prelude::*;
use derive_more::Deref;
use tracing::info;

use super::join::AttemptToken;
use super::login::IsAuthenticated;
#[cfg(feature = "online-mode")]
use crate::chat_signing;
use crate::{
    client::JoinedClientBundle, connection::RawConnection, local_player::WorldHolder, mining,
    tick_counter::TicksConnected,
};

pub struct DisconnectPlugin;
impl Plugin for DisconnectPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<DisconnectEvent>().add_systems(
            PostUpdate,
            (
                update_read_packets_task_running_component,
                remove_components_from_disconnected_players,
                // this happens after `remove_components_from_disconnected_players` since that
                // system removes `IsConnectionAlive`, which ensures that
                // `DisconnectEvent` won't get called again from
                // `disconnect_on_connection_dead`
                disconnect_on_connection_dead,
            )
                .chain(),
        );
    }
}

/// An event sent when a client got disconnected from the server.
///
/// If the client was kicked with a reason, that reason will be present in the
/// [`reason`](DisconnectEvent::reason) field.
///
/// This event won't be sent if creating the initial connection to the server
/// failed, for that see [`ConnectionFailedEvent`].
///
/// [`ConnectionFailedEvent`]: crate::join::ConnectionFailedEvent

#[derive(Message)]
pub struct DisconnectEvent {
    pub entity: Entity,
    pub reason: Option<FormattedText>,
    /// The join attempt that produced this disconnect. `None` is only valid
    /// for legacy/test event writers; production events are stamped at their
    /// production site and are never matched against the entity's "current"
    /// attempt at read time.
    pub attempt_token: Option<AttemptToken>,
}

/// A bundle of components that are removed when a client disconnects.
///
/// This shouldn't be used for inserts because not all of the components should
/// always be present.
#[derive(Bundle)]
pub struct RemoveOnDisconnectBundle {
    pub joined_client: JoinedClientBundle,

    pub entity: EntityBundle,
    pub minecraft_entity_id: MinecraftEntityId,
    pub world_holder: WorldHolder,
    pub player_metadata: PlayerMetadataBundle,
    pub in_loaded_chunk: InLoadedChunk,
    //// This makes it close the TCP connection.
    pub raw_connection: RawConnection,
    /// This makes it not send [`DisconnectEvent`] again.
    pub is_connection_alive: IsConnectionAlive,
    /// Resend our chat signing certs next time.
    #[cfg(feature = "online-mode")]
    pub chat_signing_session: chat_signing::ChatSigningSession,
    /// They're not authenticated anymore if they disconnected.
    pub is_authenticated: IsAuthenticated,
    // send ServerboundPlayerLoaded next time we join.
    pub has_client_loaded: HasClientLoaded,
    // TickCounter is reset on reconnect
    pub ticks_alive: TicksConnected,

    // the rest of the mining components are already removed, as JoinedClientBundle includes
    // MineBundle
    pub mining: mining::Mining,
}

/// A system that removes the several components from our clients when they get
/// a [`DisconnectEvent`].
pub fn remove_components_from_disconnected_players(
    mut commands: Commands,
    mut events: MessageReader<DisconnectEvent>,
    mut loaded_by_query: Query<&mut azalea_entity::LoadedBy>,
    current_attempt: Query<Option<&AttemptToken>>,
) {
    for DisconnectEvent {
        entity,
        reason,
        attempt_token,
    } in events.read()
    {
        let event_token = *attempt_token;
        let matches_current_attempt = match (event_token, current_attempt.get(*entity)) {
            (Some(event_token), Ok(Some(current_token))) => event_token == *current_token,
            // Legacy/test disconnect without an identity is only honored on an
            // entity that also has no current attempt identity. A tokenless
            // event must never fall back to "the entity's current attempt".
            (None, Ok(None)) => true,
            _ => false,
        };
        if !matches_current_attempt {
            tracing::debug!("Ignoring disconnect event for a stale attempt on entity {entity:?}");
            continue;
        }
        info!(
            "A client {entity:?} was disconnected{}",
            if let Some(reason) = reason {
                format!(": {reason}")
            } else {
                "".to_owned()
            }
        );
        commands
            .entity(*entity)
            .remove::<RemoveOnDisconnectBundle>();
        // note that we don't remove the client from the ECS, so if they decide
        // to reconnect they'll keep their state

        // now we have to remove ourselves from the LoadedBy for every entity.
        // in theory this could be inefficient if we have massive swarms... but in
        // practice this is fine.
        for mut loaded_by in &mut loaded_by_query.iter_mut() {
            loaded_by.remove(entity);
        }
    }
}

#[derive(Clone, Component, Copy, Debug, Deref)]
pub struct IsConnectionAlive(bool);

fn update_read_packets_task_running_component(
    query: Query<(Entity, &RawConnection)>,
    mut commands: Commands,
) {
    for (entity, raw_connection) in &query {
        let running = raw_connection.is_alive();
        commands.entity(entity).insert(IsConnectionAlive(running));
    }
}

#[allow(clippy::type_complexity)]
fn disconnect_on_connection_dead(
    query: Query<
        (Entity, &IsConnectionAlive, &RawConnection),
        (Changed<IsConnectionAlive>, With<LocalEntity>),
    >,
    mut disconnect_events: MessageWriter<DisconnectEvent>,
) {
    for (entity, &is_connection_alive, raw_connection) in &query {
        if !*is_connection_alive {
            disconnect_events.write(DisconnectEvent {
                entity,
                reason: None,
                attempt_token: Some(raw_connection.attempt_token()),
            });
        }
    }
}
