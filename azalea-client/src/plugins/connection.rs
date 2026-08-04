use std::{
    fmt::Debug,
    io::Cursor,
    mem,
    sync::{
        Arc,
        atomic::{self, AtomicBool},
    },
};

use azalea_crypto::Aes128CfbEnc;
use azalea_protocol::{
    connect::{RawReadConnection, RawWriteConnection},
    packets::{
        ConnectionProtocol, Packet, ProtocolPacket, config::ClientboundConfigPacket,
        game::ClientboundGamePacket, login::ClientboundLoginPacket,
    },
    read::{ReadPacketError, deserialize_packet},
    write::serialize_packet,
};
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_tasks::{IoTaskPool, futures_lite::future};
use thiserror::Error;
use tokio::{
    io::AsyncWriteExt,
    net::tcp::OwnedWriteHalf,
    sync::mpsc::{self},
};
use tracing::{debug, error, info, trace};

use super::packet::{
    config::ReceiveConfigPacketEvent, game::ReceiveGamePacketEvent, login::ReceiveLoginPacketEvent,
};
use crate::join::AttemptToken;
use crate::packet::{config, game, login};

pub struct ConnectionPlugin;
impl Plugin for ConnectionPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PreUpdate, (read_packets, poll_all_writer_tasks).chain());
    }
}

pub fn read_packets(ecs: &mut World) {
    let mut entity_and_conn_query = ecs.query::<(Entity, &mut RawConnection)>();
    let mut conn_query = ecs.query::<&mut RawConnection>();

    let mut entities_handling_packets = Vec::new();
    let mut entities_with_injected_packets = Vec::new();
    for (entity, mut raw_conn) in entity_and_conn_query.iter_mut(ecs) {
        if !raw_conn.injected_clientbound_packets.is_empty() {
            entities_with_injected_packets.push((
                entity,
                mem::take(&mut raw_conn.injected_clientbound_packets),
            ));
        }

        if raw_conn.network.is_none() {
            // no network connection, don't bother with the normal packet handling
            continue;
        }

        entities_handling_packets.push(entity);
    }

    let mut queued_packet_events = QueuedPacketEvents::default();

    // handle injected packets, see the comment on
    // RawConnection::injected_clientbound_packets for more info
    for (entity, raw_packets) in entities_with_injected_packets {
        for raw_packet in raw_packets {
            let conn = conn_query.get(ecs, entity).unwrap();
            let state = conn.state;
            let attempt_token = conn.attempt_token();

            trace!("Received injected packet with bytes: {raw_packet:?}");
            if let Err(e) = handle_raw_packet(
                ecs,
                &raw_packet,
                entity,
                state,
                attempt_token,
                &mut queued_packet_events,
            ) {
                error!("Error reading injected packet: {e}");
            }
        }
    }

    for entity in entities_handling_packets {
        loop {
            let mut conn = conn_query.get_mut(ecs, entity).unwrap();
            let net_conn = conn.net_conn().unwrap();
            let read_res = net_conn.reader.try_read();
            let state = conn.state;
            let attempt_token = conn.attempt_token();
            match read_res {
                Ok(Some(raw_packet)) => {
                    let raw_packet = Arc::<[u8]>::from(raw_packet);
                    if let Err(e) = handle_raw_packet(
                        ecs,
                        &raw_packet,
                        entity,
                        state,
                        attempt_token,
                        &mut queued_packet_events,
                    ) {
                        error!("Error reading packet: {e}");
                    }
                }
                Ok(None) => {
                    // no packets available
                    break;
                }
                Err(err) => {
                    log_for_error(&err);

                    if matches!(
                        &*err,
                        ReadPacketError::IoError { .. } | ReadPacketError::ConnectionClosed
                    ) {
                        info!("Server closed connection");
                        // ungraceful disconnect :(
                        conn.network = None;
                        // setting this will make us send a DisconnectEvent
                        conn.is_alive = false;
                    }

                    break;
                }
            }
        }
    }

    queued_packet_events.write_messages(ecs);
}

fn poll_all_writer_tasks(mut conn_query: Query<&mut RawConnection>) {
    for mut conn in conn_query.iter_mut() {
        if let Some(net_conn) = &mut conn.network {
            // this needs to be done at some point every update to make sure packets are
            // actually sent to the network

            if net_conn.poll_writer().is_some() {
                // means the writer task ended
                conn.network = None;
                conn.is_alive = false;
            }
        }
    }
}

#[derive(Default)]
pub struct QueuedPacketEvents {
    login: Vec<ReceiveLoginPacketEvent>,
    config: Vec<ReceiveConfigPacketEvent>,
    game: Vec<ReceiveGamePacketEvent>,
}
impl QueuedPacketEvents {
    fn write_messages(&mut self, ecs: &mut World) {
        ecs.write_message_batch(self.login.drain(..));
        ecs.write_message_batch(self.config.drain(..));
        ecs.write_message_batch(self.game.drain(..));
    }
}

fn log_for_error(error: &ReadPacketError) {
    if !matches!(*error, ReadPacketError::ConnectionClosed) {
        error!("Error reading packet from Client: {error:?}");
    }
}

/// The client's connection to the server.
#[derive(Component)]
pub struct RawConnection {
    /// The network connection to the server.
    ///
    /// This isn't guaranteed to be present, for example during the main packet
    /// handlers or at all times during tests.
    ///
    /// You shouldn't rely on this. Instead, use the events for sending packets
    /// like [`SendGamePacketEvent`](crate::packet::game::SendGamePacketEvent) /
    /// [`SendConfigPacketEvent`](crate::packet::config::SendConfigPacketEvent)
    /// / [`SendLoginPacketEvent`](crate::packet::login::SendLoginPacketEvent).
    ///
    /// To check if we haven't disconnected from the server, use
    /// [`Self::is_alive`].
    pub(crate) network: Option<NetworkConnection>,
    pub state: ConnectionProtocol,
    pub(crate) is_alive: bool,

    /// This exists for internal testing purposes and probably shouldn't be used
    /// for normal bots.
    ///
    /// It's basically a way to make our client think it received a packet from
    /// the server without needing to interact with the network.
    pub injected_clientbound_packets: Vec<Box<[u8]>>,
    /// The join attempt that installed this connection. Events produced from
    /// this connection are stamped with this identity at the production site.
    attempt_token: AttemptToken,
}
impl RawConnection {
    pub fn new(
        reader: RawReadConnection,
        writer: RawWriteConnection,
        state: ConnectionProtocol,
        attempt_token: AttemptToken,
    ) -> Self {
        let task_pool = IoTaskPool::get();

        let (network_packet_writer_tx, network_packet_writer_rx) =
            mpsc::unbounded_channel::<Box<[u8]>>();

        let writer_task =
            task_pool.spawn(write_task(network_packet_writer_rx, writer.write_stream));

        let mut conn = Self::new_networkless_with_token(state, attempt_token);
        conn.network = Some(NetworkConnection {
            reader,
            enc_cipher: writer.enc_cipher,
            network_packet_writer_tx,
            writer_task,
        });

        conn
    }

    pub fn new_networkless(state: ConnectionProtocol) -> Self {
        // Legacy/test convenience constructor: each networkless connection is
        // its own identity, so a fresh token is minted per object. Production
        // join attempts pass their attempt token to `RawConnection::new`.
        Self::new_networkless_with_token(state, AttemptToken::mint())
    }

    pub fn new_networkless_with_token(
        state: ConnectionProtocol,
        attempt_token: AttemptToken,
    ) -> Self {
        Self {
            network: None,
            state,
            is_alive: true,
            injected_clientbound_packets: Vec::new(),
            attempt_token,
        }
    }

    pub fn is_alive(&self) -> bool {
        self.is_alive
    }

    /// The immutable join-attempt identity of this connection.
    pub fn attempt_token(&self) -> AttemptToken {
        self.attempt_token
    }

    /// Write a packet to the server without emitting any events.
    ///
    /// This is called by the handlers for [`SendGamePacketEvent`],
    /// [`SendConfigPacketEvent`], and [`SendLoginPacketEvent`].
    ///
    /// [`SendGamePacketEvent`]: crate::packet::game::SendGamePacketEvent
    /// [`SendConfigPacketEvent`]: crate::packet::config::SendConfigPacketEvent
    /// [`SendLoginPacketEvent`]: crate::packet::login::SendLoginPacketEvent
    pub fn write<P: ProtocolPacket + Debug>(
        &mut self,
        packet: impl Packet<P>,
    ) -> Result<(), WritePacketError> {
        if let Some(network) = &mut self.network {
            network.write(packet)?;
        } else {
            static WARNED: AtomicBool = AtomicBool::new(false);
            if !WARNED.swap(true, atomic::Ordering::Relaxed) {
                debug!(
                    "tried to write packet to the network but there is no NetworkConnection. if you're trying to send a packet from the handler function, use self.write instead"
                );
            }
        }
        Ok(())
    }

    pub fn net_conn(&mut self) -> Option<&mut NetworkConnection> {
        self.network.as_mut()
    }
}

pub fn handle_raw_packet(
    ecs: &mut World,
    raw_packet: &[u8],
    entity: Entity,
    state: ConnectionProtocol,
    attempt_token: AttemptToken,
    queued_packet_events: &mut QueuedPacketEvents,
) -> Result<(), Box<ReadPacketError>> {
    // Fail-closed source admission. A packet may only be deserialized and
    // processed when the entity's current attempt token is exactly the token
    // of the connection that produced it. During entity reuse, a stale A
    // `RawConnection` can still be attached while the entity's current
    // attempt is already B; such packets must be ignored before any
    // deserialization or ECS mutation (which could otherwise drive derived
    // events/Added components into B's channel). An entity that is missing or
    // tokenless is also ignored; legacy packets are never back-stamped from
    // the current attempt.
    let Some(current_attempt_token) = ecs.get::<AttemptToken>(entity).copied() else {
        return Ok(());
    };
    if current_attempt_token != attempt_token {
        return Ok(());
    }

    let stream = &mut Cursor::new(raw_packet);
    match state {
        ConnectionProtocol::Handshake => {
            unreachable!()
        }
        ConnectionProtocol::Game => {
            let packet = Arc::new(deserialize_packet::<ClientboundGamePacket>(stream)?);
            trace!("Packet: {packet:?}");
            game::process_packet(ecs, entity, packet.as_ref(), attempt_token);
            queued_packet_events.game.push(ReceiveGamePacketEvent {
                entity,
                packet,
                attempt_token,
            });
        }
        ConnectionProtocol::Status => {
            unreachable!()
        }
        ConnectionProtocol::Login => {
            let packet = Arc::new(deserialize_packet::<ClientboundLoginPacket>(stream)?);
            trace!("Packet: {packet:?}");
            login::process_packet(ecs, entity, &packet, attempt_token);
            queued_packet_events
                .login
                .push(ReceiveLoginPacketEvent { entity, packet });
        }
        ConnectionProtocol::Configuration => {
            let packet = Arc::new(deserialize_packet::<ClientboundConfigPacket>(stream)?);
            trace!("Packet: {packet:?}");
            config::process_packet(ecs, entity, &packet, attempt_token);
            queued_packet_events
                .config
                .push(ReceiveConfigPacketEvent { entity, packet });
        }
    };

    Ok(())
}

pub struct NetworkConnection {
    reader: RawReadConnection,
    // compression threshold is in the RawReadConnection
    pub enc_cipher: Option<Aes128CfbEnc>,

    pub writer_task: bevy_tasks::Task<()>,
    /// A queue of raw TCP packets to send.
    ///
    /// These will not be modified further, they should already be serialized
    /// and compressed and encrypted before being added here.
    network_packet_writer_tx: mpsc::UnboundedSender<Box<[u8]>>,
}
impl NetworkConnection {
    pub fn write<P: ProtocolPacket + Debug>(
        &mut self,
        packet: impl Packet<P>,
    ) -> Result<(), WritePacketError> {
        let packet = packet.into_variant();
        let raw_packet = serialize_packet(&packet)?;
        self.write_raw(&raw_packet)?;

        Ok(())
    }

    pub fn write_raw(&mut self, raw_packet: &[u8]) -> Result<(), WritePacketError> {
        let network_packet = azalea_protocol::write::encode_to_network_packet(
            raw_packet,
            self.reader.compression_threshold,
            &mut self.enc_cipher,
        );
        self.network_packet_writer_tx
            .send(network_packet.into_boxed_slice())?;
        Ok(())
    }

    /// Makes sure packets get sent and returns Some(()) if the connection has
    /// closed.
    pub fn poll_writer(&mut self) -> Option<()> {
        let poll_once_res = future::poll_once(&mut self.writer_task);
        future::block_on(poll_once_res)
    }

    pub fn set_compression_threshold(&mut self, threshold: Option<u32>) {
        trace!("Set compression threshold to {threshold:?}");
        self.reader.compression_threshold = threshold;
    }
    /// Set the encryption key that is used to encrypt and decrypt packets.
    ///
    /// The same key is used for both reading and writing.
    pub fn set_encryption_key(&mut self, key: [u8; 16]) {
        trace!("Enabled protocol encryption");
        let (enc_cipher, dec_cipher) = azalea_crypto::create_cipher(&key);
        self.reader.dec_cipher = Some(dec_cipher);
        self.enc_cipher = Some(enc_cipher);
    }
}

async fn write_task(
    mut network_packet_writer_rx: mpsc::UnboundedReceiver<Box<[u8]>>,
    mut write_half: OwnedWriteHalf,
) {
    while let Some(network_packet) = network_packet_writer_rx.recv().await {
        if let Err(e) = write_half.write_all(&network_packet).await {
            debug!("Error writing packet to server: {e}");
            break;
        };
    }

    trace!("write task is done");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_player::Hunger;
    use azalea_entity::metadata::Health;
    use azalea_protocol::write::serialize_packet;

    fn set_health_bytes(health: f32) -> Box<[u8]> {
        serialize_packet(
            &azalea_protocol::packets::game::ClientboundSetHealth {
                health,
                food: 20,
                saturation: 20.,
            }
            .into_variant(),
        )
        .unwrap()
    }

    #[test]
    fn stale_attempt_packets_are_ignored_before_parsing() {
        let mut world = World::new();
        let entity = world.spawn_empty().id();
        let token_a = AttemptToken::mint();
        let token_b = AttemptToken::mint();
        world.entity_mut(entity).insert((
            token_b,
            Health(10.),
            Hunger {
                food: 5,
                saturation: 5.,
            },
        ));

        // Invalid bytes with the stale A token: safely ignored, nothing
        // queued, and the parser is never reached.
        let mut queued = QueuedPacketEvents::default();
        let result = handle_raw_packet(
            &mut world,
            &[0xFF],
            entity,
            ConnectionProtocol::Game,
            token_a,
            &mut queued,
        );
        assert!(result.is_ok());
        assert!(queued.game.is_empty());
        assert_eq!(world.get::<Health>(entity).unwrap().0, 10.);

        // The same invalid bytes with the matching B token reach the parser
        // and produce an error, proving the stale branch short-circuits before
        // deserialization.
        let result = handle_raw_packet(
            &mut world,
            &[0xFF],
            entity,
            ConnectionProtocol::Game,
            token_b,
            &mut queued,
        );
        assert!(
            result.is_err(),
            "a matching token must reach the parser and fail on invalid bytes"
        );
    }

    #[test]
    fn missing_or_tokenless_entities_are_fail_closed() {
        let mut world = World::new();
        let token = AttemptToken::mint();

        let tokenless = world.spawn_empty().id();
        let mut queued = QueuedPacketEvents::default();
        let result = handle_raw_packet(
            &mut world,
            &[0xFF],
            tokenless,
            ConnectionProtocol::Game,
            token,
            &mut queued,
        );
        assert!(result.is_ok());
        assert!(queued.game.is_empty());

        let missing = world.spawn_empty().id();
        world.despawn(missing);
        let result = handle_raw_packet(
            &mut world,
            &[0xFF],
            missing,
            ConnectionProtocol::Game,
            token,
            &mut queued,
        );
        assert!(result.is_ok());
        assert!(queued.game.is_empty());
    }

    #[test]
    fn stale_serializable_packet_does_not_mutate_ecs_and_matching_applies() {
        let mut world = World::new();
        let entity = world.spawn_empty().id();
        let token_a = AttemptToken::mint();
        let token_b = AttemptToken::mint();
        world.entity_mut(entity).insert((
            token_b,
            Health(10.),
            Hunger {
                food: 5,
                saturation: 5.,
            },
        ));

        let bytes = set_health_bytes(15.);
        let mut queued = QueuedPacketEvents::default();

        let result = handle_raw_packet(
            &mut world,
            &bytes,
            entity,
            ConnectionProtocol::Game,
            token_a,
            &mut queued,
        );
        assert!(result.is_ok());
        assert!(queued.game.is_empty());
        assert_eq!(
            world.get::<Health>(entity).unwrap().0,
            10.,
            "a stale A packet must not mutate the current attempt B's ECS"
        );

        let result = handle_raw_packet(
            &mut world,
            &bytes,
            entity,
            ConnectionProtocol::Game,
            token_b,
            &mut queued,
        );
        assert!(result.is_ok());
        assert_eq!(queued.game.len(), 1);
        assert_eq!(queued.game[0].attempt_token, token_b);
        assert_eq!(world.get::<Health>(entity).unwrap().0, 15.);
    }
}

#[derive(Debug, Error)]
pub enum WritePacketError {
    #[error("Wrong protocol state: expected {expected:?}, got {got:?}")]
    WrongState {
        expected: ConnectionProtocol,
        got: ConnectionProtocol,
    },
    #[error(transparent)]
    Encoding(#[from] azalea_protocol::write::PacketEncodeError),
    #[error(transparent)]
    SendError {
        #[from]
        #[backtrace]
        source: mpsc::error::SendError<Box<[u8]>>,
    },
}
