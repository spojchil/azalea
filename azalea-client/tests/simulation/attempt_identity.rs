use azalea_client::{
    connection::RawConnection,
    disconnect::DisconnectEvent,
    join::AttemptToken,
    packet::game::{ReceiveGamePacketEvent, WorldLoadedEvent},
    test_utils::prelude::*,
};
use azalea_protocol::packets::{
    ConnectionProtocol,
    config::{ClientboundFinishConfiguration, ClientboundRegistryData},
    game::ClientboundSetHealth,
};
use azalea_registry::identifier::Identifier;
use bevy_ecs::message::Messages;
use simdnbt::owned::{NbtCompound, NbtTag};

fn setup_game_simulation() -> Simulation {
    let mut simulation = Simulation::new(ConnectionProtocol::Configuration);
    simulation.receive_packet(ClientboundRegistryData {
        registry_id: Identifier::new("minecraft:dimension_type"),
        entries: vec![(
            Identifier::new("minecraft:overworld"),
            Some(NbtCompound::from_values(vec![
                ("height".into(), NbtTag::Int(384)),
                ("min_y".into(), NbtTag::Int(-64)),
            ])),
        )]
        .into_iter()
        .collect(),
    });
    simulation.tick();
    simulation.receive_packet(ClientboundFinishConfiguration);
    simulation.tick();
    simulation
}

#[test]
fn packet_and_world_loaded_events_are_stamped_at_the_production_site() {
    let _lock = init();

    let mut simulation = setup_game_simulation();
    let attempt_token = simulation.component::<AttemptToken>();

    simulation.receive_packet(ClientboundSetHealth {
        health: 15.,
        food: 20,
        saturation: 20.,
    });
    simulation.tick();

    let messages = simulation
        .app
        .world()
        .resource::<Messages<ReceiveGamePacketEvent>>();
    let mut cursor = messages.get_cursor();
    let packets: Vec<_> = cursor.read(messages).cloned().collect();
    assert!(!packets.is_empty());
    assert!(
        packets
            .iter()
            .all(|event| event.attempt_token == attempt_token),
        "every game packet event must carry the attempt token that produced it"
    );

    simulation.receive_packet(default_login_packet());
    simulation.tick();

    let messages = simulation
        .app
        .world()
        .resource::<Messages<WorldLoadedEvent>>();
    let mut cursor = messages.get_cursor();
    let worlds: Vec<_> = cursor.read(messages).cloned().collect();
    assert!(!worlds.is_empty());
    assert!(
        worlds
            .iter()
            .all(|event| event.attempt_token == attempt_token),
        "every world-loaded event must carry the attempt token that produced it"
    );
}

#[test]
fn stale_disconnect_cannot_remove_current_attempt_and_matching_disconnect_cleans_up() {
    let _lock = init();

    let mut simulation = setup_game_simulation();
    let current_token = simulation.component::<AttemptToken>();
    let stale_token = AttemptToken::mint();
    assert_ne!(stale_token, current_token);

    simulation.write_message(DisconnectEvent {
        entity: simulation.entity,
        reason: None,
        attempt_token: Some(stale_token),
    });
    simulation.tick();
    assert!(
        simulation.has_component::<RawConnection>(),
        "a stale DisconnectEvent(A) must not remove the current attempt B"
    );

    simulation.write_message(DisconnectEvent {
        entity: simulation.entity,
        reason: None,
        attempt_token: Some(current_token),
    });
    simulation.tick();
    assert!(
        !simulation.has_component::<RawConnection>(),
        "a matching DisconnectEvent(B) must clean up normally"
    );
}
