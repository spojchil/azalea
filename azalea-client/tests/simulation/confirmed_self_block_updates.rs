use azalea_client::{
    block_update::ConfirmedSelfBlockUpdates, mining::StartMiningBlockEvent, test_utils::prelude::*,
};
use azalea_core::position::{BlockPos, ChunkPos};
use azalea_protocol::packets::{
    ConnectionProtocol,
    game::{ClientboundBlockChangedAck, ClientboundBlockUpdate},
};
use azalea_registry::builtin::BlockKind;

fn confirmed(simulation: &mut Simulation) -> Vec<(BlockPos, azalea_block::BlockState)> {
    simulation.query_self::<&mut ConfirmedSelfBlockUpdates, _>(|mut updates| {
        std::mem::take(&mut updates.list)
    })
}

fn arm(simulation: &mut Simulation) {
    simulation
        .app
        .world_mut()
        .entity_mut(simulation.entity)
        .insert(ConfirmedSelfBlockUpdates::default());
}

fn mine_at(simulation: &mut Simulation, pos: BlockPos) {
    simulation.receive_packet(ClientboundBlockUpdate {
        pos,
        // tnt is insta-mineable, so mining finishes within one tick
        block_state: BlockKind::Tnt.into(),
    });
    simulation.tick();
    arm(simulation);
    simulation.write_message(StartMiningBlockEvent {
        entity: simulation.entity,
        position: pos,
        force: true,
    });
    simulation.tick();
}

/// A block update and its acknowledgement arriving in the same batch is the
/// normal case — measured 2ms apart on the wire against a real server. The
/// update is only queued during `PreUpdate` while the ack is handled inline, so
/// the ack must match the queued update against the pending prediction before
/// settling it. Otherwise the prediction is already gone by the time the queue
/// drains in `Update`, the update looks passive, and nothing is ever reported
/// as self-caused.
#[test]
fn ack_in_the_same_batch_still_reports_the_change_as_self_caused() {
    let _lock = init();

    let mut simulation = Simulation::new(ConnectionProtocol::Game);
    simulation.receive_packet(default_login_packet());
    simulation.receive_packet(make_basic_empty_chunk(ChunkPos::new(0, 0), (384 + 64) / 16));
    simulation.tick();

    let pos = BlockPos::new(1, 2, 3);
    mine_at(&mut simulation, pos);
    assert_eq!(simulation.get_block_state(pos), Some(BlockKind::Air.into()));

    simulation.receive_packet(ClientboundBlockUpdate {
        pos,
        block_state: BlockKind::Air.into(),
    });
    simulation.receive_packet(ClientboundBlockChangedAck { seq: 1 });
    simulation.tick();

    assert_eq!(
        confirmed(&mut simulation),
        vec![(pos, BlockKind::Air.into())],
        "the block we just mined has to be reported as our own confirmed change"
    );
    assert_eq!(simulation.get_block_state(pos), Some(BlockKind::Air.into()));
}

/// A passive update — one the server sends about a position we never acted on —
/// must not be reported as self-caused. This is what keeps a fog-of-war
/// consumer from learning about blocks it has no business knowing.
#[test]
fn a_passive_update_is_not_reported_as_self_caused() {
    let _lock = init();

    let mut simulation = Simulation::new(ConnectionProtocol::Game);
    simulation.receive_packet(default_login_packet());
    simulation.receive_packet(make_basic_empty_chunk(ChunkPos::new(0, 0), (384 + 64) / 16));
    simulation.tick();
    arm(&mut simulation);

    // Somebody else changed a block far away, and an unrelated ack arrives.
    let elsewhere = BlockPos::new(9, 2, 9);
    simulation.receive_packet(ClientboundBlockUpdate {
        pos: elsewhere,
        block_state: BlockKind::Stone.into(),
    });
    simulation.receive_packet(ClientboundBlockChangedAck { seq: 1 });
    simulation.tick();

    assert!(
        confirmed(&mut simulation).is_empty(),
        "we never acted there, so it is not our change"
    );
    assert_eq!(
        simulation.get_block_state(elsewhere),
        Some(BlockKind::Stone.into()),
        "it still has to reach the world model as a passive update"
    );
}

/// The server acknowledging without sending an update means the action did
/// nothing (out of reach, rejected, a right-click that opened a screen). The
/// world rolls back to the retained state and nothing is reported.
#[test]
fn an_ack_without_an_update_rolls_back_and_reports_nothing() {
    let _lock = init();

    let mut simulation = Simulation::new(ConnectionProtocol::Game);
    simulation.receive_packet(default_login_packet());
    simulation.receive_packet(make_basic_empty_chunk(ChunkPos::new(0, 0), (384 + 64) / 16));
    simulation.tick();

    let pos = BlockPos::new(1, 2, 3);
    mine_at(&mut simulation, pos);

    simulation.receive_packet(ClientboundBlockChangedAck { seq: 1 });
    simulation.tick();

    assert!(
        confirmed(&mut simulation).is_empty(),
        "a rollback is not a confirmed change"
    );
    assert_eq!(
        simulation.get_block_state(pos),
        Some(BlockKind::Tnt.into()),
        "the block comes back — the server never agreed we broke it"
    );
}
