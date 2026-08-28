use azalea_block::BlockState;
use azalea_core::position::BlockPos;
use bevy_app::{App, Plugin, Update};
use bevy_ecs::prelude::*;

use crate::{
    chunks::handle_receive_chunk_event, interact::BlockStatePredictionHandler,
    local_player::WorldHolder,
};

pub struct BlockUpdatePlugin;
impl Plugin for BlockUpdatePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            // has to be after ReceiveChunkEvent is handled so if we get chunk+blockupdate in one
            // Update then the block update actually gets applied
            handle_block_update_event.after(handle_receive_chunk_event),
        );
    }
}

/// A component that holds the list of block updates that need to be handled.
///
/// This is updated by `read_packets` (in `PreUpdate`) and handled/cleared by
/// [`handle_block_update_event`] (`Update`).
///
/// This is a component instead of an ECS event for performance reasons.
#[derive(Clone, Component, Debug, Default)]
pub struct QueuedServerBlockUpdates {
    pub list: Vec<(BlockPos, BlockState)>,
}

/// Block changes the server has confirmed at positions this client predicted.
///
/// An update lands here only when
/// [`BlockStatePredictionHandler::update_known_server_state`] matched a pending
/// prediction, i.e. the position was registered by one of our own actions
/// (mining, placing). Everything else is a passive update about the
/// world and is applied to the world model as before. Consumers drain this list
/// to learn what their own actions did to the world.
///
/// Optional in [`handle_block_update_event`]: clients that never insert it keep
/// the previous behavior exactly.
#[derive(Clone, Component, Debug, Default)]
pub struct ConfirmedSelfBlockUpdates {
    pub list: Vec<(BlockPos, BlockState)>,
}

pub fn handle_block_update_event(
    mut query: Query<(
        &mut QueuedServerBlockUpdates,
        &WorldHolder,
        &mut BlockStatePredictionHandler,
        Option<&mut ConfirmedSelfBlockUpdates>,
    )>,
) {
    for (mut queued, world_holder, mut prediction_handler, mut confirmed_self) in query.iter_mut() {
        let world = world_holder.shared.read();
        for (pos, block_state) in queued.list.drain(..) {
            if prediction_handler.update_known_server_state(pos, block_state) {
                if let Some(ref mut confirmed_self) = confirmed_self {
                    confirmed_self.list.push((pos, block_state));
                }
            } else {
                world.chunks.set_block_state(pos, block_state);
            }
        }
    }
}
