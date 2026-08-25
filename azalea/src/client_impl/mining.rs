use azalea_client::{
    interact::pick::HitResultComponent,
    mining::{LeftClickMine, Mining, MiningQueued, forced_mining_direction},
};
use azalea_core::{direction::Direction, position::BlockPos};

use crate::Client;

impl Client {
    pub fn start_mining(&self, position: BlockPos) {
        let mut ecs = self.ecs.write();
        let direction = ecs
            .get::<HitResultComponent>(self.entity)
            .map_or(Direction::Down, |hit_result| {
                forced_mining_direction(hit_result, position)
            });

        // `Client::start_mining` is the force=true API. Make its accepted
        // request visible before releasing the ECS lock so callers can
        // distinguish this queued state from an interrupted request. The
        // message-based API remains available for systems and force=false.
        ecs.entity_mut(self.entity).insert(MiningQueued {
            position,
            direction,
            force: true,
        });
    }

    /// Returns true if the client is currently trying to mine a block.
    pub fn is_mining(&self) -> bool {
        let ecs = self.ecs.read();
        ecs.get::<Mining>(self.entity).is_some() || ecs.get::<MiningQueued>(self.entity).is_some()
    }

    /// When enabled, the bot will mine any block that it is looking at if it is
    /// reachable.
    pub fn left_click_mine(&self, enabled: bool) {
        let mut ecs = self.ecs.write();
        let mut entity_mut = ecs.entity_mut(self.entity);

        if enabled {
            entity_mut.insert(LeftClickMine);
        } else {
            entity_mut.remove::<LeftClickMine>();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use azalea_client::mining::MiningQueued;
    use bevy_ecs::world::World;
    use parking_lot::RwLock;

    use super::*;

    #[test]
    fn start_mining_publishes_the_queued_target_synchronously() {
        let mut world = World::new();
        let entity = world.spawn_empty().id();
        let ecs = Arc::new(RwLock::new(world));
        let client = Client::new(entity, ecs.clone());
        let target = BlockPos::new(4, 70, -3);

        client.start_mining(target);

        assert!(client.is_mining());
        let ecs = ecs.read();
        let queued = ecs.get::<MiningQueued>(entity).unwrap();
        assert_eq!(queued.position, target);
        assert_eq!(queued.direction, Direction::Down);
        assert!(queued.force);
    }
}
