use core::f32;
use std::{
    array,
    cell::{RefCell, UnsafeCell},
    fmt::Debug,
    mem,
    sync::Arc,
};

use azalea_block::{
    BlockState,
    properties::{self, SlabKind, StairShape},
};
use azalea_core::{
    bitset::FastFixedBitSet,
    position::{BlockPos, ChunkPos, ChunkSectionBlockPos},
};
use azalea_physics::collision::BlockWithShape;
use azalea_registry::builtin::BlockKind;
use azalea_world::{World, palette::PalettedContainer};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;

use super::{mining::MiningCache, positions::RelBlockPos};
use crate::pathfinder::positions::SmallChunkSectionPos;

const MAX_VIEW_DISTANCE: usize = 32;

/// Where the pathfinder reads blocks from.
///
/// The default is the client's loaded world, which contains every chunk the
/// server has sent — including places the bot has never looked at. Callers that
/// must plan only over what the bot has actually observed can install their own
/// source instead; see [`CachedWorld::with_block_source`].
///
/// `None` means **unknown**. It is replaced with
/// [`BlockSource::missing_block_state`], which defaults to air for backwards
/// compatibility. Strict sources can return an unwalkable, non-standable state
/// there so unknown cells never enter the executable graph. Mining costs never
/// fall through to the loaded world: an unknown cell can only inherit the
/// fallback's geometry, and is never treated as a known block to break.
pub trait BlockSource: Send + Sync {
    fn get_block_state(&self, pos: BlockPos) -> Option<BlockState>;

    fn missing_block_state(&self) -> BlockState {
        BlockState::AIR
    }
}

/// Install a [`BlockSource`] on an entity so initial search, mining costs,
/// replanning, and path patching use it instead of the loaded world.
#[derive(bevy_ecs::component::Component, Clone)]
pub struct PathfinderBlockSource(pub Arc<dyn BlockSource>);

/// An efficient representation of the world used for the pathfinder.
pub struct CachedWorld {
    /// The origin that the [`RelBlockPos`] types will be relative to.
    ///
    /// This is for an optimization that reduces the size of the block positions
    /// that are used by the pathfinder.
    origin: BlockPos,

    min_y: i32,
    world_lock: Arc<RwLock<World>>,

    // we use the bounded cache by default and then switch if it gets too big
    bounded_chunk_cache: RefCell<[(ChunkPos, CachedChunk); MAX_VIEW_DISTANCE * MAX_VIEW_DISTANCE]>,
    unbounded_chunk_cache: RefCell<FxHashMap<ChunkPos, CachedChunk>>,

    cached_blocks: UnsafeCell<CachedSections>,

    #[allow(clippy::type_complexity)]
    cached_mining_costs: UnsafeCell<Option<Box<[(RelBlockPos, f32)]>>>,

    /// Installed by [`CachedWorld::with_block_source`]; `None` reads the world.
    block_source: Option<Arc<dyn BlockSource>>,
}

// we store `PalettedContainer`s instead of `Chunk`s or `Section`s because it
// doesn't contain any unnecessary data like heightmaps or biomes.
type CachedChunk = Box<[PalettedContainer<BlockState>]>;

pub struct CachedSections {
    pub fast_sections: Box<[Option<CachedSection>; FAST_SECTIONS_CACHE_SIZE]>,
    pub fallback_sections: Vec<CachedSection>,
}

const FAST_SECTIONS_CACHE_SIZE: usize = 16 * 16 * 16;
fn fast_section_idx(pos: SmallChunkSectionPos) -> usize {
    (pos.y as usize % 16) + (pos.x as usize % 16) * 16 + (pos.z as usize % 16) * 16 * 16
}

impl CachedSections {
    pub fn get_mut(&mut self, pos: SmallChunkSectionPos) -> Option<&mut CachedSection> {
        let idx = fast_section_idx(pos);

        if let Some(fast_item) = &mut self.fast_sections[idx]
            && fast_item.pos == pos
        {
            return Some(fast_item);
        }

        if let Some(item) = self.fallback_sections.iter_mut().find(|s| s.pos == pos) {
            return Some(item);
        }

        None
    }

    #[inline]
    pub fn insert(&mut self, section: CachedSection) {
        let idx = fast_section_idx(section.pos);

        if let item @ None = &mut self.fast_sections[idx] {
            *item = Some(section);
            return;
        }

        // this benchmarks better than pushing even when we linear search later. i guess
        // it has better cache locality?
        let index = self
            .fallback_sections
            .binary_search_by(|s| s.pos.cmp(&section.pos))
            .unwrap_or_else(|e| e);
        self.fallback_sections.insert(index, section);
    }
}
impl Default for CachedSections {
    fn default() -> Self {
        Self {
            fast_sections: (0..FAST_SECTIONS_CACHE_SIZE)
                .map(|_| None)
                .collect::<Box<[_]>>()
                .try_into()
                .unwrap(),
            fallback_sections: Default::default(),
        }
    }
}

pub struct CachedSection {
    pub pos: SmallChunkSectionPos,
    pub bitsets: Box<SectionBitsets>,
}
impl Debug for CachedSection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedSection")
            .field("pos", &self.pos)
            .finish()
    }
}

#[derive(Default)]
pub struct SectionBitsets {
    /// Blocks that we can fully pass through (like air).
    pub passable: FastFixedBitSet<4096>,
    /// Blocks that we can stand on and do parkour from.
    pub solid: FastFixedBitSet<4096>,
    /// Blocks that we can stand on but might not be able to parkour from.
    pub standable: FastFixedBitSet<4096>,
    /// Water source blocks.
    pub water: FastFixedBitSet<4096>,
}

impl CachedWorld {
    pub fn new(world_lock: Arc<RwLock<World>>, origin: BlockPos) -> Self {
        let min_y = world_lock.read().chunks.min_y();
        Self {
            origin,
            min_y,
            world_lock,
            bounded_chunk_cache: RefCell::new(array::from_fn(|_| {
                (ChunkPos::new(i32::MAX, i32::MAX), Default::default())
            })),
            unbounded_chunk_cache: Default::default(),
            cached_blocks: Default::default(),
            cached_mining_costs: UnsafeCell::new(None),
            block_source: None,
        }
    }

    /// Plan over the given source instead of the client's loaded world.
    ///
    /// Everything the source reports `None` for uses
    /// [`BlockSource::missing_block_state`].
    pub fn with_block_source(mut self, source: Arc<dyn BlockSource>) -> Self {
        self.block_source = Some(source);
        self
    }

    // ```
    // fn get_block_state(&self, pos: BlockPos) -> Option<BlockState> {
    //     self.with_section(ChunkSectionPos::from(pos), |section| {
    //         let state = section.get(pos.x as usize, pos.y as usize, pos.z as usize);
    //         BlockState::try_from(state).unwrap_or(BlockState::AIR)
    //     })
    // }
    // ```

    fn with_section<T>(
        &self,
        section_pos: SmallChunkSectionPos,
        f: impl FnOnce(&azalea_world::palette::PalettedContainer<BlockState>) -> T,
    ) -> Option<T> {
        if section_pos.y * 16 < self.min_y {
            // y position is out of bounds
            return None;
        }

        let chunk_pos = ChunkPos::new(section_pos.x as i32, section_pos.z as i32);
        let section_index =
            azalea_world::chunk::section_index(section_pos.y * 16, self.min_y) as usize;

        let mut cache_idx = 0;

        let mut unbounded_chunk_cache = self.unbounded_chunk_cache.borrow_mut();
        let mut bounded_chunk_cache = self.bounded_chunk_cache.borrow_mut();
        if unbounded_chunk_cache.is_empty() {
            const D: i32 = MAX_VIEW_DISTANCE as i32;
            let cache_x = i32::rem_euclid(chunk_pos.x, D) * D;
            let cache_z = i32::rem_euclid(chunk_pos.z, D);
            cache_idx = (cache_x + cache_z) as usize;

            // get section from cache
            if !bounded_chunk_cache[cache_idx].1.is_empty() {
                if bounded_chunk_cache[cache_idx].0 != chunk_pos {
                    // switch to the unbounded cache :(

                    for (moving_chunk_pos, moving_chunk) in bounded_chunk_cache.iter_mut() {
                        if !moving_chunk.is_empty() {
                            unbounded_chunk_cache
                                .insert(*moving_chunk_pos, mem::take(moving_chunk));
                        }
                    }
                }

                let sections = &bounded_chunk_cache[cache_idx].1;
                if section_index >= sections.len() {
                    // y position is out of bounds
                    return None;
                };
                let section = &sections[section_index];
                return Some(f(section));
            }
        } else if let Some(sections) = unbounded_chunk_cache.get(&chunk_pos) {
            if section_index >= sections.len() {
                // y position is out of bounds
                return None;
            };
            let section = &sections[section_index];
            return Some(f(section));
        }

        let world = self.world_lock.read();
        let chunk = world.chunks.get(&chunk_pos)?;
        let chunk = chunk.read();

        let sections = chunk
            .sections
            .iter()
            .map(|section| section.states.clone())
            .collect::<Box<[PalettedContainer<BlockState>]>>();

        if section_index >= sections.len() {
            // y position is out of bounds
            return None;
        };

        let section = &sections[section_index];
        let r = f(section);

        // add the sections to the chunk cache
        if unbounded_chunk_cache.is_empty() {
            bounded_chunk_cache[cache_idx] = (chunk_pos, sections);
        } else {
            unbounded_chunk_cache.insert(chunk_pos, sections);
        }

        Some(r)
    }

    fn calculate_bitsets_for_section(&self, section_pos: SmallChunkSectionPos) -> CachedSection {
        if let Some(source) = &self.block_source {
            return CachedSection {
                pos: section_pos,
                bitsets: bitsets_from_source(source.as_ref(), section_pos),
            };
        }
        let bitsets = self
            .with_section(section_pos, |section| {
                let mut bitsets = SectionBitsets {
                    passable: FastFixedBitSet::<4096>::new(),
                    solid: FastFixedBitSet::<4096>::new(),
                    standable: FastFixedBitSet::<4096>::new(),
                    water: FastFixedBitSet::<4096>::new(),
                };

                for i in 0..4096 {
                    let block_state = section.get_at_index(i);
                    if is_block_state_passable(block_state) {
                        bitsets.passable.set(i);
                    }
                    if is_block_state_solid(block_state) {
                        bitsets.solid.set(i);
                    }
                    if is_block_state_standable(block_state) {
                        bitsets.standable.set(i);
                    }
                    if is_block_state_water(block_state) {
                        bitsets.water.set(i);
                    }
                }
                Box::new(bitsets)
            })
            .unwrap_or_default();

        CachedSection {
            pos: section_pos,
            bitsets,
        }
    }

    fn check_bitset_for_block(
        &self,
        pos: BlockPos,
        cb: impl FnOnce(&SectionBitsets, usize) -> bool,
    ) -> bool {
        let (section_pos, section_block_pos) = (
            SmallChunkSectionPos::from(pos),
            ChunkSectionBlockPos::from(pos),
        );
        let index = u16::from(section_block_pos) as usize;
        // SAFETY: we're only accessing this from one thread
        let cached_blocks = unsafe { &mut *self.cached_blocks.get() };
        if let Some(cached) = cached_blocks.get_mut(section_pos) {
            return cb(&cached.bitsets, index);
        }

        let cached = self.calculate_bitsets_for_section(section_pos);
        let passable = cb(&cached.bitsets, index);
        cached_blocks.insert(cached);
        passable
    }

    pub fn is_block_passable(&self, pos: RelBlockPos) -> bool {
        self.is_block_pos_passable(pos.apply(self.origin))
    }
    fn is_block_pos_passable(&self, pos: BlockPos) -> bool {
        self.check_bitset_for_block(pos, |bitsets, index| bitsets.passable.index(index))
    }

    pub fn is_block_water(&self, pos: RelBlockPos) -> bool {
        self.is_block_pos_water(pos.apply(self.origin))
    }
    fn is_block_pos_water(&self, pos: BlockPos) -> bool {
        self.check_bitset_for_block(pos, |bitsets, index| bitsets.water.index(index))
    }

    /// Get the block state at the given position.
    ///
    /// This is relatively slow, so you should avoid it whenever possible.
    pub fn get_block_state(&self, pos: RelBlockPos) -> BlockState {
        self.get_block_state_at_pos(pos.apply(self.origin))
    }

    fn get_block_state_at_pos(&self, pos: BlockPos) -> BlockState {
        if let Some(source) = &self.block_source {
            return source
                .get_block_state(pos)
                .unwrap_or_else(|| source.missing_block_state());
        }
        let (section_pos, section_block_pos) = (
            SmallChunkSectionPos::from(pos),
            ChunkSectionBlockPos::from(pos),
        );
        let index = u16::from(section_block_pos) as usize;

        self.with_section(section_pos, |section| section.get_at_index(index))
            .unwrap_or_default()
    }

    pub fn is_block_solid(&self, pos: RelBlockPos) -> bool {
        self.is_block_pos_solid(pos.apply(self.origin))
    }
    pub fn is_block_standable(&self, pos: RelBlockPos) -> bool {
        self.is_block_pos_standable(pos.apply(self.origin))
    }

    fn is_block_pos_solid(&self, pos: BlockPos) -> bool {
        self.check_bitset_for_block(pos, |bitsets, index| bitsets.solid.index(index))
    }
    fn is_block_pos_standable(&self, pos: BlockPos) -> bool {
        self.check_bitset_for_block(pos, |bitsets, index| bitsets.standable.index(index))
    }

    /// Returns how much it costs to break this block.
    ///
    /// Returns 0 if the block is already passable.
    pub fn cost_for_breaking_block(&self, pos: RelBlockPos, mining_cache: &MiningCache) -> f32 {
        let cached_mining_costs = self.cached_mining_costs();

        let hash_index = calculate_cached_mining_costs_index(pos);
        let &(cached_pos, potential_cost) =
            unsafe { cached_mining_costs.get_unchecked(hash_index) };
        if cached_pos == pos {
            return potential_cost;
        }

        let cost = self.uncached_cost_for_breaking_block(pos, mining_cache);
        unsafe {
            *cached_mining_costs.get_unchecked_mut(hash_index) = (pos, cost);
        };

        cost
    }

    // this is fine because pathfinding is single-threaded
    #[allow(clippy::mut_from_ref)]
    fn cached_mining_costs(&self) -> &mut [(RelBlockPos, f32)] {
        // SAFETY: again, pathfinding is single-threaded
        let cached_mining_costs = unsafe { &mut *self.cached_mining_costs.get() };
        if let Some(cached_mining_costs) = cached_mining_costs {
            return cached_mining_costs;
        }
        // delay initialization so we don't have to create this if it's unused

        // this uses about 2mb of memory. it *really* helps though.
        *cached_mining_costs = Some(
            vec![(RelBlockPos::new(i16::MAX, i32::MAX, i16::MAX), 0.); CACHED_MINING_COSTS_SIZE]
                .into(),
        );

        cached_mining_costs.as_mut().unwrap()
    }

    fn uncached_cost_for_breaking_block(
        &self,
        pos: RelBlockPos,
        mining_cache: &MiningCache,
    ) -> f32 {
        if let Some(source) = &self.block_source {
            let block_state = match source.get_block_state(pos.apply(self.origin)) {
                Some(block_state) => block_state,
                None => {
                    // Preserve the source's declared geometry for unknown cells
                    // without ever consulting or mining the hidden loaded world.
                    return if is_block_state_passable(source.missing_block_state()) {
                        0.
                    } else {
                        f32::INFINITY
                    };
                }
            };
            if is_block_state_passable(block_state) {
                return 0.;
            }
            return self.uncached_cost_for_breaking_block_from_source(
                pos,
                mining_cache,
                source.as_ref(),
                block_state,
            );
        }

        if self.is_block_passable(pos) {
            // if the block is passable then it doesn't need to be broken
            return 0.;
        }

        let rel_pos = pos;
        let pos = pos.apply(self.origin);

        let (section_pos, section_block_pos) = (
            SmallChunkSectionPos::from(pos),
            ChunkSectionBlockPos::from(pos),
        );

        // we use this as an optimization to avoid getting the section again if the
        // block is in the same section
        let up_is_in_same_section = section_block_pos.y != 15;
        let north_is_in_same_section = section_block_pos.z != 0;
        let east_is_in_same_section = section_block_pos.x != 15;
        let south_is_in_same_section = section_block_pos.z != 15;
        let west_is_in_same_section = section_block_pos.x != 0;

        let mut is_falling_block_above = false;

        let Some(mut mining_cost) = self.with_section(section_pos, |section| {
            let block_state = section.get_at_index(u16::from(section_block_pos) as usize);
            let mining_cost = mining_cache.cost_for(block_state);

            if mining_cost == f32::INFINITY {
                // the block is unbreakable
                return f32::INFINITY;
            }

            // if there's a falling block or liquid above this block, abort
            if up_is_in_same_section {
                let up_block = section.get_at_index(u16::from(section_block_pos.up(1)) as usize);
                if mining_cache.is_liquid(up_block) {
                    return f32::INFINITY;
                }
                if mining_cache.is_falling_block(up_block) {
                    is_falling_block_above = true;
                }
            }

            // if there's a liquid to the north of this block, abort
            if north_is_in_same_section {
                let north_block =
                    section.get_at_index(u16::from(section_block_pos.north(1)) as usize);
                if mining_cache.is_liquid(north_block) {
                    return f32::INFINITY;
                }
            }

            // liquid to the east
            if east_is_in_same_section {
                let east_block =
                    section.get_at_index(u16::from(section_block_pos.east(1)) as usize);
                if mining_cache.is_liquid(east_block) {
                    return f32::INFINITY;
                }
            }

            // liquid to the south
            if south_is_in_same_section {
                let south_block =
                    section.get_at_index(u16::from(section_block_pos.south(1)) as usize);
                if mining_cache.is_liquid(south_block) {
                    return f32::INFINITY;
                }
            }

            // liquid to the west
            if west_is_in_same_section {
                let west_block =
                    section.get_at_index(u16::from(section_block_pos.west(1)) as usize);
                if mining_cache.is_liquid(west_block) {
                    return f32::INFINITY;
                }
            }

            // the block is probably safe to break, we'll have to check the adjacent blocks
            // that weren't in the same section next though
            mining_cost
        }) else {
            // the chunk isn't loaded
            let cost = if self.is_block_pos_solid(pos) {
                // assume it's unbreakable if it's solid and out of render distance
                f32::INFINITY
            } else {
                0.
            };
            return cost;
        };

        if mining_cost == f32::INFINITY {
            // the block is unbreakable
            return f32::INFINITY;
        }

        fn check_should_avoid_this_block(
            world: &CachedWorld,
            pos: BlockPos,
            check: impl FnOnce(BlockState) -> bool,
        ) -> bool {
            let block_state = world
                .with_section(SmallChunkSectionPos::from(pos), |section| {
                    section.get_at_index(u16::from(ChunkSectionBlockPos::from(pos)) as usize)
                })
                .unwrap_or_default();
            check(block_state)
        }

        // check the adjacent blocks that weren't in the same section
        if !up_is_in_same_section
            && check_should_avoid_this_block(self, pos.up(1), |b| {
                if mining_cache.is_falling_block(b) {
                    is_falling_block_above = true;
                }
                mining_cache.is_liquid(b)
            })
        {
            return f32::INFINITY;
        }
        if !north_is_in_same_section
            && check_should_avoid_this_block(self, pos.north(1), |b| mining_cache.is_liquid(b))
        {
            return f32::INFINITY;
        }
        if !east_is_in_same_section
            && check_should_avoid_this_block(self, pos.east(1), |b| mining_cache.is_liquid(b))
        {
            return f32::INFINITY;
        }
        if !south_is_in_same_section
            && check_should_avoid_this_block(self, pos.south(1), |b| mining_cache.is_liquid(b))
        {
            return f32::INFINITY;
        }
        if !west_is_in_same_section
            && check_should_avoid_this_block(self, pos.west(1), |b| mining_cache.is_liquid(b))
        {
            return f32::INFINITY;
        }

        if is_falling_block_above {
            mining_cost += self.cost_for_breaking_block(rel_pos.up(1), mining_cache);
        }

        mining_cost
    }

    /// Calculate mining cost without consulting the loaded world.
    ///
    /// In particular, a source's `None` for the target cannot fall through to
    /// a raw chunk that happens to contain that block. Neighbor safety checks
    /// use the same source and its configured missing-state semantics.
    fn uncached_cost_for_breaking_block_from_source(
        &self,
        rel_pos: RelBlockPos,
        mining_cache: &MiningCache,
        source: &dyn BlockSource,
        block_state: BlockState,
    ) -> f32 {
        let pos = rel_pos.apply(self.origin);
        let mut mining_cost = mining_cache.cost_for(block_state);
        if mining_cost == f32::INFINITY {
            return f32::INFINITY;
        }

        let state_or_missing = |pos| {
            source
                .get_block_state(pos)
                .unwrap_or_else(|| source.missing_block_state())
        };

        let up_block = state_or_missing(pos.up(1));
        if mining_cache.is_liquid(up_block) {
            return f32::INFINITY;
        }
        let is_falling_block_above = mining_cache.is_falling_block(up_block);

        for neighbor in [pos.north(1), pos.east(1), pos.south(1), pos.west(1)] {
            if mining_cache.is_liquid(state_or_missing(neighbor)) {
                return f32::INFINITY;
            }
        }

        if is_falling_block_above {
            mining_cost += self.cost_for_breaking_block(rel_pos.up(1), mining_cache);
        }

        mining_cost
    }

    /// Whether this block and the block above are passable
    pub fn is_passable(&self, pos: RelBlockPos) -> bool {
        self.is_passable_at_block_pos(pos.apply(self.origin))
    }
    fn is_passable_at_block_pos(&self, pos: BlockPos) -> bool {
        self.is_block_pos_passable(pos) && self.is_block_pos_passable(pos.up(1))
    }

    pub fn cost_for_passing(&self, pos: RelBlockPos, mining_cache: &MiningCache) -> f32 {
        self.cost_for_breaking_block(pos, mining_cache)
            + self.cost_for_breaking_block(pos.up(1), mining_cache)
    }

    /// Whether we can stand in this position.
    ///
    /// Checks if the block below is solid, and that the two blocks above that
    /// are passable.
    pub fn is_standable(&self, pos: RelBlockPos) -> bool {
        self.is_standable_at_block_pos(pos.apply(self.origin))
    }
    fn is_standable_at_block_pos(&self, pos: BlockPos) -> bool {
        self.is_block_pos_standable(pos.down(1)) && self.is_passable_at_block_pos(pos)
    }

    pub fn cost_for_standing(&self, pos: RelBlockPos, mining_cache: &MiningCache) -> f32 {
        if !self.is_block_standable(pos.down(1)) {
            return f32::INFINITY;
        }
        self.cost_for_passing(pos, mining_cache)
    }

    /// Get the amount of air/passable blocks until the next non-passable block
    /// below this one.
    pub fn fall_distance(&self, pos: RelBlockPos) -> u32 {
        let mut distance = 0;
        let mut current_pos = pos.down(1);
        while self.is_block_passable(current_pos) {
            distance += 1;
            current_pos = current_pos.down(1);

            if current_pos.y < self.min_y {
                return u32::MAX;
            }
        }
        distance
    }

    pub fn origin(&self) -> BlockPos {
        self.origin
    }
}

const CACHED_MINING_COSTS_SIZE: usize = 2usize.pow(18);
fn calculate_cached_mining_costs_index(pos: RelBlockPos) -> usize {
    // create an 18-bit index by taking the bottom bits from each axis

    const X_BITS: usize = 6;
    const Y_BITS: usize = 6;
    const Z_BITS: usize = 6;

    const X_MASK: usize = (1 << X_BITS) - 1;
    const Y_MASK: usize = (1 << Y_BITS) - 1;
    const Z_MASK: usize = (1 << Z_BITS) - 1;

    let hash_index = ((pos.x as usize & X_MASK) << (Y_BITS + Z_BITS))
        | ((pos.z as usize & Z_MASK) << Y_BITS)
        | (pos.y as usize & Y_MASK);
    debug_assert!(hash_index < CACHED_MINING_COSTS_SIZE);
    hash_index
}

/// Whether our client could pass through this block.
/// Build a section's bitsets by asking a [`BlockSource`] about each cell.
///
/// Cells the source does not know use [`BlockSource::missing_block_state`].
///
/// The index encoding matches `u16::from(ChunkSectionBlockPos)`:
/// `((y << 4) | z) << 4 | x`, with x/y/z each 0..16 inside the section.
fn bitsets_from_source(
    source: &dyn BlockSource,
    section_pos: SmallChunkSectionPos,
) -> Box<SectionBitsets> {
    let mut bitsets = SectionBitsets {
        passable: FastFixedBitSet::<4096>::new(),
        solid: FastFixedBitSet::<4096>::new(),
        standable: FastFixedBitSet::<4096>::new(),
        water: FastFixedBitSet::<4096>::new(),
    };

    let base_x = (section_pos.x as i32) << 4;
    let base_y = section_pos.y << 4;
    let base_z = (section_pos.z as i32) << 4;

    for i in 0..4096 {
        let pos = BlockPos::new(
            base_x + (i & 0xF) as i32,
            base_y + ((i >> 8) & 0xF) as i32,
            base_z + ((i >> 4) & 0xF) as i32,
        );
        let block_state = source
            .get_block_state(pos)
            .unwrap_or_else(|| source.missing_block_state());
        if is_block_state_passable(block_state) {
            bitsets.passable.set(i);
        }
        if is_block_state_solid(block_state) {
            bitsets.solid.set(i);
        }
        if is_block_state_standable(block_state) {
            bitsets.standable.set(i);
        }
        if is_block_state_water(block_state) {
            bitsets.water.set(i);
        }
    }
    Box::new(bitsets)
}

pub fn is_block_state_passable(block_state: BlockState) -> bool {
    // i already tried optimizing this by having it cache in an IntMap/FxHashMap but
    // it wasn't measurably faster

    if block_state.is_air() {
        // fast path
        return true;
    }
    if !block_state.is_collision_shape_empty() {
        return false;
    }
    let registry_block = BlockKind::from(block_state);
    if registry_block == BlockKind::Water {
        return false;
    }
    if block_state
        .property::<azalea_block::properties::Waterlogged>()
        .unwrap_or_default()
    {
        return false;
    }
    if registry_block == BlockKind::Lava {
        return false;
    }
    // block.waterlogged currently doesn't account for seagrass and some other water
    // blocks
    if block_state == BlockKind::Seagrass.into() {
        return false;
    }

    // don't walk into fire
    if registry_block == BlockKind::Fire || registry_block == BlockKind::SoulFire {
        return false;
    }

    if registry_block == BlockKind::PowderSnow {
        // we can't jump out of powder snow
        return false;
    }

    if registry_block == BlockKind::SweetBerryBush {
        // these hurt us
        return false;
    }

    true
}

/// Whether this block has a solid hitbox at the top (i.e. we can stand on it
/// and do parkour from it).
#[inline]
pub fn is_block_state_solid(block_state: BlockState) -> bool {
    if block_state.is_air() {
        // fast path
        return false;
    }

    if block_state.is_collision_shape_full() {
        // hazard
        if block_state == BlockState::from(BlockKind::MagmaBlock) {
            return false;
        };

        return true;
    }

    if matches!(
        block_state.property::<properties::SlabKind>(),
        Some(properties::SlabKind::Top | properties::SlabKind::Double)
    ) {
        // top slabs
        return true;
    }

    let block = BlockKind::from(block_state);
    // solid enough
    if matches!(block, BlockKind::DirtPath | BlockKind::Farmland) {
        return true;
    }

    false
}

/// Whether we can stand on this block (but not necessarily do parkour jumps
/// from it).
pub fn is_block_state_standable(block_state: BlockState) -> bool {
    if block_state.is_air() {
        // fast path
        return false;
    }

    if is_block_state_solid(block_state) {
        return true;
    }

    if block_state.property::<SlabKind>().is_some()
        || block_state.property::<StairShape>().is_some()
    {
        return true;
    }

    false
}

pub fn is_block_state_water(block_state: BlockState) -> bool {
    // only the default blockstate, which is source blocks
    block_state == BlockState::from(BlockKind::Water)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use azalea_inventory::{Menu, Player};
    use azalea_world::{Chunk, ChunkStorage, PartialWorld};

    use super::*;

    struct DefaultUnknown;

    impl BlockSource for DefaultUnknown {
        fn get_block_state(&self, _pos: BlockPos) -> Option<BlockState> {
            None
        }
    }

    struct StrictUnknown;

    impl BlockSource for StrictUnknown {
        fn get_block_state(&self, _pos: BlockPos) -> Option<BlockState> {
            None
        }

        fn missing_block_state(&self) -> BlockState {
            BlockKind::Fire.into()
        }
    }

    struct KnownBlocks {
        blocks: HashMap<BlockPos, BlockState>,
        missing: BlockState,
    }

    impl BlockSource for KnownBlocks {
        fn get_block_state(&self, pos: BlockPos) -> Option<BlockState> {
            self.blocks.get(&pos).copied()
        }

        fn missing_block_state(&self) -> BlockState {
            self.missing
        }
    }

    fn world_with_block(
        pos: BlockPos,
        block_state: BlockState,
    ) -> (Arc<RwLock<World>>, PartialWorld) {
        let mut partial_world = PartialWorld::default();
        let mut chunks = ChunkStorage::default();
        partial_world
            .chunks
            .set(&ChunkPos::from(pos), Some(Chunk::default()), &mut chunks);
        partial_world
            .chunks
            .set_block_state(pos, block_state, &chunks);
        (Arc::new(RwLock::new(chunks.into())), partial_world)
    }

    fn mining_cache_with_empty_inventory() -> MiningCache {
        MiningCache::new(Some(Menu::Player(Player::default())))
    }

    #[test]
    fn a_block_source_controls_how_unknown_cells_enter_the_graph() {
        let section = SmallChunkSectionPos::from(BlockPos::new(0, 0, 0));
        let index = u16::from(ChunkSectionBlockPos::from(BlockPos::new(0, 0, 0))) as usize;

        let optimistic = bitsets_from_source(&DefaultUnknown, section);
        assert!(optimistic.passable.index(index));
        assert!(!optimistic.standable.index(index));

        let strict = bitsets_from_source(&StrictUnknown, section);
        assert!(!strict.passable.index(index));
        assert!(!strict.standable.index(index));

        let direct = CachedWorld::new(
            Arc::new(RwLock::new(ChunkStorage::default().into())),
            BlockPos::default(),
        )
        .with_block_source(Arc::new(StrictUnknown));
        assert_eq!(
            direct.get_block_state_at_pos(BlockPos::default()),
            BlockState::from(BlockKind::Fire),
            "direct state reads and section bitsets must share the fallback"
        );
    }

    #[test]
    fn a_block_source_controls_mining_targets_without_loaded_world_fallback() {
        let target = BlockPos::new(0, 1, 0);
        let rel_target = RelBlockPos::from_origin(BlockPos::default(), target);
        let mining_cache = mining_cache_with_empty_inventory();

        let (stone_world, _stone_world_owner) = world_with_block(target, BlockKind::Stone.into());
        let loaded_world = CachedWorld::new(stone_world.clone(), BlockPos::default());
        assert!(
            loaded_world
                .cost_for_breaking_block(rel_target, &mining_cache)
                .is_finite(),
            "the raw stone fixture must be mineable"
        );

        let strict_unknown = CachedWorld::new(stone_world, BlockPos::default())
            .with_block_source(Arc::new(StrictUnknown));
        assert_eq!(
            strict_unknown.cost_for_breaking_block(rel_target, &mining_cache),
            f32::INFINITY,
            "an unknown target must not fall through to a loaded raw chunk"
        );

        let (optimistic_world, _optimistic_world_owner) =
            world_with_block(target, BlockKind::Stone.into());
        let optimistic_unknown = CachedWorld::new(optimistic_world, BlockPos::default())
            .with_block_source(Arc::new(DefaultUnknown));
        assert_eq!(
            optimistic_unknown.cost_for_breaking_block(rel_target, &mining_cache),
            0.,
            "an optimistic source may expose unknown geometry as air without mining it"
        );

        let (bedrock_world, _bedrock_world_owner) =
            world_with_block(target, BlockKind::Bedrock.into());
        let loaded_world = CachedWorld::new(bedrock_world.clone(), BlockPos::default());
        assert_eq!(
            loaded_world.cost_for_breaking_block(rel_target, &mining_cache),
            f32::INFINITY,
            "the raw bedrock fixture must be unmineable"
        );

        let source = KnownBlocks {
            blocks: HashMap::from([(target, BlockKind::Stone.into())]),
            missing: BlockKind::Fire.into(),
        };
        let source_world = CachedWorld::new(bedrock_world, BlockPos::default())
            .with_block_source(Arc::new(source));
        assert!(
            source_world
                .cost_for_breaking_block(rel_target, &mining_cache)
                .is_finite(),
            "a known source block must take precedence over hidden raw bedrock"
        );
    }

    #[test]
    fn mining_safety_neighbors_are_read_from_the_block_source() {
        let target = BlockPos::new(0, 1, 0);
        let rel_target = RelBlockPos::from_origin(BlockPos::default(), target);
        let mining_cache = mining_cache_with_empty_inventory();
        let (raw_world, _raw_world_owner) = world_with_block(target, BlockKind::Stone.into());

        let loaded_world = CachedWorld::new(raw_world.clone(), BlockPos::default());
        assert!(
            loaded_world
                .cost_for_breaking_block(rel_target, &mining_cache)
                .is_finite(),
            "the raw fixture has no liquid around the target"
        );

        let source = KnownBlocks {
            blocks: HashMap::from([
                (target, BlockKind::Stone.into()),
                (target.up(1), BlockKind::Water.into()),
            ]),
            missing: BlockKind::Fire.into(),
        };
        let source_world =
            CachedWorld::new(raw_world, BlockPos::default()).with_block_source(Arc::new(source));
        assert_eq!(
            source_world.cost_for_breaking_block(rel_target, &mining_cache),
            f32::INFINITY,
            "liquid visible only through the source must still prevent mining"
        );
    }

    #[test]
    fn test_is_passable() {
        let mut partial_world = PartialWorld::default();
        let mut world = ChunkStorage::default();

        partial_world
            .chunks
            .set(&ChunkPos { x: 0, z: 0 }, Some(Chunk::default()), &mut world);
        partial_world.chunks.set_block_state(
            BlockPos::new(0, 0, 0),
            BlockKind::Stone.into(),
            &world,
        );
        partial_world
            .chunks
            .set_block_state(BlockPos::new(0, 1, 0), BlockState::AIR, &world);

        let ctx = CachedWorld::new(Arc::new(RwLock::new(world.into())), BlockPos::default());
        assert!(!ctx.is_block_pos_passable(BlockPos::new(0, 0, 0)));
        assert!(ctx.is_block_pos_passable(BlockPos::new(0, 1, 0)));
    }

    #[test]
    fn test_is_solid() {
        let mut partial_world = PartialWorld::default();
        let mut world = ChunkStorage::default();
        partial_world
            .chunks
            .set(&ChunkPos { x: 0, z: 0 }, Some(Chunk::default()), &mut world);
        partial_world.chunks.set_block_state(
            BlockPos::new(0, 0, 0),
            BlockKind::Stone.into(),
            &world,
        );
        partial_world
            .chunks
            .set_block_state(BlockPos::new(0, 1, 0), BlockState::AIR, &world);

        let ctx = CachedWorld::new(Arc::new(RwLock::new(world.into())), BlockPos::default());
        assert!(ctx.is_block_pos_solid(BlockPos::new(0, 0, 0)));
        assert!(!ctx.is_block_pos_solid(BlockPos::new(0, 1, 0)));
    }

    #[test]
    fn test_is_standable() {
        let mut partial_world = PartialWorld::default();
        let mut world = ChunkStorage::default();
        partial_world
            .chunks
            .set(&ChunkPos { x: 0, z: 0 }, Some(Chunk::default()), &mut world);
        partial_world.chunks.set_block_state(
            BlockPos::new(0, 0, 0),
            BlockKind::Stone.into(),
            &world,
        );
        partial_world
            .chunks
            .set_block_state(BlockPos::new(0, 1, 0), BlockState::AIR, &world);
        partial_world
            .chunks
            .set_block_state(BlockPos::new(0, 2, 0), BlockState::AIR, &world);
        partial_world
            .chunks
            .set_block_state(BlockPos::new(0, 3, 0), BlockState::AIR, &world);

        let ctx = CachedWorld::new(Arc::new(RwLock::new(world.into())), BlockPos::default());
        assert!(ctx.is_standable_at_block_pos(BlockPos::new(0, 1, 0)));
        assert!(!ctx.is_standable_at_block_pos(BlockPos::new(0, 0, 0)));
        assert!(!ctx.is_standable_at_block_pos(BlockPos::new(0, 2, 0)));
    }
}
