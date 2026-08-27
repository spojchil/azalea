//! 规划期对「这一格能不能放、放一格多贵」的回答。
//!
//! 挖和放是同一件事——替换（见 [`super::moves::ReplaceTarget`]）。挖那一半由
//! [`MiningCache`] 回答，这里回答放那一半。
//!
//! 目前「拿什么放」是一张固定的候选清单，抄自 Baritone 的
//! `acceptableThrowawayItems`（dirt / cobblestone / netherrack / stone）。它是
//! 占位：真正该决定放什么的是模型的策略，清单落地后这里换成按策略求值，调用点
//! 不用改。
//!
//! [`MiningCache`]: super::mining::MiningCache

use azalea_inventory::Menu;
use azalea_registry::builtin::ItemKind;

use super::{costs::BLOCK_PLACEMENT_PENALTY, positions::RelBlockPos, world::CachedWorld};

/// 可以拿来垫脚/搭桥的方块。抄 Baritone `acceptableThrowawayItems` 的默认值。
const THROWAWAY_ITEMS: [ItemKind; 4] = [
    ItemKind::Dirt,
    ItemKind::Cobblestone,
    ItemKind::Netherrack,
    ItemKind::Stone,
];

/// 快捷栏里第一格可以拿来放的方块。没有就是 `None`——一格也放不了。
pub fn placeable_in_hotbar(menu: &Menu) -> Option<usize> {
    let hotbar = &menu.slots()[menu.hotbar_slots_range()];
    hotbar
        .iter()
        .position(|slot| slot.is_present() && THROWAWAY_ITEMS.contains(&slot.kind()))
}

/// 一次规划里「能不能放」的冻结答案。
///
/// 和 [`MiningCache`] 一样按次构造：同一次 A* 从头到尾读同一个答案，不会算到
/// 一半因为背包变了而前后矛盾。
///
/// [`MiningCache`]: super::mining::MiningCache
pub struct PlacementCache {
    /// 规划开始时快捷栏里可放方块所在的格。`None` = 这次不允许放，或者没料。
    hotbar_slot: Option<usize>,
}

impl PlacementCache {
    /// `inventory_menu` 为 `None` 表示这次规划不允许放置。
    pub fn new(inventory_menu: Option<Menu>) -> Self {
        Self {
            hotbar_slot: inventory_menu.as_ref().and_then(placeable_in_hotbar),
        }
    }

    /// 这次规划允许放置吗（允许且有料）。
    pub fn is_allowed(&self) -> bool {
        self.hotbar_slot.is_some()
    }

    /// 在 `pos` 放一格要多少代价。放不了就是 `INFINITY`。
    ///
    /// 只判断「这一格现在腾得出来吗」——支撑面由各个 movement 自己按几何判断，
    /// 因为哪一面可以贴取决于往哪个方向放。
    pub fn cost_for_placing(&self, pos: RelBlockPos, world: &CachedWorld) -> f32 {
        if !self.is_allowed() {
            return f32::INFINITY;
        }
        // 只往空的地方放。已经有东西的格子要先挖——那是另一条替换，得由 movement
        // 单独声明，不能在放置这里悄悄替它决定。
        if !world.is_block_passable(pos) {
            return f32::INFINITY;
        }
        // 流体里不放。Baritone 把这拆成 source/flow 两个开关，我们没有那两个开关
        // 就一律不放，宁可少一条边也不要把方块丢进水里。
        if world.is_block_water(pos) {
            return f32::INFINITY;
        }
        BLOCK_PLACEMENT_PENALTY
    }
}

#[cfg(test)]
mod tests {
    use azalea_inventory::{ItemStack, ItemStackData};

    use super::*;

    fn hotbar_with(kinds: [ItemKind; 9]) -> Menu {
        let mut menu = Menu::Player(azalea_inventory::Player::default());
        let range = menu.hotbar_slots_range();
        for (index, kind) in range.zip(kinds) {
            let slot = menu.slot_mut(index).expect("快捷栏格位存在");
            *slot = if kind == ItemKind::Air {
                ItemStack::Empty
            } else {
                ItemStack::Present(ItemStackData::new(kind, 64))
            };
        }
        menu
    }

    #[test]
    fn a_hotbar_without_throwaway_blocks_cannot_place() {
        let menu = hotbar_with([
            ItemKind::DiamondPickaxe,
            ItemKind::Air,
            ItemKind::Air,
            ItemKind::Air,
            ItemKind::Air,
            ItemKind::Air,
            ItemKind::Air,
            ItemKind::Air,
            ItemKind::Air,
        ]);
        assert!(placeable_in_hotbar(&menu).is_none());
        assert!(!PlacementCache::new(Some(menu)).is_allowed());
    }

    #[test]
    fn the_first_throwaway_block_in_the_hotbar_wins() {
        let menu = hotbar_with([
            ItemKind::DiamondPickaxe,
            ItemKind::Cobblestone,
            ItemKind::Dirt,
            ItemKind::Air,
            ItemKind::Air,
            ItemKind::Air,
            ItemKind::Air,
            ItemKind::Air,
            ItemKind::Air,
        ]);
        assert_eq!(placeable_in_hotbar(&menu), Some(1));
        assert!(PlacementCache::new(Some(menu)).is_allowed());
    }

    /// 不允许放置时，有料也答不允许——权限先于库存。
    #[test]
    fn a_plan_that_forbids_placing_never_reports_it_can_place() {
        assert!(!PlacementCache::new(None).is_allowed());
    }
}
