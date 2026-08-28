//! 规划期对「这一格能不能放、放什么、多贵」的回答。
//!
//! 挖和放是同一件事——替换（见 [`super::policy::ReplaceTo`]）。挖那一半的代价由
//! [`MiningCache`] 算，这里算放那一半。
//!
//! **放什么由策略决定，不由这里决定。** 这里只负责把「手上有什么」和策略的答案
//! 对上：拿快捷栏里的每一格去问策略，谁过谁上。
//!
//! [`MiningCache`]: super::mining::MiningCache

use std::str::FromStr;

use azalea_inventory::Menu;
use azalea_registry::builtin::BlockKind;

use super::{costs::BLOCK_PLACEMENT_PENALTY, positions::RelBlockPos, world::CachedWorld};

/// 快捷栏里能放的东西：格位号和它是什么方块。
///
/// 物品未必对应方块（镐子就不是），对不上的直接不进候选。
pub fn placeable_candidates(menu: &Menu) -> Vec<(usize, BlockKind)> {
    let hotbar = &menu.slots()[menu.hotbar_slots_range()];
    hotbar
        .iter()
        .enumerate()
        .filter_map(|(index, slot)| {
            if !slot.is_present() {
                return None;
            }
            // 物品名和方块名同口径（都是注册表 identifier），对不上的不进候选：
            // 镐子没有对应的方块，放不了。
            BlockKind::from_str(slot.kind().to_str())
                .ok()
                .map(|kind| (index, kind))
        })
        .collect()
}

/// 一次规划里「手上有什么可以放」的冻结答案。
///
/// 和 [`MiningCache`] 一样按次构造：同一次 A*
/// 从头到尾读同一份库存，不会算到一半 因为背包变了而前后矛盾。
///
/// [`MiningCache`]: super::mining::MiningCache
pub struct PlacementCache {
    candidates: Vec<(usize, BlockKind)>,
}

impl PlacementCache {
    /// `inventory_menu` 为 `None` 表示这次规划拿不到库存（也就放不了）。
    pub fn new(inventory_menu: Option<Menu>) -> Self {
        Self {
            candidates: inventory_menu
                .as_ref()
                .map(|menu| placeable_candidates(menu))
                .unwrap_or_default(),
        }
    }

    /// 在 `pos` 放一格：策略放行且手上有的第一种方块，连同它在快捷栏的格位。
    ///
    /// 「准不准放」和「放什么」是同一个问题，所以一次问完——策略可能只允许放圆石
    /// 不允许放沙子，那答案就得带着是哪一种。
    pub fn choose_placement(
        &self,
        pos: RelBlockPos,
        world: &CachedWorld,
    ) -> Option<(usize, BlockKind)> {
        if self.candidates.is_empty() {
            return None;
        }
        // 只往腾得出来的地方放。已经有东西的格子要先挖——那是另一条替换，得由
        // movement 单独声明，不能在这里悄悄替它决定。
        if !world.is_block_passable(pos) {
            return None;
        }
        let absolute = world.absolute(pos);
        let from = world.get_block_state(pos);
        self.candidates
            .iter()
            .copied()
            .find(|(_, kind)| world.replace_policy().allows_place(absolute, from, *kind))
    }

    /// 在 `pos` 放一格要多少代价。放不了就是 `INFINITY`。
    pub fn cost_for_placing(&self, pos: RelBlockPos, world: &CachedWorld) -> f32 {
        if self.choose_placement(pos, world).is_some() {
            BLOCK_PLACEMENT_PENALTY
        } else {
            f32::INFINITY
        }
    }
}

#[cfg(test)]
mod tests {
    use azalea_inventory::{ItemStack, ItemStackData, Player};
    use azalea_registry::builtin::ItemKind;

    use super::*;

    fn hotbar_with(kinds: [ItemKind; 9]) -> Menu {
        let mut menu = Menu::Player(Player::default());
        for (index, kind) in menu.hotbar_slots_range().zip(kinds) {
            let slot = menu.slot_mut(index).expect("快捷栏格位存在");
            *slot = if kind == ItemKind::Air {
                ItemStack::Empty
            } else {
                ItemStack::Present(ItemStackData::new(kind, 64))
            };
        }
        menu
    }

    const NOTHING: ItemKind = ItemKind::Air;

    /// 放不了的东西不进候选：镐子没有对应的方块。
    #[test]
    fn only_items_that_are_blocks_become_candidates() {
        let menu = hotbar_with([
            ItemKind::DiamondPickaxe,
            ItemKind::Cobblestone,
            ItemKind::Stick,
            ItemKind::Dirt,
            NOTHING,
            NOTHING,
            NOTHING,
            NOTHING,
            NOTHING,
        ]);
        assert_eq!(
            placeable_candidates(&menu),
            [(1, BlockKind::Cobblestone), (3, BlockKind::Dirt)]
        );
    }

    /// 空手就一个候选都没有。
    #[test]
    fn an_empty_hotbar_has_no_candidates() {
        assert!(placeable_candidates(&hotbar_with([NOTHING; 9])).is_empty());
    }

    /// 拿不到库存等于放不了——权限先于库存。
    #[test]
    fn a_plan_without_an_inventory_has_no_candidates() {
        assert!(PlacementCache::new(None).candidates.is_empty());
    }
}
