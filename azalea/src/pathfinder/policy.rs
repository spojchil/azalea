//! 寻路器可以**自作主张**改动世界到什么程度。
//!
//! 挖和放在这里是同一个问题，不是两个：挖是把一格换成空气，放是把一格换成方块
//! （见 [`MovementSideEffects`]）。所以这个 trait 只有一个方法。
//!
//! 装法和 [`BlockSource`] 一样：trait
//! 在这里，实现在调用方。规则怎么写、怎么组合、
//! 谁改的，寻路器一概不知道——它只问「这一格从 X 换成 Y，行吗」。
//!
//! 没装就是原样：由 [`PathfinderOpts::allow_mining`] / [`allow_placing`] 那两个
//! 开关全权决定。装了之后两者都要点头才动手——策略放行不能越过调用方的总开关。
//!
//! [`MovementSideEffects`]: super::moves::MovementSideEffects
//! [`BlockSource`]: super::world::BlockSource
//! [`PathfinderOpts::allow_mining`]: super::PathfinderOpts::allow_mining
//! [`allow_placing`]: super::PathfinderOpts::allow_placing

use std::sync::Arc;

use azalea_block::BlockState;
use azalea_core::position::BlockPos;
use azalea_registry::builtin::BlockKind;
use bevy_ecs::component::Component;

/// 替换之后那一格应该是什么。
///
/// 挖的目标恒为 [`Self::Air`]，没得选；放的目标是从手上有的东西里挑，所以带着
/// 具体是哪一种——「准不准放」和「放什么」是同一个问题，不该拆成两次判断。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplaceTo {
    Air,
    Block(BlockKind),
}

/// 寻路器自作主张的世界改动要过的那一关。
///
/// 只管**寻路器自己决定**的替换。调用方点名要挖/要放的坐标不走这里——点名本身
/// 就是授权。
pub trait ReplacePolicy: Send + Sync {
    /// `pos` 这一格从 `from` 换成 `to`，允许吗。
    ///
    /// 会在 A\* 热路径上按格调用，实现应当便宜且**无副作用**：
    /// 同一次规划里同一个 问题必须给同一个答案，
    /// 否则搜索出来的路对哪一版都不成立。
    fn allows_replace(&self, pos: BlockPos, from: BlockState, to: ReplaceTo) -> bool;
}

/// 装一份 [`ReplacePolicy`] 到实体上，规划、补路与执行都读它。
#[derive(Component, Clone)]
pub struct PathfinderReplacePolicy(pub Arc<dyn ReplacePolicy>);

/// 一次规划/一条路线冻结的那一份。
///
/// `None` = 没装，只受总开关约束。
///
/// **一次规划取一次，取到就不再变**：路线执行到一半模型改了规则，这条路线继续
/// 按当时那一份走完，补路也复用同一份。否则补出来的一段是按新规则批的，却要接进
/// 按旧规则批的路线里。下一条腿自然会拿到新的。
#[derive(Clone, Default)]
pub struct PolicySnapshot {
    policy: Option<Arc<dyn ReplacePolicy>>,
    /// 调用方的总开关。策略和它都点头才动手；两者谁都不能越过谁。
    allow_break: bool,
    allow_place: bool,
}

impl PolicySnapshot {
    pub fn new(
        policy: Option<Arc<dyn ReplacePolicy>>,
        allow_break: bool,
        allow_place: bool,
    ) -> Self {
        Self {
            policy,
            allow_break,
            allow_place,
        }
    }

    /// 不受任何约束：没装策略，两个开关都开。默认值（`Default`）反之，
    /// 两个开关都关。
    pub fn open() -> Self {
        Self::new(None, true, true)
    }

    /// 调用方的总开关本身。用于「要不要去取库存」这类**与坐标无关**的判断；
    /// 「这一格准不准动」永远问 [`Self::allows_break`] /
    /// [`Self::allows_place`]。
    pub fn may_break(&self) -> bool {
        self.allow_break
    }

    pub fn may_place(&self) -> bool {
        self.allow_place
    }

    /// 换成空气（挖）允许吗。
    pub fn allows_break(&self, pos: BlockPos, from: BlockState) -> bool {
        self.allow_break && self.allows(pos, from, ReplaceTo::Air)
    }

    /// 换成 `block`（放）允许吗。
    pub fn allows_place(&self, pos: BlockPos, from: BlockState, block: BlockKind) -> bool {
        self.allow_place && self.allows(pos, from, ReplaceTo::Block(block))
    }

    fn allows(&self, pos: BlockPos, from: BlockState, to: ReplaceTo) -> bool {
        match &self.policy {
            // 没装策略就是原样：总开关说了算。
            None => true,
            Some(policy) => policy.allows_replace(pos, from, to),
        }
    }
}

impl std::fmt::Debug for PolicySnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolicySnapshot")
            .field("installed", &self.policy.is_some())
            .field("allow_break", &self.allow_break)
            .field("allow_place", &self.allow_place)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DenyAll;
    impl ReplacePolicy for DenyAll {
        fn allows_replace(&self, _: BlockPos, _: BlockState, _: ReplaceTo) -> bool {
            false
        }
    }

    struct OnlyAir;
    impl ReplacePolicy for OnlyAir {
        fn allows_replace(&self, _: BlockPos, _: BlockState, to: ReplaceTo) -> bool {
            to == ReplaceTo::Air
        }
    }

    const POS: BlockPos = BlockPos::new(3, 64, -7);

    fn open(policy: Option<Arc<dyn ReplacePolicy>>) -> PolicySnapshot {
        PolicySnapshot::new(policy, true, true)
    }

    #[test]
    fn without_a_policy_the_caller_switch_decides_alone() {
        assert!(open(None).allows_break(POS, BlockState::AIR));
        assert!(open(None).allows_place(POS, BlockState::AIR, BlockKind::Cobblestone));

        let shut = PolicySnapshot::new(None, false, false);
        assert!(!shut.allows_break(POS, BlockState::AIR));
        assert!(!shut.allows_place(POS, BlockState::AIR, BlockKind::Cobblestone));
    }

    /// 策略放行也不能越过总开关——两者都要点头。
    #[test]
    fn a_permissive_policy_cannot_override_the_caller_switch() {
        assert!(open(Some(Arc::new(OnlyAir))).allows_break(POS, BlockState::AIR));
        assert!(
            !PolicySnapshot::new(Some(Arc::new(OnlyAir)), false, true)
                .allows_break(POS, BlockState::AIR),
            "调用方关掉挖掘时，策略说行也不行"
        );
    }

    /// 总开关开着也不能越过策略。
    #[test]
    fn an_open_caller_switch_cannot_override_the_policy() {
        let snapshot = open(Some(Arc::new(DenyAll)));
        assert!(!snapshot.allows_break(POS, BlockState::AIR));
        assert!(!snapshot.allows_place(POS, BlockState::AIR, BlockKind::Cobblestone));
    }

    /// 挖和放是同一个问题的两个取值，策略可以只放行其中一边。
    #[test]
    fn the_same_question_can_be_answered_differently_per_target() {
        let snapshot = open(Some(Arc::new(OnlyAir)));
        assert!(snapshot.allows_break(POS, BlockKind::Stone.into()));
        assert!(!snapshot.allows_place(POS, BlockState::AIR, BlockKind::Cobblestone));
    }
}
