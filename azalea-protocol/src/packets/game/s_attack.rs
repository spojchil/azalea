use azalea_buf::AzBuf;
use azalea_core::entity_id::MinecraftEntityId;
use azalea_protocol_macros::ServerboundGamePacket;

#[derive(Clone, Debug, AzBuf, PartialEq, ServerboundGamePacket)]
pub struct ServerboundAttack {
    /// Vanilla `ServerboundAttackPacket` encodes this with `ByteBufCodecs.VAR_INT`.
    #[var]
    pub entity_id: MinecraftEntityId,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_id_is_a_varint() {
        let mut bytes = Vec::new();
        ServerboundAttack {
            entity_id: MinecraftEntityId(300),
        }
        .azalea_write(&mut bytes)
        .unwrap();
        assert_eq!(bytes, vec![0xAC, 0x02]);
    }
}
