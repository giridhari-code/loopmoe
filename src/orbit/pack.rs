//! PackedBatch — Orbit O2 batch descriptor for one engine step.
//!
//! Holds everything needed to run decode (and later chunked prefill) without
//! re-querying the scheduler mid-forward:
//! - sequence ids + roles (decode vs prefill)
//! - input token ids + positions
//! - optional BlockLedger slot indices

use super::ledger::SlotMapping;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackRole {
    Decode,
    Prefill,
}

#[derive(Debug, Clone)]
pub struct PackItem {
    pub seq_id: u64,
    pub role: PackRole,
    /// Tokens to feed this step (decode: len=1; prefill: chunk).
    pub token_ids: Vec<u32>,
    /// Cache start position for the first token in `token_ids`.
    pub position: usize,
}

#[derive(Debug, Clone, Default)]
pub struct PackedBatch {
    pub items: Vec<PackItem>,
    /// Flattened slot mapping aligned with total tokens (optional O1 link).
    pub slots: SlotMapping,
    pub num_decode: usize,
    pub num_prefill: usize,
    pub num_tokens: usize,
}

impl PackedBatch {
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn decode_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.items
            .iter()
            .filter(|i| i.role == PackRole::Decode)
            .map(|i| i.seq_id)
    }

    pub fn prefill_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.items
            .iter()
            .filter(|i| i.role == PackRole::Prefill)
            .map(|i| i.seq_id)
    }

    /// Build from scheduler outputs + per-seq token/position lookups.
    pub fn from_schedule(
        prefill: &[(u64, Vec<u32>, usize)], // id, prompt_tokens, position(=0)
        decode: &[(u64, u32, usize)],       // id, last_token, position
        slots: SlotMapping,
    ) -> Self {
        let mut items = Vec::with_capacity(prefill.len() + decode.len());
        let mut num_tokens = 0usize;
        for (id, toks, pos) in prefill {
            num_tokens += toks.len();
            items.push(PackItem {
                seq_id: *id,
                role: PackRole::Prefill,
                token_ids: toks.clone(),
                position: *pos,
            });
        }
        for (id, tok, pos) in decode {
            num_tokens += 1;
            items.push(PackItem {
                seq_id: *id,
                role: PackRole::Decode,
                token_ids: vec![*tok],
                position: *pos,
            });
        }
        Self {
            num_prefill: prefill.len(),
            num_decode: decode.len(),
            num_tokens,
            items,
            slots,
        }
    }
}
