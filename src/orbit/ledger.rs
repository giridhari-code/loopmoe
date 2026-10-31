//! BlockLedger — physical paged memory for GQA KV or MLA latent cache.
//!
//! Design goals (Orbit O1):
//! - Fixed-size **pages** (tokens per page = `block_size`)
//! - Global free list with ref-counts (share / prefix later)
//! - Per-sequence **block table** (logical token order → physical page ids)
//! - **Slot mapping** for a scheduled step: each token position → physical slot
//!   index into the flattened page pool (attention kernel reads here)
//!
//! Physical storage is still owned by per-seq `RuntimeCaches` today; Ledger
//! tracks *addressing*. O2 binds Ledger slots into a shared pool tensor.

use std::collections::HashMap;

/// Which attention cache bank this ledger serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheKind {
    Gqa,
    Mla,
}

#[derive(Debug, Clone)]
pub struct LedgerConfig {
    pub num_pages: usize,
    pub block_size: usize,
    pub kind: CacheKind,
    /// For capacity estimates / future pool alloc.
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub kv_lora_rank: usize,
    pub rope_dim: usize,
}

impl Default for LedgerConfig {
    fn default() -> Self {
        Self {
            num_pages: 1024,
            block_size: 16,
            kind: CacheKind::Gqa,
            num_layers: 32,
            num_kv_heads: 8,
            head_dim: 128,
            kv_lora_rank: 512,
            rope_dim: 64,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct LedgerStats {
    pub pages_allocated: u64,
    pub pages_freed: u64,
    pub peak_used: usize,
    pub alloc_failures: u64,
}

#[derive(Debug, Clone)]
struct PageMeta {
    ref_count: i32,
    /// Optional content hash for future prefix sharing (0 = none).
    content_hash: u64,
}

/// Per-sequence logical → physical page ids (1-based; 0 unused).
#[derive(Debug, Clone, Default)]
pub struct SeqBlockTable {
    pub pages: Vec<u32>,
    pub num_tokens: usize,
}

impl SeqBlockTable {
    pub fn num_pages(&self) -> usize {
        self.pages.len()
    }

    pub fn clear(&mut self) {
        self.pages.clear();
        self.num_tokens = 0;
    }
}

/// Maps each position in a *scheduled step* to a physical slot.
///
/// `physical_slots[i]` = page_id * block_size + offset_in_page
/// for the i-th token in the packed batch (decode or prefill chunk).
#[derive(Debug, Clone, Default)]
pub struct SlotMapping {
    pub physical_slots: Vec<u32>,
    pub seq_ids: Vec<u64>,
    /// Cumulative token counts per seq in this mapping (CSR-style).
    pub cu_seqlens: Vec<u32>,
}

impl SlotMapping {
    pub fn num_tokens(&self) -> usize {
        self.physical_slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.physical_slots.is_empty()
    }
}

#[derive(Debug)]
pub struct BlockLedger {
    cfg: LedgerConfig,
    pages: Vec<PageMeta>,
    /// Free list as stack of 1-based page ids.
    free: Vec<u32>,
    tables: HashMap<u64, SeqBlockTable>,
    pub stats: LedgerStats,
}

impl BlockLedger {
    pub fn new(cfg: LedgerConfig) -> Self {
        let n = cfg.num_pages.max(1);
        let mut pages = Vec::with_capacity(n + 1);
        pages.push(PageMeta {
            ref_count: 0,
            content_hash: 0,
        }); // sentinel index 0
        for _ in 0..n {
            pages.push(PageMeta {
                ref_count: 0,
                content_hash: 0,
            });
        }
        let free: Vec<u32> = (1..=n as u32).rev().collect();
        Self {
            cfg,
            pages,
            free,
            tables: HashMap::new(),
            stats: LedgerStats::default(),
        }
    }

    pub fn config(&self) -> &LedgerConfig {
        &self.cfg
    }

    pub fn block_size(&self) -> usize {
        self.cfg.block_size
    }

    pub fn num_free(&self) -> usize {
        self.free.len()
    }

    pub fn num_used(&self) -> usize {
        self.cfg.num_pages.saturating_sub(self.free.len())
    }

    pub fn pages_for_tokens(&self, num_tokens: usize) -> usize {
        if self.cfg.block_size == 0 {
            return 0;
        }
        num_tokens.div_ceil(self.cfg.block_size)
    }

    /// Register a new sequence with empty table.
    pub fn register_seq(&mut self, seq_id: u64) {
        self.tables.entry(seq_id).or_default();
    }

    pub fn table(&self, seq_id: u64) -> Option<&SeqBlockTable> {
        self.tables.get(&seq_id)
    }

    pub fn table_mut(&mut self, seq_id: u64) -> Option<&mut SeqBlockTable> {
        self.tables.get_mut(&seq_id)
    }

    /// Ensure `seq_id` has pages for at least `num_tokens`. Allocates if needed.
    /// Returns false if not enough free pages.
    pub fn ensure_tokens(&mut self, seq_id: u64, num_tokens: usize) -> bool {
        self.tables.entry(seq_id).or_default();
        let need_pages = self.pages_for_tokens(num_tokens);
        let have = self.tables.get(&seq_id).map(|t| t.pages.len()).unwrap_or(0);
        if need_pages <= have {
            if let Some(t) = self.tables.get_mut(&seq_id) {
                t.num_tokens = t.num_tokens.max(num_tokens);
            }
            return true;
        }
        let extra = need_pages - have;
        if extra > self.free.len() {
            self.stats.alloc_failures += 1;
            return false;
        }
        let mut new_ids = Vec::with_capacity(extra);
        for _ in 0..extra {
            let id = self.free.pop().unwrap();
            self.pages[id as usize].ref_count = 1;
            new_ids.push(id);
            self.stats.pages_allocated += 1;
        }
        let used = self.num_used();
        if used > self.stats.peak_used {
            self.stats.peak_used = used;
        }
        if let Some(t) = self.tables.get_mut(&seq_id) {
            t.pages.extend(new_ids);
            t.num_tokens = t.num_tokens.max(num_tokens);
        }
        true
    }

    /// Grow by one token (decode step). Allocates a new page if crossing boundary.
    pub fn append_token_slot(&mut self, seq_id: u64) -> Option<u32> {
        let block_size = self.cfg.block_size.max(1);
        let num_tokens = self.tables.get(&seq_id).map(|t| t.num_tokens).unwrap_or(0);
        let next = num_tokens + 1;
        if !self.ensure_tokens(seq_id, next) {
            return None;
        }
        let t = self.tables.get_mut(&seq_id)?;
        t.num_tokens = next;
        let page_idx = (next - 1) / block_size;
        let offset = (next - 1) % block_size;
        let page_id = *t.pages.get(page_idx)?;
        // Physical slot = (page_id - 1) * block_size + offset  (0-based dense index)
        let slot = (page_id as u64 - 1) * block_size as u64 + offset as u64;
        Some(slot as u32)
    }

    /// Build slot mapping for a set of (seq_id, start_pos, len) ranges.
    /// Used by Pulse to index paged attention.
    pub fn build_slot_mapping(
        &self,
        ranges: &[(u64, usize, usize)], // (seq_id, start_token, num_tokens)
    ) -> SlotMapping {
        let block_size = self.cfg.block_size.max(1);
        let mut physical_slots = Vec::new();
        let mut seq_ids = Vec::new();
        let mut cu_seqlens = vec![0u32];

        for &(seq_id, start, len) in ranges {
            let Some(table) = self.tables.get(&seq_id) else {
                cu_seqlens.push(physical_slots.len() as u32);
                continue;
            };
            for pos in start..start + len {
                if pos >= table.num_tokens && pos >= table.pages.len() * block_size {
                    break;
                }
                let page_idx = pos / block_size;
                let offset = pos % block_size;
                if let Some(&page_id) = table.pages.get(page_idx) {
                    let slot = (page_id as u64 - 1) * block_size as u64 + offset as u64;
                    physical_slots.push(slot as u32);
                    seq_ids.push(seq_id);
                }
            }
            cu_seqlens.push(physical_slots.len() as u32);
        }

        SlotMapping {
            physical_slots,
            seq_ids,
            cu_seqlens,
        }
    }

    /// Free all pages for a sequence (refcount).
    pub fn free_seq(&mut self, seq_id: u64) {
        let Some(table) = self.tables.remove(&seq_id) else {
            return;
        };
        for page_id in table.pages {
            self.release_page(page_id);
        }
    }

    fn release_page(&mut self, page_id: u32) {
        if page_id == 0 || page_id as usize >= self.pages.len() {
            return;
        }
        let p = &mut self.pages[page_id as usize];
        if p.ref_count <= 0 {
            return;
        }
        p.ref_count -= 1;
        if p.ref_count == 0 {
            p.content_hash = 0;
            self.free.push(page_id);
            self.stats.pages_freed += 1;
        }
    }

    /// Pin page (future prefix share).
    pub fn pin(&mut self, page_id: u32) {
        if page_id == 0 || page_id as usize >= self.pages.len() {
            return;
        }
        let p = &mut self.pages[page_id as usize];
        if p.ref_count > 0 {
            p.ref_count += 1;
        }
    }

    /// Rough bytes if a dense pool were allocated (planning helper).
    pub fn estimated_pool_bytes(&self, bytes_per_elem: usize) -> usize {
        let pages = self.cfg.num_pages;
        let bs = self.cfg.block_size;
        let layers = self.cfg.num_layers;
        match self.cfg.kind {
            CacheKind::Gqa => {
                // K + V per layer
                layers
                    * 2
                    * pages
                    * bs
                    * self.cfg.num_kv_heads
                    * self.cfg.head_dim
                    * bytes_per_elem
            }
            CacheKind::Mla => {
                // c_kv + k_pe per layer
                let per = pages * bs * self.cfg.kv_lora_rank
                    + pages * bs * self.cfg.rope_dim;
                layers * per * bytes_per_elem
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_and_slot() {
        let mut led = BlockLedger::new(LedgerConfig {
            num_pages: 8,
            block_size: 4,
            ..Default::default()
        });
        assert!(led.ensure_tokens(1, 5));
        assert_eq!(led.table(1).unwrap().pages.len(), 2);
        let s0 = led.append_token_slot(1).unwrap();
        // after ensure 5 tokens, append makes 6
        assert!(led.table(1).unwrap().num_tokens >= 6);
        let _ = s0;
        led.free_seq(1);
        assert_eq!(led.num_free(), 8);
    }
}
