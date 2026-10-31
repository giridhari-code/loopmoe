//! BlockLedger — physical paged memory + O5 prefix hash sharing.

use std::collections::HashMap;

use super::prefix::{hash_prompt_pages, hash_token_block, PrefixStats};

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
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub kv_lora_rank: usize,
    pub rope_dim: usize,
    /// Enable prefix content-hash sharing (O5).
    pub enable_prefix_cache: bool,
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
            enable_prefix_cache: true,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct LedgerStats {
    pub pages_allocated: u64,
    pub pages_freed: u64,
    pub peak_used: usize,
    pub alloc_failures: u64,
    pub prefix: PrefixStats,
}

#[derive(Debug, Clone)]
struct PageMeta {
    ref_count: i32,
    content_hash: u64,
}

#[derive(Debug, Clone, Default)]
pub struct SeqBlockTable {
    pub pages: Vec<u32>,
    pub num_tokens: usize,
    /// How many leading pages came from prefix cache (shared).
    pub prefix_pages: usize,
}

impl SeqBlockTable {
    pub fn num_pages(&self) -> usize {
        self.pages.len()
    }

    pub fn clear(&mut self) {
        self.pages.clear();
        self.num_tokens = 0;
        self.prefix_pages = 0;
    }
}

#[derive(Debug, Clone, Default)]
pub struct SlotMapping {
    pub physical_slots: Vec<u32>,
    pub seq_ids: Vec<u64>,
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
    free: Vec<u32>,
    tables: HashMap<u64, SeqBlockTable>,
    /// content_hash → physical page id (only pages with ref_count > 0).
    prefix_index: HashMap<u64, u32>,
    pub stats: LedgerStats,
}

impl BlockLedger {
    pub fn new(cfg: LedgerConfig) -> Self {
        let n = cfg.num_pages.max(1);
        let mut pages = Vec::with_capacity(n + 1);
        pages.push(PageMeta {
            ref_count: 0,
            content_hash: 0,
        });
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
            prefix_index: HashMap::new(),
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

    pub fn register_seq(&mut self, seq_id: u64) {
        self.tables.entry(seq_id).or_default();
    }

    pub fn table(&self, seq_id: u64) -> Option<&SeqBlockTable> {
        self.tables.get(&seq_id)
    }

    pub fn table_mut(&mut self, seq_id: u64) -> Option<&mut SeqBlockTable> {
        self.tables.get_mut(&seq_id)
    }

    /// Allocate / reuse pages for a prompt using prefix cache when enabled.
    /// Returns number of pages satisfied from cache (prefix hits).
    pub fn allocate_with_prefix(&mut self, seq_id: u64, prompt_tokens: &[u32]) -> usize {
        self.tables.entry(seq_id).or_default();
        let bs = self.cfg.block_size.max(1);
        let hashes = hash_prompt_pages(prompt_tokens, bs);
        let mut shared = 0usize;
        let mut pages = Vec::with_capacity(hashes.len());

        if self.cfg.enable_prefix_cache {
            for &h in &hashes {
                if let Some(&pid) = self.prefix_index.get(&h) {
                    if pid > 0 && (pid as usize) < self.pages.len() {
                        let meta = &mut self.pages[pid as usize];
                        if meta.ref_count > 0 && meta.content_hash == h {
                            meta.ref_count += 1;
                            pages.push(pid);
                            shared += 1;
                            self.stats.prefix.hits += 1;
                            self.stats.prefix.pages_shared += 1;
                            continue;
                        }
                    }
                    self.prefix_index.remove(&h);
                }
                self.stats.prefix.misses += 1;
                // miss — allocate fresh
                if let Some(pid) = self.alloc_one() {
                    pages.push(pid);
                } else {
                    self.stats.alloc_failures += 1;
                    break;
                }
            }
        } else {
            for _ in 0..hashes.len() {
                self.stats.prefix.misses += 1;
                if let Some(pid) = self.alloc_one() {
                    pages.push(pid);
                } else {
                    self.stats.alloc_failures += 1;
                    break;
                }
            }
        }

        if let Some(t) = self.tables.get_mut(&seq_id) {
            t.pages = pages;
            t.num_tokens = prompt_tokens.len();
            t.prefix_pages = shared;
        }
        shared
    }

    /// After computing KV for a unique page, seal it into the prefix index.
    pub fn seal_page(&mut self, page_id: u32, token_block: &[u32], parent_hash: u64) {
        if !self.cfg.enable_prefix_cache || page_id == 0 || page_id as usize >= self.pages.len() {
            return;
        }
        let h = hash_token_block(token_block, parent_hash);
        let meta = &mut self.pages[page_id as usize];
        if meta.ref_count <= 0 {
            return;
        }
        meta.content_hash = h;
        self.prefix_index.insert(h, page_id);
        self.stats.prefix.pages_sealed += 1;
    }

    /// Seal all non-shared pages of a sequence after prefill.
    pub fn seal_seq_prompt(&mut self, seq_id: u64, prompt_tokens: &[u32]) {
        if !self.cfg.enable_prefix_cache {
            return;
        }
        let bs = self.cfg.block_size.max(1);
        let Some(table) = self.tables.get(&seq_id).cloned() else {
            return;
        };
        let mut parent = 0u64;
        for (i, &pid) in table.pages.iter().enumerate() {
            if i < table.prefix_pages {
                // already shared — chain parent hash
                if let Some(h) = self.pages.get(pid as usize).map(|p| p.content_hash) {
                    parent = h;
                }
                continue;
            }
            let start = i * bs;
            if start >= prompt_tokens.len() {
                break;
            }
            let end = (start + bs).min(prompt_tokens.len());
            self.seal_page(pid, &prompt_tokens[start..end], parent);
            parent = self.pages[pid as usize].content_hash;
        }
    }

    fn alloc_one(&mut self) -> Option<u32> {
        let id = self.free.pop()?;
        self.pages[id as usize].ref_count = 1;
        self.pages[id as usize].content_hash = 0;
        self.stats.pages_allocated += 1;
        let used = self.num_used();
        if used > self.stats.peak_used {
            self.stats.peak_used = used;
        }
        Some(id)
    }

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
            if let Some(id) = self.alloc_one() {
                new_ids.push(id);
            } else {
                return false;
            }
        }
        if let Some(t) = self.tables.get_mut(&seq_id) {
            t.pages.extend(new_ids);
            t.num_tokens = t.num_tokens.max(num_tokens);
        }
        true
    }

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
        let slot = (page_id as u64 - 1) * block_size as u64 + offset as u64;
        Some(slot as u32)
    }

    pub fn build_slot_mapping(&self, ranges: &[(u64, usize, usize)]) -> SlotMapping {
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
        let h = self.pages[page_id as usize].content_hash;
        let p = &mut self.pages[page_id as usize];
        if p.ref_count <= 0 {
            return;
        }
        p.ref_count -= 1;
        if p.ref_count == 0 {
            if h != 0 {
                if self.prefix_index.get(&h) == Some(&page_id) {
                    self.prefix_index.remove(&h);
                }
            }
            p.content_hash = 0;
            self.free.push(page_id);
            self.stats.pages_freed += 1;
        }
    }

    pub fn pin(&mut self, page_id: u32) {
        if page_id == 0 || page_id as usize >= self.pages.len() {
            return;
        }
        let p = &mut self.pages[page_id as usize];
        if p.ref_count > 0 {
            p.ref_count += 1;
        }
    }

    pub fn estimated_pool_bytes(&self, bytes_per_elem: usize) -> usize {
        let pages = self.cfg.num_pages;
        let bs = self.cfg.block_size;
        let layers = self.cfg.num_layers;
        match self.cfg.kind {
            CacheKind::Gqa => {
                layers * 2 * pages * bs * self.cfg.num_kv_heads * self.cfg.head_dim * bytes_per_elem
            }
            CacheKind::Mla => {
                let per = pages * bs * self.cfg.kv_lora_rank + pages * bs * self.cfg.rope_dim;
                layers * per * bytes_per_elem
            }
        }
    }
}
