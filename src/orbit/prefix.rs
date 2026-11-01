//! Prefix page hashing for BlockLedger (Orbit O5).
//!
//! Shared system prompts / repeated prefixes reuse physical pages via content hash.
//! Hash is xxHash-style FNV-1a over block token ids (stable, no extra deps).

/// FNV-1a 64-bit over a slice of token ids (one ledger page worth).
pub fn hash_token_block(tokens: &[u32], parent_hash: u64) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut h = if parent_hash == 0 {
        FNV_OFFSET
    } else {
        // chain with parent so position in prefix matters
        parent_hash
    };
    for &t in tokens {
        h ^= t as u64;
        h = h.wrapping_mul(FNV_PRIME);
        // mix length position lightly
        h ^= 0x9e3779b97f4a7c15;
        h = h.wrapping_mul(FNV_PRIME);
    }
    if h == 0 {
        1 // 0 reserved = "no hash"
    } else {
        h
    }
}

/// Hash successive pages of a prompt; returns one hash per full/partial page.
pub fn hash_prompt_pages(tokens: &[u32], block_size: usize) -> Vec<u64> {
    let bs = block_size.max(1);
    let mut out = Vec::new();
    let mut parent = 0u64;
    let mut i = 0;
    while i < tokens.len() {
        let end = (i + bs).min(tokens.len());
        let h = hash_token_block(&tokens[i..end], parent);
        out.push(h);
        parent = h;
        i = end;
    }
    out
}

#[derive(Debug, Default, Clone)]
pub struct PrefixStats {
    pub hits: u64,
    pub misses: u64,
    pub pages_shared: u64,
    pub pages_sealed: u64,
}
