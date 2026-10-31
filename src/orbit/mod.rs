//! # Orbit
//!
//! Continuous-batch inference runtime for LoopedMoE + Tesla.
//!
//! - O1 BlockLedger
//! - O2 / O2.1 PackedBatch + Pulse B>1
//! - O3 Chunked prefill + Tesla schedule
//! - O4 Frontier HTTP
//! - **O5 Prefix hash + EP packing**

pub mod ledger;
pub mod sched;
pub mod pack;
pub mod pulse;
pub mod core;
pub mod prefix;

pub use ledger::{
    BlockLedger, CacheKind, LedgerConfig, LedgerStats, SeqBlockTable, SlotMapping,
};
pub use sched::OrbitSched;
pub use pack::{PackItem, PackRole, PackedBatch};
pub use pulse::{Pulse, PulseMetrics, PulseOutput};
pub use core::{OrbitCore, OrbitMetrics, PackResolve};
pub use prefix::{hash_prompt_pages, hash_token_block, PrefixStats};
