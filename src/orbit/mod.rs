//! # Orbit
//!
//! Continuous-batch inference runtime for LoopedMoE + Tesla.
//!
//! - **O1** BlockLedger — physical pages + slot_mapping
//! - **O2** PackedBatch + Pulse — single execute entry (per-seq forward today)
//! - **O3** Chunked prefill + Tesla-in-schedule
//! - **O4** Frontier server
//! - **O5** Prefix hash + EP

pub mod ledger;
pub mod sched;
pub mod pack;
pub mod pulse;
pub mod core;

pub use ledger::{
    BlockLedger, CacheKind, LedgerConfig, LedgerStats, SeqBlockTable, SlotMapping,
};
pub use sched::OrbitSched;
pub use pack::{PackItem, PackRole, PackedBatch};
pub use pulse::{Pulse, PulseMetrics, PulseOutput};
pub use core::{OrbitCore, OrbitMetrics, PackResolve};
