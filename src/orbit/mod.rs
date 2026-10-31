//! # Orbit
//!
//! Continuous-batch inference runtime for LoopedMoE + Tesla.
//!
//! Phases:
//! - **O1** BlockLedger — physical paged KV/MLA + slot_mapping (this module)
//! - **O2** PackedBatch decode (B>1 one forward)
//! - **O3** Chunked prefill + Tesla-in-schedule
//! - **O4** Frontier OpenAI server on Orbit core
//! - **O5** Prefix hash + EP collectives
//!
//! Layout:
//! ```text
//! OrbitCore
//!   ├── OrbitSched   (time — who runs this step)
//!   ├── BlockLedger  (space — physical pages)
//!   └── Pulse        (execute — packed forward)  // O2+
//! ```

pub mod ledger;
pub mod sched;

pub use ledger::{
    BlockLedger, CacheKind, LedgerConfig, LedgerStats, SlotMapping, SeqBlockTable,
};
pub use sched::OrbitSched;
