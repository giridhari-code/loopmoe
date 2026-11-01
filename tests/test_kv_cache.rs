//! KV / ledger smoke tests — GPU first (DEVICE.md).

use loopmoe::orbit::{BlockLedger, LedgerConfig};

fn select_test_device() -> candle_core::Device {
    if let Ok(d) = candle_core::Device::new_cuda(0) {
        return d;
    }
    eprintln!("CUDA unavailable — falling back to CPU");
    candle_core::Device::Cpu
}

#[test]
fn ledger_allocate_and_free_seq() {
    let _device = select_test_device(); // policy: try CUDA first
    let cfg = LedgerConfig {
        num_pages: 32,
        block_size: 16,
        ..LedgerConfig::default()
    };
    let mut ledger = BlockLedger::new(cfg);
    let seq = 1u64;
    ledger.register_seq(seq);
    let ok = ledger.ensure_tokens(seq, 40); // ceil(40/16)=3 pages
    assert!(ok, "ensure_tokens should succeed with free pages");
    let table = ledger.table(seq);
    assert!(table.is_some());
    let n = table.map(|t| t.pages.len()).unwrap_or(0);
    assert!(n >= 3, "expected >= 3 pages, got {n}");
    ledger.free_seq(seq);
    assert!(ledger.table(seq).is_none());
}
