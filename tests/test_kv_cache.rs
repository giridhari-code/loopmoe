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
        num_gpu_blocks: 32,
        block_size: 16,
        ..LedgerConfig::default()
    };
    let mut ledger = BlockLedger::new(cfg);
    let seq = 1u64;
    let ok = ledger.allocate(seq, 40); // needs ceil(40/16)=3 blocks
    assert!(ok, "allocate should succeed with free pages");
    let table = ledger.block_table(seq);
    assert!(table.is_some());
    let n = table.map(|t| t.blocks.len()).unwrap_or(0);
    assert!(n >= 3, "expected >= 3 blocks, got {n}");
    ledger.free_seq(seq);
    assert!(ledger.block_table(seq).is_none());
}
