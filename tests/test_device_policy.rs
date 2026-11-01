//! Device policy (see DEVICE.md): try CUDA first; CPU only if CUDA missing.
//! Also covers Orbit O5 logical EP pack/unpack.

use candle_core::{Device, DType, Tensor};
use loopmoe::parallel::{expert_to_rank, pack_by_ep_rank, unpack_combine, CombineMeta};

/// Resolve device: CUDA first, then CPU with a clear warning (same policy as train/mesh).
fn select_test_device() -> Device {
    if let Ok(d) = Device::new_cuda(0) {
        return d;
    }
    eprintln!("CUDA unavailable — falling back to CPU");
    Device::Cpu
}

#[test]
fn device_prefers_cuda_when_available() {
    let d = select_test_device();
    match Device::new_cuda(0) {
        Ok(_) => assert!(d.is_cuda(), "CUDA is available but test device is not CUDA"),
        Err(_) => assert!(d.is_cpu(), "no CUDA → must use CPU"),
    }
}

#[test]
fn expert_to_rank_matches_contiguous_shards() {
    let n_routed = 10;
    let ep = 4;
    // ranks 0..3 get counts 3,3,2,2 (base=2, rem=2)
    assert_eq!(expert_to_rank(0, n_routed, ep), 0);
    assert_eq!(expert_to_rank(2, n_routed, ep), 0);
    assert_eq!(expert_to_rank(3, n_routed, ep), 1);
    assert_eq!(expert_to_rank(5, n_routed, ep), 1);
    assert_eq!(expert_to_rank(6, n_routed, ep), 2);
    assert_eq!(expert_to_rank(7, n_routed, ep), 2);
    assert_eq!(expert_to_rank(8, n_routed, ep), 3);
    assert_eq!(expert_to_rank(9, n_routed, ep), 3);
}

#[test]
fn pack_unpack_roundtrip_preserves_rows() -> candle_core::Result<()> {
    let device = select_test_device();
    let t = 6usize;
    let h = 4usize;
    // rows = [0,1,2,...,23] shaped (6,4)
    let data: Vec<f32> = (0..(t * h)).map(|i| i as f32).collect();
    let tokens = Tensor::from_vec(data, (t, h), &device)?;
    // primary experts → ranks via n_routed=8, ep=2 (experts 0-3 → rank0, 4-7 → rank1)
    let topk = Tensor::from_vec(
        vec![0i64, 5, 1, 6, 2, 7],
        (t, 1),
        &device,
    )?;

    let (packed, meta) = pack_by_ep_rank(&tokens, &topk, 8, 2)?;
    assert_eq!(meta.tokens_per_rank.len(), 2);
    assert_eq!(meta.tokens_per_rank.iter().sum::<usize>(), t);
    assert!(meta.inverse_perm.is_some());

    let restored = unpack_combine(&packed, &CombineMeta::from_dispatch(&meta))?;
    let a = tokens.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = restored.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(a, b, "unpack must restore original row order");
    Ok(())
}

#[test]
fn pack_ep1_is_identity() -> candle_core::Result<()> {
    let device = select_test_device();
    let tokens = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
    let topk = Tensor::from_vec(vec![0i64, 1], (2, 1), &device)?;
    let (out, meta) = pack_by_ep_rank(&tokens, &topk, 4, 1)?;
    assert!(meta.inverse_perm.is_none());
    let a = tokens.flatten_all()?.to_vec1::<f32>()?;
    let b = out.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(a, b);
    Ok(())
}
