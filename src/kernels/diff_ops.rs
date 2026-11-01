//! When training, custom CUDA ops use `apply_op*_no_bwd` and break autograd.
//! Set [`FORCE_DIFF_OPS`] so residual / silu_mul / rmsnorm use pure Candle paths.

use std::sync::atomic::{AtomicBool, Ordering};

/// When true, fused CUDA kernels that lack backward are skipped.
static FORCE_DIFF_OPS: AtomicBool = AtomicBool::new(false);

/// Enable differentiable Candle ops (call at train start).
pub fn set_force_diff_ops(on: bool) {
    FORCE_DIFF_OPS.store(on, Ordering::Relaxed);
}

#[inline]
pub fn force_diff_ops() -> bool {
    FORCE_DIFF_OPS.load(Ordering::Relaxed)
}

/// RAII guard — restores previous flag on drop.
pub struct DiffOpsGuard {
    prev: bool,
}

impl DiffOpsGuard {
    pub fn enable() -> Self {
        let prev = force_diff_ops();
        set_force_diff_ops(true);
        Self { prev }
    }
}

impl Drop for DiffOpsGuard {
    fn drop(&mut self) {
        set_force_diff_ops(self.prev);
    }
}
