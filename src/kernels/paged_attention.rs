//! Paged attention — Rust entry (device kernel TBD in CUDA C++).
//!
//! See KERNELS.md. Implementation will call a sm_75-safe PTX kernel
//! registered via build.rs, with Candle fallback.

use candle_core::{Result as CResult, Tensor};

/// Placeholder: returns error until CUDA kernel is wired.
pub fn paged_attention_forward(
    _q: &Tensor,
    _k_cache: &Tensor,
    _v_cache: &Tensor,
    _block_tables: &Tensor,
    _context_lens: &Tensor,
) -> CResult<Tensor> {
    candle_core::bail!("paged_attention: not implemented — see KERNELS.md")
}
