//! Fertile backend selection: CUDA → ARM → CPU.

use candle_core::Device;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Cuda,
    Arm,
    Cpu,
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendKind::Cuda => write!(f, "fertile-cuda"),
            BackendKind::Arm => write!(f, "fertile-arm"),
            BackendKind::Cpu => write!(f, "fertile-cpu"),
        }
    }
}

/// 1) CUDA if device is Cuda  2) ARM on aarch64  3) CPU
pub fn select_backend(device: &Device) -> BackendKind {
    if device.is_cuda() {
        return BackendKind::Cuda;
    }
    if cfg!(target_arch = "aarch64") {
        return BackendKind::Arm;
    }
    BackendKind::Cpu
}

#[derive(Clone, Debug)]
pub struct BackendInfo {
    pub kind: BackendKind,
    pub device_dbg: String,
}

impl BackendInfo {
    pub fn from_device(device: &Device) -> Self {
        Self {
            kind: select_backend(device),
            device_dbg: format!("{device:?}"),
        }
    }
}
