//! FP8 / INT8 / AWQ loaders & dequant kernels

pub enum QuantType {
    Fp8,
    Int8,
    Awq,
}
