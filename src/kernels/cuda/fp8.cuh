// Portable FP8 E4M3 helpers (software — works on all archs).
#pragma once

__device__ __forceinline__ float fp8_e4m3_to_float(unsigned char v) {
    int sign = (v >> 7) & 1;
    int exp = (v >> 3) & 0xF;
    int mant = v & 0x7;
    float f;
    if (exp == 0) {
        if (mant == 0) {
            f = 0.f;
        } else {
            // subnormal: 2^(1-bias) * (mant/2^3), bias=7 → 2^-6 * mant/8
            f = ldexpf((float)mant, -9);
        }
    } else if (exp == 0xF) {
        // E4M3 has no Inf; max finite uses exp=15 in some formats.
        // Common E4M3: exp=15 is NaN. Treat as NaN.
        f = nanf("");
    } else {
        // (1 + mant/8) * 2^(exp-7)
        f = ldexpf(1.f + (float)mant * 0.125f, exp - 7);
    }
    return sign ? -f : f;
}

__device__ __forceinline__ unsigned char float_to_fp8_e4m3(float x) {
    if (isnan(x)) return 0x7F; // NaN pattern
    int sign = x < 0.f ? 1 : 0;
    x = fabsf(x);
    if (isinf(x)) return (unsigned char)((sign << 7) | 0x7E);
    if (x == 0.f) return (unsigned char)(sign << 7);
    // Clamp to max E4M3 finite ≈ 448
    if (x > 448.f) x = 448.f;

    int exp;
    float frac = frexpf(x, &exp); // x = frac * 2^exp, 0.5 <= frac < 1
    // Want (1.m) * 2^(e-7) with e in 1..14
    // x = frac * 2^exp = (2*frac) * 2^(exp-1) → mantissa in [1,2)
    float mant1 = frac * 2.f; // [1, 2)
    int e = exp - 1 + 7;      // biased
    if (e <= 0) {
        // subnormal
        float den = ldexpf(x, 9); // mant stored as integer 0..7 scale
        int m = (int)(den + 0.5f);
        if (m > 7) m = 7;
        if (m < 0) m = 0;
        return (unsigned char)((sign << 7) | (m & 7));
    }
    if (e >= 15) {
        // max finite 0b0_1110_111 = 0x7E → 448
        return (unsigned char)((sign << 7) | 0x7E);
    }
    int m = (int)((mant1 - 1.f) * 8.f + 0.5f);
    if (m > 7) {
        m = 0;
        e += 1;
        if (e >= 15) return (unsigned char)((sign << 7) | 0x7E);
    }
    return (unsigned char)((sign << 7) | ((e & 0xF) << 3) | (m & 7));
}
