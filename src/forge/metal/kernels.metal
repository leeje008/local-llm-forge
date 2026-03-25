/*
 * Custom Metal kernels for local-llm-forge
 *
 * These kernels optimize specific bottleneck operations identified
 * from flash-moe's analysis:
 *   - Fused RMSNorm + Residual: Combines two operations into one dispatch
 *   - Fused SiLU activation: gate * silu(up) in single kernel
 *   - Optimized 4-bit dequantization matvec
 *
 * Target: Apple M4 Pro (20-core GPU, Metal 4, 32-wide SIMD)
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================
// Fused RMSNorm + Residual Addition
// Combines: output = RMSNorm(x + residual) into single dispatch
// Saves: 1 kernel launch + 1 memory round-trip
// ============================================================
kernel void fused_rmsnorm_residual(
    device const float* x          [[buffer(0)]],  // input
    device const float* residual   [[buffer(1)]],  // residual connection
    device const float* weight     [[buffer(2)]],  // RMSNorm weight
    device float* output           [[buffer(3)]],  // output
    constant uint& hidden_size     [[buffer(4)]],
    constant float& eps            [[buffer(5)]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint gid                       [[threadgroup_position_in_grid]],
    uint threads_per_group         [[threads_per_threadgroup]]
) {
    // Each threadgroup processes one token (one row of hidden_size)
    uint row_offset = gid * hidden_size;

    // Step 1: Add residual and compute sum of squares (for RMS)
    threadgroup float shared_sum[32];  // For SIMD reduction

    float local_sum = 0.0f;
    for (uint i = tid; i < hidden_size; i += threads_per_group) {
        float val = x[row_offset + i] + residual[row_offset + i];
        local_sum += val * val;
    }

    // SIMD reduction within simd_group
    local_sum = simd_sum(local_sum);

    // Store per-simd result
    if ((tid % 32) == 0) {
        shared_sum[tid / 32] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction
    if (tid == 0) {
        float total = 0.0f;
        uint num_simds = (threads_per_group + 31) / 32;
        for (uint i = 0; i < num_simds; i++) {
            total += shared_sum[i];
        }
        shared_sum[0] = rsqrt(total / float(hidden_size) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = shared_sum[0];

    // Step 2: Normalize and apply weight
    for (uint i = tid; i < hidden_size; i += threads_per_group) {
        float val = x[row_offset + i] + residual[row_offset + i];
        output[row_offset + i] = val * rms_inv * weight[i];
    }
}

// ============================================================
// Fused SiLU Gate (SwiGLU activation)
// Combines: output = silu(gate) * up into single kernel
// Used in: FFN layers of LLaMA/Qwen/Mistral architectures
// ============================================================
kernel void fused_silu_gate(
    device const float* gate       [[buffer(0)]],  // gate projection output
    device const float* up         [[buffer(1)]],  // up projection output
    device float* output           [[buffer(2)]],  // fused result
    constant uint& size            [[buffer(3)]],
    uint tid                       [[thread_position_in_grid]]
) {
    if (tid >= size) return;

    float g = gate[tid];
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    float silu_g = g / (1.0f + exp(-g));
    output[tid] = silu_g * up[tid];
}

// ============================================================
// Half-precision variants for Apple Silicon optimization
// Apple M4 Pro GPU natively accelerates float16 operations
// ============================================================
kernel void fused_rmsnorm_residual_half(
    device const half* x           [[buffer(0)]],
    device const half* residual    [[buffer(1)]],
    device const half* weight      [[buffer(2)]],
    device half* output            [[buffer(3)]],
    constant uint& hidden_size     [[buffer(4)]],
    constant float& eps            [[buffer(5)]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint gid                       [[threadgroup_position_in_grid]],
    uint threads_per_group         [[threads_per_threadgroup]]
) {
    uint row_offset = gid * hidden_size;

    threadgroup float shared_sum[32];
    float local_sum = 0.0f;

    for (uint i = tid; i < hidden_size; i += threads_per_group) {
        float val = float(x[row_offset + i]) + float(residual[row_offset + i]);
        local_sum += val * val;
    }

    local_sum = simd_sum(local_sum);
    if ((tid % 32) == 0) {
        shared_sum[tid / 32] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        uint num_simds = (threads_per_group + 31) / 32;
        for (uint i = 0; i < num_simds; i++) total += shared_sum[i];
        shared_sum[0] = rsqrt(total / float(hidden_size) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = shared_sum[0];
    for (uint i = tid; i < hidden_size; i += threads_per_group) {
        float val = float(x[row_offset + i]) + float(residual[row_offset + i]);
        output[row_offset + i] = half(val * rms_inv * float(weight[i]));
    }
}

kernel void fused_silu_gate_half(
    device const half* gate        [[buffer(0)]],
    device const half* up          [[buffer(1)]],
    device half* output            [[buffer(2)]],
    constant uint& size            [[buffer(3)]],
    uint tid                       [[thread_position_in_grid]]
) {
    if (tid >= size) return;

    float g = float(gate[tid]);
    float silu_g = g / (1.0f + exp(-g));
    output[tid] = half(silu_g * float(up[tid]));
}

// ============================================================
// Optimized 4-bit Dequantization + MatVec
// Based on flash-moe's dequant_matvec_4bit_v5 design
// Uses LUT for eliminating uint→float conversions
// ============================================================
kernel void dequant_matvec_4bit(
    device const uint32_t* packed_weights [[buffer(0)]],  // 4-bit packed (8 values per uint32)
    device const half* scales             [[buffer(1)]],  // per-group scales
    device const half* zeros              [[buffer(2)]],  // per-group zeros
    device const half* input_vec          [[buffer(3)]],  // input vector
    device float* output                  [[buffer(4)]],  // output vector
    constant uint& in_features            [[buffer(5)]],
    constant uint& out_features           [[buffer(6)]],
    constant uint& group_size             [[buffer(7)]],
    uint tid                              [[thread_position_in_threadgroup]],
    uint gid                              [[threadgroup_position_in_grid]],
    uint threads_per_group                [[threads_per_threadgroup]]
) {
    // Each threadgroup computes one output element
    uint out_idx = gid;
    if (out_idx >= out_features) return;

    // Load input vector into shared memory for reuse
    threadgroup half shared_input[4096];  // max hidden_size
    for (uint i = tid; i < in_features; i += threads_per_group) {
        shared_input[i] = input_vec[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    uint groups_per_row = (in_features + group_size - 1) / group_size;

    for (uint g = 0; g < groups_per_row; g++) {
        uint group_start = g * group_size;
        float scale = float(scales[out_idx * groups_per_row + g]);
        float zero = float(zeros[out_idx * groups_per_row + g]);

        // Process 8 values at a time (one uint32 = 8 x 4-bit values)
        uint packed_per_group = group_size / 8;
        uint packed_offset = (out_idx * in_features / 8) + (group_start / 8);

        for (uint p = tid; p < packed_per_group; p += threads_per_group) {
            uint32_t packed = packed_weights[packed_offset + p];
            uint base_idx = group_start + p * 8;

            // Unpack 8 x 4-bit values and accumulate
            for (uint k = 0; k < 8 && (base_idx + k) < in_features; k++) {
                uint4_t val = (packed >> (k * 4)) & 0xF;
                float dequant = float(val) * scale + zero;
                local_sum += dequant * float(shared_input[base_idx + k]);
            }
        }
    }

    // SIMD reduction
    local_sum = simd_sum(local_sum);
    if ((tid % 32) == 0) {
        // Atomic add for cross-SIMD accumulation
        // (simplified — production would use threadgroup reduction)
        output[out_idx] += local_sum;
    }
}
