// SPDX-License-Identifier: AGPL-3.0-only

//! GLM Kimi Delta Attention kernel dispatch.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

#[allow(clippy::too_many_arguments)]
pub fn kda_recurrent_decode(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    state: DevicePtr,
    qkv: DevicePtr,
    forget: DevicePtr,
    beta_input: DevicePtr,
    a_log: DevicePtr,
    dt_bias: DevicePtr,
    output: DevicePtr,
    num_rows: u32,
    num_heads: u32,
    head_dim: u32,
    lower_bound: Option<f32>,
    stream: u64,
) -> Result<()> {
    let (lower_bound, has_lower_bound) = match lower_bound {
        Some(value) => (value, 1),
        None => (0.0, 0),
    };
    KernelLaunch::new(gpu, kernel)
        .grid([num_heads, num_rows, 1])
        .block([head_dim, 1, 1])
        .arg_ptr(state)
        .arg_ptr(qkv)
        .arg_ptr(forget)
        .arg_ptr(beta_input)
        .arg_ptr(a_log)
        .arg_ptr(dt_bias)
        .arg_ptr(output)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .arg_f32(lower_bound)
        .arg_u32(has_lower_bound)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn kda_gated_rms_norm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
    num_rows: u32,
    num_heads: u32,
    head_dim: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_heads, num_rows, 1])
        .block([head_dim, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate)
        .arg_ptr(weight)
        .arg_ptr(output)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .arg_f32(eps)
        .launch(stream)
}
