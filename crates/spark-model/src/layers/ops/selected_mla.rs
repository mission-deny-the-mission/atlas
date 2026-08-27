// SPDX-License-Identifier: AGPL-3.0-only

//! Reference GLM selected-token latent MLA dispatch.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

pub fn selected_mla_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_pool: DevicePtr,
    v_pool: DevicePtr,
    block_table: DevicePtr,
    token_indices: DevicePtr,
    output: DevicePtr,
    selected_count: u32,
    block_size: u32,
    kv_lora: u32,
    num_heads: u32,
    inv_sqrt_d: f32,
    stream: u64,
) -> Result<()> {
    ensure!(
        kv_lora <= 512,
        "selected MLA reference kernel supports kv_lora <= 512"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([num_heads, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_pool)
        .arg_ptr(v_pool)
        .arg_ptr(block_table)
        .arg_ptr(token_indices)
        .arg_ptr(output)
        .arg_u32(selected_count)
        .arg_u32(block_size)
        .arg_u32(kv_lora)
        .arg_u32(num_heads)
        .arg_f32(inv_sqrt_d)
        .launch(stream)
}
