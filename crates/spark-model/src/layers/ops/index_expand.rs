// SPDX-License-Identifier: AGPL-3.0-only

//! GLM IndexPool pool-to-token expansion dispatch.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

pub fn index_pool_expand(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    selected_pools: DevicePtr,
    token_indices: DevicePtr,
    selected_count: u32,
    kpool: u32,
    seq_len: u32,
    tail_count: u32,
    stream: u64,
) -> Result<()> {
    ensure!(kpool > 0, "GLM IndexPool expansion requires kpool > 0");
    let width = selected_count
        .checked_mul(kpool)
        .and_then(|n| n.checked_add(tail_count))
        .ok_or_else(|| anyhow::anyhow!("GLM IndexPool expansion width overflow"))?;
    KernelLaunch::new(gpu, kernel)
        .grid([width.div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(selected_pools)
        .arg_ptr(token_indices)
        .arg_u32(selected_count)
        .arg_u32(kpool)
        .arg_u32(seq_len)
        .arg_u32(tail_count)
        .launch(stream)
}
