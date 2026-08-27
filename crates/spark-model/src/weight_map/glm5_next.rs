// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3-Flash-specific weight contracts.
//!
//! GLM's Kimi Delta Attention (KDA) and DeepSeek Sparse Attention (DSA)
//! deliberately do not use the Qwen GDN or DeepSeek-V4 tensor layouts. Keep
//! their names and dimensions explicit at this boundary so execution code
//! cannot silently substitute an only-similar architecture.

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use super::{DenseWeight, dense_auto, dense_keep_f32};

/// Kimi Delta Attention weights for one GLM linear-attention block.
pub struct Glm5KdaWeights {
    pub q_proj: DenseWeight,
    pub k_proj: DenseWeight,
    pub v_proj: DenseWeight,
    pub q_conv1d: DenseWeight,
    pub k_conv1d: DenseWeight,
    pub v_conv1d: DenseWeight,
    pub f_a_proj: DenseWeight,
    pub f_b_proj: DenseWeight,
    pub g_a_proj: DenseWeight,
    pub g_b_proj: DenseWeight,
    pub b_proj: DenseWeight,
    pub a_log: DenseWeight,
    pub dt_bias: DenseWeight,
    pub o_norm: DenseWeight,
    pub o_proj: DenseWeight,
}

impl Glm5KdaWeights {
    /// Pack GLM's three separately named depthwise convolution kernels into
    /// the `[Q | K | V]` channel order consumed by the causal-conv kernel.
    pub fn pack_conv1d(&self, config: &ModelConfig, gpu: &dyn GpuBackend) -> Result<DenseWeight> {
        require_glm(config)?;
        let qkv_dim = config
            .linear_num_key_heads
            .checked_mul(config.linear_key_head_dim)
            .context("GLM KDA qkv dimension overflow")?;
        let bytes = qkv_dim
            .checked_mul(config.linear_conv_kernel_dim)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<u16>()))
            .context("GLM KDA convolution byte size overflow")?;
        let packed = gpu.alloc(
            bytes
                .checked_mul(3)
                .context("GLM KDA convolution size overflow")?,
        )?;
        gpu.copy_d2d(self.q_conv1d.weight, packed, bytes)?;
        gpu.copy_d2d(self.k_conv1d.weight, packed.offset(bytes), bytes)?;
        gpu.copy_d2d(self.v_conv1d.weight, packed.offset(bytes * 2), bytes)?;
        Ok(DenseWeight { weight: packed })
    }
}

/// GLM's k-pool semantic indexer for one DSA/MLA block.
pub struct Glm5IndexerWeights {
    pub wq_b: DenseWeight,
    pub wk: DenseWeight,
    pub weights_proj: DenseWeight,
    pub k_norm_weight: DenseWeight,
    pub k_norm_bias: DenseWeight,
    pub kpool_compress_ape: DenseWeight,
    pub kpool_compress_gate: DenseWeight,
}

/// DeepSeek Sparse Attention / MLA weights for one GLM attention block.
pub struct Glm5DsaWeights {
    pub q_a_proj: DenseWeight,
    pub q_a_norm: DenseWeight,
    pub q_b_proj: DenseWeight,
    pub kv_a_proj_with_mqa: DenseWeight,
    pub kv_a_norm: DenseWeight,
    pub kv_b_proj: DenseWeight,
    pub o_proj: DenseWeight,
    pub indexer: Glm5IndexerWeights,
}

fn require_shape(store: &WeightStore, name: &str, expected: &[usize], context: &str) -> Result<()> {
    let actual = &store.get(name)?.shape;
    ensure!(
        actual == expected,
        "{context}: {name} shape {:?}, expected {:?}",
        actual,
        expected,
    );
    Ok(())
}

fn load_auto(
    store: &WeightStore,
    prefix: &str,
    expected: &[usize],
    gpu: &dyn GpuBackend,
    context: &str,
) -> Result<DenseWeight> {
    let name = format!("{prefix}.weight");
    require_shape(store, &name, expected, context)?;
    dense_auto(store, &name, gpu)
}

fn load_auto_name(
    store: &WeightStore,
    name: &str,
    expected: &[usize],
    gpu: &dyn GpuBackend,
    context: &str,
) -> Result<DenseWeight> {
    require_shape(store, name, expected, context)?;
    dense_auto(store, name, gpu)
}

fn load_f32(
    store: &WeightStore,
    name: &str,
    expected: &[usize],
    gpu: &dyn GpuBackend,
    context: &str,
) -> Result<DenseWeight> {
    require_shape(store, name, expected, context)?;
    dense_keep_f32(store, name, gpu)
}

fn require_glm(config: &ModelConfig) -> Result<()> {
    ensure!(
        config.model_type == "glm5_next",
        "GLM-5 weight contract requires model_type=glm5_next, got `{}`",
        config.model_type,
    );
    ensure!(
        config.linear_num_key_heads == config.linear_num_value_heads,
        "GLM KDA requires equal key/value head counts",
    );
    ensure!(
        config.linear_key_head_dim == config.linear_value_head_dim,
        "GLM KDA requires equal key/value head dimensions",
    );
    Ok(())
}

/// Load and validate the native KDA contract under a GLM block prefix.
pub fn load_glm5_kda(
    store: &WeightStore,
    layer_prefix: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Glm5KdaWeights> {
    require_glm(config)?;
    let p = format!("{layer_prefix}.self_attn");
    let heads = config.linear_num_key_heads;
    let head_dim = config.linear_key_head_dim;
    let qkv_dim = heads
        .checked_mul(head_dim)
        .context("GLM KDA qkv dimension overflow")?;
    let hidden = config.hidden_size;
    let conv = config.linear_conv_kernel_dim;

    Ok(Glm5KdaWeights {
        q_proj: load_auto(
            store,
            &format!("{p}.q_proj"),
            &[qkv_dim, hidden],
            gpu,
            "GLM KDA",
        )?,
        k_proj: load_auto(
            store,
            &format!("{p}.k_proj"),
            &[qkv_dim, hidden],
            gpu,
            "GLM KDA",
        )?,
        v_proj: load_auto(
            store,
            &format!("{p}.v_proj"),
            &[qkv_dim, hidden],
            gpu,
            "GLM KDA",
        )?,
        q_conv1d: load_auto(
            store,
            &format!("{p}.q_conv1d"),
            &[qkv_dim, 1, conv],
            gpu,
            "GLM KDA",
        )?,
        k_conv1d: load_auto(
            store,
            &format!("{p}.k_conv1d"),
            &[qkv_dim, 1, conv],
            gpu,
            "GLM KDA",
        )?,
        v_conv1d: load_auto(
            store,
            &format!("{p}.v_conv1d"),
            &[qkv_dim, 1, conv],
            gpu,
            "GLM KDA",
        )?,
        f_a_proj: load_auto(
            store,
            &format!("{p}.f_a_proj"),
            &[head_dim, hidden],
            gpu,
            "GLM KDA",
        )?,
        f_b_proj: load_auto(
            store,
            &format!("{p}.f_b_proj"),
            &[qkv_dim, head_dim],
            gpu,
            "GLM KDA",
        )?,
        g_a_proj: load_auto(
            store,
            &format!("{p}.g_a_proj"),
            &[head_dim, hidden],
            gpu,
            "GLM KDA",
        )?,
        g_b_proj: load_auto(
            store,
            &format!("{p}.g_b_proj"),
            &[qkv_dim, head_dim],
            gpu,
            "GLM KDA",
        )?,
        b_proj: load_auto(
            store,
            &format!("{p}.b_proj"),
            &[heads, hidden],
            gpu,
            "GLM KDA",
        )?,
        a_log: load_f32(store, &format!("{p}.A_log"), &[heads], gpu, "GLM KDA")?,
        dt_bias: load_f32(store, &format!("{p}.dt_bias"), &[qkv_dim], gpu, "GLM KDA")?,
        o_norm: load_auto(store, &format!("{p}.o_norm"), &[head_dim], gpu, "GLM KDA")?,
        o_proj: load_auto(
            store,
            &format!("{p}.o_proj"),
            &[hidden, qkv_dim],
            gpu,
            "GLM KDA",
        )?,
    })
}

/// Load and validate the native DSA/MLA + IndexPool contract.
pub fn load_glm5_dsa(
    store: &WeightStore,
    layer_prefix: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Glm5DsaWeights> {
    require_glm(config)?;
    let p = format!("{layer_prefix}.self_attn");
    let hidden = config.hidden_size;
    let q_rows = config
        .num_attention_heads
        .checked_mul(config.head_dim)
        .context("GLM DSA query dimension overflow")?;
    let kv_rows = config
        .num_key_value_heads
        .checked_mul(
            config
                .qk_nope_head_dim
                .checked_add(config.v_head_dim)
                .context("GLM DSA KV dimension overflow")?,
        )
        .context("GLM DSA KV dimension overflow")?;
    let index = format!("{p}.indexer");

    Ok(Glm5DsaWeights {
        q_a_proj: load_auto(
            store,
            &format!("{p}.q_a_proj"),
            &[config.q_lora_rank, hidden],
            gpu,
            "GLM DSA",
        )?,
        q_a_norm: load_auto(
            store,
            &format!("{p}.q_a_layernorm"),
            &[config.q_lora_rank],
            gpu,
            "GLM DSA",
        )?,
        q_b_proj: load_auto(
            store,
            &format!("{p}.q_b_proj"),
            &[q_rows, config.q_lora_rank],
            gpu,
            "GLM DSA",
        )?,
        kv_a_proj_with_mqa: load_auto(
            store,
            &format!("{p}.kv_a_proj_with_mqa"),
            &[config.kv_lora_rank, hidden],
            gpu,
            "GLM DSA",
        )?,
        kv_a_norm: load_auto(
            store,
            &format!("{p}.kv_a_layernorm"),
            &[config.kv_lora_rank],
            gpu,
            "GLM DSA",
        )?,
        kv_b_proj: load_auto(
            store,
            &format!("{p}.kv_b_proj"),
            &[kv_rows, config.kv_lora_rank],
            gpu,
            "GLM DSA",
        )?,
        o_proj: load_auto(
            store,
            &format!("{p}.o_proj"),
            &[hidden, q_rows],
            gpu,
            "GLM DSA",
        )?,
        indexer: Glm5IndexerWeights {
            wq_b: load_auto(
                store,
                &format!("{index}.wq_b"),
                &[
                    config.index_n_heads * config.index_head_dim,
                    config.q_lora_rank,
                ],
                gpu,
                "GLM IndexPool",
            )?,
            wk: load_auto(
                store,
                &format!("{index}.wk"),
                &[config.index_head_dim, hidden],
                gpu,
                "GLM IndexPool",
            )?,
            weights_proj: load_auto(
                store,
                &format!("{index}.weights_proj"),
                &[config.index_n_heads, hidden],
                gpu,
                "GLM IndexPool",
            )?,
            k_norm_weight: load_auto(
                store,
                &format!("{index}.k_norm"),
                &[config.index_head_dim],
                gpu,
                "GLM IndexPool",
            )?,
            k_norm_bias: load_auto_name(
                store,
                &format!("{index}.k_norm.bias"),
                &[config.index_head_dim],
                gpu,
                "GLM IndexPool",
            )?,
            kpool_compress_ape: load_auto(
                store,
                &format!("{index}.index_kpool_compress_ape"),
                &[config.index_kpool, config.index_head_dim],
                gpu,
                "GLM IndexPool",
            )?,
            kpool_compress_gate: load_auto(
                store,
                &format!("{index}.index_kpool_compress_gate"),
                &[config.index_head_dim, hidden],
                gpu,
                "GLM IndexPool",
            )?,
        },
    })
}
