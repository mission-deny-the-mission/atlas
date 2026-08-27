// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3-Flash weight loader.
//!
//! GLM keeps Kimi Delta Attention and sparse MLA under the same decoder-layer
//! namespace. KDA has its own layer implementation; the no-RoPE MLA projection
//! geometry can use Atlas's absorbed MLA cache path while the IndexPool tensors
//! remain attached to the DSA contract for the sparse-attention follow-up.

use anyhow::{Result, ensure};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::layers::qwen3_attention::{HcSiteWeights, HcWeights, MlaWeights, Qwen3AttentionLayer};
use crate::layers::{
    DenseFfnLayer, DenseFfnWeights, FfnComponent, Glm5KdaLayer, MoeLayer, VisionEncoder,
};
use crate::weight_map::{
    AttentionWeights, DenseWeight, ExpertWeight, MoeWeights, QuantizedWeight, dense_auto,
    dense_keep_f32, load_glm5_dsa, load_glm5_kda, quantize_to_nvfp4,
};

pub struct Glm5NextWeightLoader;

fn null_dense() -> DenseWeight {
    DenseWeight {
        weight: DevicePtr::NULL,
    }
}

fn load_dense_ffn(
    store: &WeightStore,
    prefix: &str,
    intermediate: usize,
    hidden: usize,
    gpu: &dyn GpuBackend,
    absmax: KernelHandle,
    quantize: KernelHandle,
    stream: u64,
) -> Result<FfnComponent> {
    let gate = dense_auto(store, &format!("{prefix}.gate_proj.weight"), gpu)?;
    let up = dense_auto(store, &format!("{prefix}.up_proj.weight"), gpu)?;
    let down = dense_auto(store, &format!("{prefix}.down_proj.weight"), gpu)?;
    let gate = quantize_to_nvfp4(&gate, intermediate, hidden, gpu, absmax, quantize, stream)?;
    let up = quantize_to_nvfp4(&up, intermediate, hidden, gpu, absmax, quantize, stream)?;
    let down = quantize_to_nvfp4(&down, hidden, intermediate, gpu, absmax, quantize, stream)?;
    let weights = DenseFfnWeights {
        gate_proj_t: Some(gate.transpose_for_gemm(gpu, intermediate, hidden)?),
        up_proj_t: Some(up.transpose_for_gemm(gpu, intermediate, hidden)?),
        down_proj_t: Some(down.transpose_for_gemm(gpu, hidden, intermediate)?),
        gate_proj: gate,
        up_proj: up,
        down_proj: down,
    };
    Ok(FfnComponent::Dense(DenseFfnLayer::new(weights, gpu)?))
}

fn load_moe(
    store: &WeightStore,
    prefix: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    absmax: KernelHandle,
    quantize: KernelHandle,
    stream: u64,
) -> Result<FfnComponent> {
    let h = config.hidden_size;
    let inter = config.moe_intermediate_size;
    let gate = dense_auto(store, &format!("{prefix}.gate.weight"), gpu)?;
    let mut experts = Vec::with_capacity(config.num_experts);
    for expert in 0..config.num_experts {
        let ep = format!("{prefix}.experts.{expert}");
        let routed_gate = dense_auto(store, &format!("{ep}.gate_proj.weight"), gpu)?;
        let routed_up = dense_auto(store, &format!("{ep}.up_proj.weight"), gpu)?;
        let routed_down = dense_auto(store, &format!("{ep}.down_proj.weight"), gpu)?;
        experts.push(ExpertWeight {
            gate_proj: quantize_to_nvfp4(&routed_gate, inter, h, gpu, absmax, quantize, stream)?,
            up_proj: quantize_to_nvfp4(&routed_up, inter, h, gpu, absmax, quantize, stream)?,
            down_proj: quantize_to_nvfp4(&routed_down, h, inter, gpu, absmax, quantize, stream)?,
        });
    }

    let shared = format!("{prefix}.shared_experts");
    let shared_gate = dense_auto(store, &format!("{shared}.gate_proj.weight"), gpu)?;
    let shared_up = dense_auto(store, &format!("{shared}.up_proj.weight"), gpu)?;
    let shared_down = dense_auto(store, &format!("{shared}.down_proj.weight"), gpu)?;
    let shared_expert = ExpertWeight {
        gate_proj: quantize_to_nvfp4(&shared_gate, inter, h, gpu, absmax, quantize, stream)?,
        up_proj: quantize_to_nvfp4(&shared_up, inter, h, gpu, absmax, quantize, stream)?,
        down_proj: quantize_to_nvfp4(&shared_down, h, inter, gpu, absmax, quantize, stream)?,
    };
    let correction_bias = Some(dense_keep_f32(
        store,
        &format!("{prefix}.gate.e_score_correction_bias"),
        gpu,
    )?);
    let weights = MoeWeights {
        gate,
        shared_expert,
        // GLM's shared expert is ungated; a null gate is the explicit Atlas
        // representation for an always-on shared contribution.
        shared_expert_gate: null_dense(),
        experts,
        router_pre_norm: None,
        correction_bias,
    };
    let mut layer = MoeLayer::new(weights, config.num_experts, None, gpu, config)?;
    layer.predequant_for_prefill(gpu, config, stream)?;
    Ok(FfnComponent::Moe(layer))
}

fn load_hc_site(
    store: &WeightStore,
    prefix: &str,
    site: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<HcSiteWeights> {
    let hc_dim = config.hc_mult * config.hidden_size;
    let mix = (2 + config.hc_mult) * config.hc_mult;
    let hc_fn = super::deepseek_v4::assemble::load_hc_f32(
        store,
        &[format!("{prefix}.hc_{site}_fn")],
        mix * hc_dim,
        gpu,
    )?;
    let hc_base = super::deepseek_v4::assemble::load_hc_f32(
        store,
        &[format!("{prefix}.hc_{site}_base")],
        mix,
        gpu,
    )?;
    let hc_scale = super::deepseek_v4::assemble::load_hc_f32(
        store,
        &[format!("{prefix}.hc_{site}_scale")],
        3,
        gpu,
    )?;
    Ok(HcSiteWeights {
        hc_fn,
        hc_base,
        hc_scale,
    })
}

fn load_hc(
    store: &WeightStore,
    prefix: &str,
    _layer_idx: usize,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<HcWeights> {
    Ok(HcWeights {
        attn: load_hc_site(store, prefix, "attn", config, gpu)?,
        ffn: load_hc_site(store, prefix, "ffn", config, gpu)?,
        head: None,
        hc_mult: config.hc_mult,
        sinkhorn_iters: config.hc_sinkhorn_iters,
        hc_eps: config.hc_eps,
        final_mean: true,
    })
}

fn load_dsa_layer(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_idx: usize,
    attn_layer_idx: usize,
    ffn: FfnComponent,
) -> Result<Box<dyn TransformerLayer>> {
    let prefix = format!("model.language_model.layers.{layer_idx}");
    let attn_prefix = format!("{prefix}.self_attn");
    let dsa = load_glm5_dsa(store, &prefix, config, gpu)?;
    let q_b_shape = store
        .get(&format!("{attn_prefix}.q_b_proj.weight"))?
        .shape
        .clone();
    let kv_b_shape = store
        .get(&format!("{attn_prefix}.kv_b_proj.weight"))?
        .shape
        .clone();
    let (w_uk_t, w_uv, wq_b_rope, w_uk_host) = super::deepseek_v4::compute::build_per_head_views(
        &dsa.kv_b_proj,
        &kv_b_shape,
        &dsa.q_b_proj,
        &q_b_shape,
        config,
        gpu,
    )?;
    let w_qk_absorbed = super::deepseek_v4::compute::build_w_qk_absorbed(
        &dsa.q_b_proj,
        &q_b_shape,
        &w_uk_t,
        config,
        gpu,
    )?;
    let (w_uk_block_diag, w_uv_block_diag) =
        super::deepseek_v4::compute::build_block_diagonals(&w_uk_host, &w_uv, config, gpu)?;
    let mla = MlaWeights {
        glm_indexer: Some(dsa.indexer),
        wq_a: dsa.q_a_proj,
        wq_a_nvfp4: None,
        wq_a_fp8: None,
        wq_b: dsa.q_b_proj,
        wq_b_nvfp4: None,
        wq_b_fp8: None,
        q_a_norm: dsa.q_a_norm,
        wkv_a: dsa.kv_a_proj_with_mqa,
        wkv_a_nvfp4: None,
        wkv_a_fp8: None,
        wkv_b: dsa.kv_b_proj,
        kv_a_norm: dsa.kv_a_norm,
        wkv_a_rope: null_dense(),
        wkv_a_merged: null_dense(),
        wo: dsa.o_proj,
        wo_nvfp4: None,
        wo_a: null_dense(),
        wo_a_nvfp4: None,
        wo_a_fp8: None,
        wo_b: null_dense(),
        wo_b_nvfp4: None,
        wo_b_fp8: None,
        w_uk_t,
        w_uv,
        wq_b_rope,
        w_qk_absorbed,
        w_uk_block_diag,
        w_uv_block_diag,
        yarn_inv_freq: DevicePtr::NULL,
        main_inv_freq: DevicePtr::NULL,
        q_lora_rank: config.q_lora_rank,
        kv_lora_rank: config.kv_lora_rank,
        o_lora_rank: 0,
        nope: config.qk_nope_head_dim,
        rope: 0,
        v_dim: config.v_head_dim,
        compressor: None,
        attn_sink: DevicePtr::NULL,
    };
    let input_norm = dense_auto(store, &format!("{prefix}.input_layernorm.weight"), gpu)?;
    let post_norm = dense_auto(
        store,
        &format!("{prefix}.post_attention_layernorm.weight"),
        gpu,
    )?;
    let attn = AttentionWeights {
        q_proj: null_dense(),
        k_proj: null_dense(),
        v_proj: null_dense(),
        o_proj: QuantizedWeight::null(),
        q_norm: null_dense(),
        k_norm: null_dense(),
        q_norm_full: None,
        k_norm_full: None,
        k_scale: 1.0,
        v_scale: 1.0,
    };
    let mut layer = Qwen3AttentionLayer::new_ungated(
        input_norm,
        attn,
        post_norm,
        ffn,
        attn_layer_idx,
        None,
        None,
        None,
        gpu,
        KvCacheDtype::Bf16,
        0,
        config,
    )?;
    layer.set_physical_layer_idx(layer_idx);
    layer.set_mla_weights(mla);
    layer.set_hc_weights(load_hc(store, &prefix, layer_idx, config, gpu)?);
    Ok(Box::new(layer))
}

impl ModelWeightLoader for Glm5NextWeightLoader {
    fn supports_tp(&self) -> bool {
        false
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        _layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        ensure!(
            config.model_type == "glm5_next",
            "GLM loader received another model type"
        );
        let absmax = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let mut attn_layer_idx = 0usize;
        for layer_idx in 0..config.num_hidden_layers {
            let prefix = format!("model.language_model.layers.{layer_idx}");
            let input_norm = dense_auto(store, &format!("{prefix}.input_layernorm.weight"), gpu)?;
            let post_norm = dense_auto(
                store,
                &format!("{prefix}.post_attention_layernorm.weight"),
                gpu,
            )?;
            let ffn = if config.mlp_only_layers.contains(&layer_idx) {
                load_dense_ffn(
                    store,
                    &format!("{prefix}.mlp"),
                    config.intermediate_size,
                    config.hidden_size,
                    gpu,
                    absmax,
                    quantize,
                    stream,
                )?
            } else {
                load_moe(
                    store,
                    &format!("{prefix}.mlp"),
                    config,
                    gpu,
                    absmax,
                    quantize,
                    stream,
                )?
            };
            if config.layer_type(layer_idx) == LayerType::LinearAttention {
                let kda = load_glm5_kda(store, &prefix, config, gpu)?;
                let conv = kda.pack_conv1d(config, gpu)?;
                layers.push(Box::new(Glm5KdaLayer::new(
                    input_norm,
                    kda,
                    conv,
                    post_norm,
                    ffn,
                    load_hc(store, &prefix, layer_idx, config, gpu)?,
                    layer_idx,
                    config,
                    gpu,
                )?) as Box<dyn TransformerLayer>);
            } else {
                layers.push(load_dsa_layer(
                    store,
                    config,
                    gpu,
                    layer_idx,
                    attn_layer_idx,
                    ffn,
                )?);
                attn_layer_idx += 1;
            }
        }
        Ok(layers)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense_auto(store, "model.language_model.embed_tokens.weight", _gpu)
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense_auto(store, "model.language_model.norm.weight", gpu)
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense_auto(store, "lm_head.weight", _gpu)
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<crate::weight_map::MtpWeights>> {
        // Layer 45 is GLM NextN's distinct module and is not representable by
        // Atlas's legacy MTP body yet. Disable speculation explicitly.
        Ok(None)
    }

    fn load_vision_encoder(
        &self,
        _store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<VisionEncoder>> {
        if config.vision.is_some() {
            // GLM's visual tensors are intentionally not fed into
            // `VisionEncoder`: that implementation assumes Qwen's 2D patch
            // projection, learned positional table, LayerNorm blocks, and
            // simple merger. Returning `None` is paired with the request-time
            // fail-fast in prefill_a/vision.rs, so image content is never
            // silently discarded while text-only GLM requests remain valid.
            tracing::warn!(
                "GLM-5.3-Flash vision tower detected but not attached: Conv3D patching, \
                 2D RoPE/RMSNorm-QK attention, and the gated downsample merger need a \
                 GLM-specific runtime. Image/video requests will be rejected; text-only \
                 generation remains available."
            );
        }
        Ok(None)
    }
}
