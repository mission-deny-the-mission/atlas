// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3-Flash configuration parser.
//!
//! GLM stores the language-model contract under `text_config`, but several
//! fields use GLM-specific names. This parser translates that contract into
//! Atlas's canonical configuration without inventing architecture defaults.

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;

use super::super::{
    LayerType, ModelConfig, finalize_config, parse_quantization_config, parse_vision_config,
};

fn required<'a>(config: &'a Value, field: &str) -> Result<&'a Value> {
    config
        .get(field)
        .with_context(|| format!("glm5_next text_config missing required field `{field}`"))
}

fn required_usize(config: &Value, field: &str) -> Result<usize> {
    let value = required(config, field)?.as_u64().with_context(|| {
        format!("glm5_next text_config field `{field}` must be an unsigned integer")
    })?;
    usize::try_from(value)
        .with_context(|| format!("glm5_next text_config field `{field}` does not fit usize"))
}

fn required_f64(config: &Value, field: &str) -> Result<f64> {
    required(config, field)?
        .as_f64()
        .with_context(|| format!("glm5_next text_config field `{field}` must be a number"))
}

fn layer_types(config: &Value, count: usize) -> Result<Vec<LayerType>> {
    let types = required(config, "layer_types")?
        .as_array()
        .context("glm5_next text_config field `layer_types` must be an array")?;
    ensure!(
        types.len() == count,
        "glm5_next layer_types length ({}) does not match num_hidden_layers ({count})",
        types.len(),
    );
    types
        .iter()
        .enumerate()
        .map(|(index, value)| match value.as_str() {
            Some("linear_attention") => Ok(LayerType::LinearAttention),
            Some("deepseek_sparse_attention") | Some("full_attention") => {
                Ok(LayerType::FullAttention)
            }
            Some(other) => bail!("glm5_next layer_types[{index}] has unsupported value `{other}`"),
            None => bail!("glm5_next layer_types[{index}] must be a string"),
        })
        .collect()
}

fn dense_mlp_layers(config: &Value, count: usize) -> Result<Vec<usize>> {
    let types = required(config, "mlp_layer_types")?
        .as_array()
        .context("glm5_next text_config field `mlp_layer_types` must be an array")?;
    ensure!(
        types.len() == count,
        "glm5_next mlp_layer_types length ({}) does not match num_hidden_layers ({count})",
        types.len(),
    );
    types
        .iter()
        .enumerate()
        .filter_map(|(index, value)| match value.as_str() {
            Some("dense") => Some(Ok(index)),
            Some("sparse") => None,
            Some(other) => Some(Err(anyhow::anyhow!(
                "glm5_next mlp_layer_types[{index}] has unsupported value `{other}`"
            ))),
            None => Some(Err(anyhow::anyhow!(
                "glm5_next mlp_layer_types[{index}] must be a string"
            ))),
        })
        .collect()
}

fn indexer_types(config: &Value, count: usize) -> Result<Vec<String>> {
    let types = required(config, "indexer_types")?
        .as_array()
        .context("glm5_next text_config field `indexer_types` must be an array")?;
    ensure!(
        types.len() == count,
        "glm5_next indexer_types length ({}) does not match num_hidden_layers ({count})",
        types.len(),
    );
    types
        .iter()
        .enumerate()
        .map(|(index, value)| match value.as_str() {
            Some("full") => Ok("full".to_owned()),
            Some("shared") => bail!("glm5_next indexer_types[{index}] = shared is not supported"),
            Some(other) => {
                bail!("glm5_next indexer_types[{index}] has unsupported value `{other}`")
            }
            None => bail!("glm5_next indexer_types[{index}] must be a string"),
        })
        .collect()
}

/// Parse the nested GLM-5.3-Flash language-model configuration.
pub(crate) fn parse_glm5_next(raw: &Value) -> Result<ModelConfig> {
    let text = raw
        .get("text_config")
        .context("glm5_next config missing text_config")?;
    ensure!(
        text.get("model_type").and_then(Value::as_str) == Some("glm5_next_text"),
        "glm5_next text_config.model_type must be `glm5_next_text`"
    );

    let mut deserializable = text.clone();
    let object = deserializable
        .as_object_mut()
        .context("glm5_next text_config must be an object")?;
    // Atlas's legacy scalar EOS field is only a fallback. The server reads the
    // complete official EOS array from generation_config.json when it exists.
    if let Some(tokens) = text.get("eos_token_id").and_then(Value::as_array) {
        let primary = tokens
            .first()
            .and_then(Value::as_u64)
            .context("glm5_next eos_token_id must contain an integer")?;
        object.insert("eos_token_id".into(), Value::from(primary));
    }
    // GLM's DSA spelling is intentionally not an Atlas `LayerType` serde
    // variant; translate it with the strict mapper below instead of teaching
    // generic deserialization an architecture-specific alias.
    object.remove("layer_types");

    let mut config: ModelConfig = serde_json::from_value(deserializable)
        .context("failed to deserialize glm5_next text_config")?;
    let layers = required_usize(text, "num_hidden_layers")?;
    let linear = required(text, "linear_attn_config")?;

    config.model_type = "glm5_next".into();
    config.eos_token_ids = required(text, "eos_token_id")?
        .as_array()
        .context("glm5_next eos_token_id must be an array")?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .context("glm5_next eos_token_id must contain integers")
                .and_then(|id| u32::try_from(id).context("glm5_next eos token does not fit u32"))
        })
        .collect::<Result<Vec<_>>>()?;
    config.hidden_size = required_usize(text, "hidden_size")?;
    config.num_hidden_layers = layers;
    config.intermediate_size = required_usize(text, "intermediate_size")?;
    config.vocab_size = required_usize(text, "vocab_size")?;
    config.num_attention_heads = required_usize(text, "num_attention_heads")?;
    config.num_key_value_heads = required_usize(text, "num_key_value_heads")?;
    config.head_dim = required_usize(text, "qk_head_dim")?;
    config.qk_nope_head_dim = required_usize(text, "qk_nope_head_dim")?;
    config.qk_rope_head_dim = required_usize(text, "qk_rope_head_dim")?;
    config.v_head_dim = required_usize(text, "v_head_dim")?;
    config.q_lora_rank = required_usize(text, "q_lora_rank")?;
    config.kv_lora_rank = required_usize(text, "kv_lora_rank")?;
    config.num_experts = required_usize(text, "n_routed_experts")?;
    config.num_experts_per_tok = required_usize(text, "num_experts_per_tok")?;
    config.moe_intermediate_size = required_usize(text, "moe_intermediate_size")?;
    config.shared_expert_intermediate_size = required_usize(text, "n_shared_experts")?
        .checked_mul(config.moe_intermediate_size)
        .context("glm5_next shared expert width overflows usize")?;
    config.routed_scaling_factor = required_f64(text, "routed_scaling_factor")?;
    config.norm_topk_prob = required(text, "norm_topk_prob")?
        .as_bool()
        .context("glm5_next text_config field `norm_topk_prob` must be a boolean")?;
    config.scoring_func = required(text, "scoring_func")?
        .as_str()
        .context("glm5_next text_config field `scoring_func` must be a string")?
        .to_owned();
    config.use_routing_bias = required(text, "topk_method")?
        .as_str()
        .context("glm5_next text_config field `topk_method` must be a string")?
        == "noaux_tc";
    config.layer_types = layer_types(text, layers)?;
    config.mlp_only_layers = dense_mlp_layers(text, layers)?;
    config.linear_num_key_heads = required_usize(linear, "num_heads")?;
    config.linear_num_value_heads = config.linear_num_key_heads;
    config.linear_key_head_dim = required_usize(linear, "head_dim")?;
    config.linear_value_head_dim = config.linear_key_head_dim;
    config.linear_conv_kernel_dim = required_usize(linear, "short_conv_kernel_size")?;
    config.linear_gate_lower_bound = match required(linear, "gate_lower_bound")? {
        Value::Null => None,
        value => {
            let value = value.as_f64().context(
                "glm5_next linear_attn_config field `gate_lower_bound` must be a number or null",
            )?;
            ensure!(
                value.is_finite() && value >= f32::MIN as f64 && value <= f32::MAX as f64,
                "glm5_next linear_attn_config field `gate_lower_bound` is outside f32 range",
            );
            Some(value as f32)
        }
    };
    config.hc_mult = required_usize(text, "hc_mult")?;
    config.hc_sinkhorn_iters = required_usize(text, "hc_sinkhorn_iters")?;
    config.hc_eps = required_f64(text, "hc_eps")? as f32;
    config.index_n_heads = required_usize(text, "index_n_heads")?;
    config.index_head_dim = required_usize(text, "index_head_dim")?;
    config.index_topk = required_usize(text, "index_topk")?;
    config.index_kpool = required_usize(text, "index_kpool")?;
    config.index_kpool_compress = required(text, "index_kpool_compress")?
        .as_bool()
        .context("glm5_next text_config field `index_kpool_compress` must be a boolean")?;
    ensure!(
        config.index_kpool_compress,
        "glm5_next requires index_kpool_compress=true"
    );
    config.index_kpool_always_select_tail = required(text, "index_kpool_always_select_tail")?
        .as_bool()
        .context(
            "glm5_next text_config field `index_kpool_always_select_tail` must be a boolean",
        )?;
    config.indexer_types = indexer_types(text, layers)?;
    config.mtp_num_hidden_layers = required_usize(text, "num_nextn_predict_layers")?;
    config.num_mtp_modules = config.mtp_num_hidden_layers;
    config.mtp_transformer_layers = 1;
    config.attn_gated = false;
    config.nested_config = true;
    config.weight_prefix = "model.language_model".into();
    config.vision = parse_vision_config(raw);
    config.quantization_config = parse_quantization_config(raw);

    ensure!(
        config.num_key_value_heads == config.num_attention_heads,
        "glm5_next requires MHA: num_key_value_heads ({}) != num_attention_heads ({})",
        config.num_key_value_heads,
        config.num_attention_heads,
    );
    ensure!(
        config.qk_rope_head_dim == 0,
        "glm5_next support requires NoPE DSA, got qk_rope_head_dim={}",
        config.qk_rope_head_dim,
    );
    ensure!(
        config.scoring_func == "sigmoid",
        "glm5_next support requires sigmoid expert routing, got `{}`",
        config.scoring_func,
    );
    ensure!(
        config.index_kpool > 0,
        "glm5_next index_kpool must be greater than zero",
    );
    ensure!(
        config.index_topk.is_multiple_of(config.index_kpool),
        "glm5_next index_topk ({}) must be divisible by index_kpool ({})",
        config.index_topk,
        config.index_kpool,
    );
    finalize_config(&mut config, raw)?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_glm53_flash_architecture_without_fallbacks() {
        let layer_types = (0..45)
            .map(|layer| {
                if layer % 4 == 3 {
                    "deepseek_sparse_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect::<Vec<_>>();
        let mlp_layer_types = (0..45)
            .map(|layer| if layer < 3 { "dense" } else { "sparse" })
            .collect::<Vec<_>>();
        let raw = serde_json::json!({
            "model_type": "glm5_next",
            "text_config": {
                "model_type": "glm5_next_text",
                "hidden_size": 4096,
                "num_hidden_layers": 45,
                "intermediate_size": 12288,
                "vocab_size": 154880,
                "num_attention_heads": 64,
                "num_key_value_heads": 64,
                "head_dim": 0,
                "qk_head_dim": 256,
                "qk_nope_head_dim": 256,
                "qk_rope_head_dim": 0,
                "v_head_dim": 256,
                "q_lora_rank": 1536,
                "kv_lora_rank": 512,
                "n_routed_experts": 288,
                "n_shared_experts": 1,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 2048,
                "routed_scaling_factor": 2.5,
                "norm_topk_prob": true,
                "scoring_func": "sigmoid",
                "topk_method": "noaux_tc",
                "layer_types": layer_types,
                "mlp_layer_types": mlp_layer_types,
                "linear_attn_config": {
                    "num_heads": 64,
                    "head_dim": 128,
                    "short_conv_kernel_size": 4,
                    "gate_lower_bound": -5.0
                },
                "hc_mult": 4,
                "hc_sinkhorn_iters": 20,
                "hc_eps": 0.000001,
                "index_n_heads": 32,
                "index_head_dim": 128,
                "index_topk": 2048,
                "index_kpool": 4,
                "index_kpool_compress": true,
                "index_kpool_always_select_tail": true,
                "indexer_types": vec!["full"; 45],
                "num_nextn_predict_layers": 1,
                "eos_token_id": [154820, 154827, 154829]
            },
            "vision_config": {
                "depth": 24,
                "hidden_size": 1024,
                "num_heads": 16,
                "patch_size": 14,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "intermediate_size": 4096,
                "out_hidden_size": 4096
            }
        });

        let config = parse_glm5_next(&raw).expect("GLM-5.3 Flash config parses");
        assert_eq!(config.model_type, "glm5_next");
        assert_eq!(config.weight_prefix, "model.language_model");
        assert_eq!(config.num_attention_layers(), 11);
        assert_eq!(config.num_ssm_layers(), 34);
        assert_eq!(config.mlp_only_layers, vec![0, 1, 2]);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.linear_key_head_dim, 128);
        assert_eq!(config.shared_expert_intermediate_size, 2048);
        assert_eq!(config.eos_token_id, 154820);
        assert_eq!(config.eos_token_ids, vec![154820, 154827, 154829]);
        assert_eq!(config.linear_gate_lower_bound, Some(-5.0));
        assert_eq!(config.index_kpool, 4);
        assert!(config.use_routing_bias);
        assert!(config.vision.is_some());
    }

    #[test]
    fn rejects_zero_index_pool_before_validating_divisibility() {
        let mut raw = serde_json::json!({
            "model_type": "glm5_next",
            "text_config": {
                "model_type": "glm5_next_text",
                "hidden_size": 4096,
                "num_hidden_layers": 1,
                "intermediate_size": 12288,
                "vocab_size": 154880,
                "num_attention_heads": 64,
                "num_key_value_heads": 64,
                "qk_head_dim": 256,
                "qk_nope_head_dim": 256,
                "qk_rope_head_dim": 0,
                "v_head_dim": 256,
                "q_lora_rank": 1536,
                "kv_lora_rank": 512,
                "n_routed_experts": 288,
                "n_shared_experts": 1,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 2048,
                "routed_scaling_factor": 2.5,
                "norm_topk_prob": true,
                "scoring_func": "sigmoid",
                "topk_method": "noaux_tc",
                "layer_types": ["linear_attention"],
                "mlp_layer_types": ["dense"],
                "linear_attn_config": {
                    "num_heads": 64,
                    "head_dim": 128,
                    "short_conv_kernel_size": 4,
                    "gate_lower_bound": null
                },
                "hc_mult": 4,
                "hc_sinkhorn_iters": 20,
                "hc_eps": 0.000001,
                "index_n_heads": 32,
                "index_head_dim": 128,
                "index_topk": 2048,
                "index_kpool": 0,
                "index_kpool_compress": true,
                "index_kpool_always_select_tail": true,
                "indexer_types": ["full"],
                "num_nextn_predict_layers": 1,
                "eos_token_id": [154820]
            }
        });
        let err = parse_glm5_next(&raw).expect_err("zero k-pool must fail");
        assert!(
            err.to_string()
                .contains("index_kpool must be greater than zero")
        );

        raw["text_config"]["index_kpool"] = serde_json::json!(4);
        assert!(parse_glm5_next(&raw).is_ok());

        raw["text_config"]["indexer_types"] = serde_json::json!(["shared"]);
        let err = parse_glm5_next(&raw).expect_err("shared indexer must fail");
        assert!(err.to_string().contains("shared is not supported"));
    }
}
