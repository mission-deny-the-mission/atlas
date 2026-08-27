// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3-Flash capabilities that must be decided before allocating weights.

use anyhow::{Result, bail};
use atlas_core::config::ModelConfig;

/// Reject speculative decoding for GLM's NextN module before the main model
/// weights are loaded. NextN is a DSA+MoE decoder block, not Atlas's Qwen MTP
/// head, so treating its tensors as the legacy MTP format would produce
/// unverified, incorrect draft tokens.
pub(super) fn ensure_supported_speculation(
    config: &ModelConfig,
    use_speculative: bool,
) -> Result<()> {
    if config.model_type == "glm5_next" && use_speculative && config.num_mtp_modules > 0 {
        bail!(
            "GLM-5.3-Flash NextN speculative decoding is not supported: its DSA+MoE \
             predictor is not compatible with Atlas's MTP runtime. Re-run with \
             --speculative disabled."
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::ensure_supported_speculation;
    use atlas_core::config::ModelConfig;

    #[test]
    fn glm_nextn_rejects_requested_speculation() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "glm5_next".into();
        config.num_mtp_modules = 1;

        let err = ensure_supported_speculation(&config, true).unwrap_err();
        assert!(err.to_string().contains("NextN speculative decoding"));
        assert!(ensure_supported_speculation(&config, false).is_ok());
    }

    #[test]
    fn non_glm_models_keep_existing_speculation_path() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.num_mtp_modules = 1;
        assert!(ensure_supported_speculation(&config, true).is_ok());
    }
}
