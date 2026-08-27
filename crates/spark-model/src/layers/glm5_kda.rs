// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3 Flash Kimi Delta Attention transformer block.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use crate::layer::{ForwardContext, LayerState, SsmLayerState, TransformerLayer};
use crate::layers::qwen3_attention::HcWeights;
use crate::layers::{FfnComponent, ops};
use crate::weight_map::{DenseWeight, Glm5KdaWeights};

/// A GLM KDA block, including its manifold-constrained hyper-connection.
///
/// It owns only immutable weights and dispatch handles; the recurrent matrix
/// and the causal-convolution tail live in [`SsmLayerState`], allocated by the
/// model's existing recurrent-state pool.
pub struct Glm5KdaLayer {
    input_norm: DenseWeight,
    kda: Glm5KdaWeights,
    conv_weight: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
    rms_norm_k: KernelHandle,
    dense_gemv_k: KernelHandle,
    conv1d_k: KernelHandle,
    kda_recurrent_k: KernelHandle,
    kda_gated_norm_k: KernelHandle,
    hc: HcWeights,
    layer_idx: usize,
    hc_expand_k: KernelHandle,
    hc_pre_k: KernelHandle,
    hc_post_k: KernelHandle,
    hc_mean_k: KernelHandle,
    h_state_bytes: usize,
    conv_state_bytes: usize,
}

impl Glm5KdaLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_norm: DenseWeight,
        kda: Glm5KdaWeights,
        conv_weight: DenseWeight,
        post_attn_norm: DenseWeight,
        ffn: FfnComponent,
        hc: HcWeights,
        layer_idx: usize,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        ensure!(
            config.model_type == "glm5_next",
            "Glm5KdaLayer requires model_type=glm5_next",
        );
        ensure!(
            config.linear_num_key_heads == config.linear_num_value_heads,
            "GLM KDA requires matching key/value head counts",
        );
        ensure!(
            config.linear_key_head_dim == 128 && config.linear_value_head_dim == 128,
            "GLM KDA kernel is specialized for 128-wide heads",
        );
        ensure!(
            hc.hc_mult == config.hc_mult && hc.hc_mult > 0,
            "GLM KDA requires a configured mHC highway",
        );
        Ok(Self {
            input_norm,
            kda,
            conv_weight,
            post_attn_norm,
            ffn,
            rms_norm_k: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            conv1d_k: gpu.kernel("causal_conv1d", "causal_conv1d_update")?,
            kda_recurrent_k: gpu.kernel("kda", "kda_recurrent_decode")?,
            kda_gated_norm_k: gpu.kernel("kda", "kda_gated_rms_norm")?,
            hc,
            layer_idx,
            hc_expand_k: gpu.kernel("hyper_connection", "hc_expand")?,
            hc_pre_k: gpu.kernel("hyper_connection", "hc_pre")?,
            hc_post_k: gpu.kernel("hyper_connection", "hc_post")?,
            hc_mean_k: gpu.kernel("hyper_connection", "hc_mean")?,
            h_state_bytes: config.ssm_h_state_bytes(),
            conv_state_bytes: config.ssm_conv_state_bytes(),
        })
    }

    fn attention_forward(
        &self,
        normed: DevicePtr,
        state: &mut SsmLayerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let config = ctx.config;
        let hidden_size = config.hidden_size as u32;
        let heads = config.linear_num_key_heads as u32;
        let head_dim = config.linear_key_head_dim as u32;
        let width = (heads * head_dim) as usize;
        let bf16 = std::mem::size_of::<u16>();
        let eps = config.rms_norm_eps as f32;

        // `ssm_qkvz` is sized from the largest four-way QKVZ contract. KDA
        // uses its first three regions for Q/K/V and its fourth for forget/gate.
        let qkv_and_gate = ctx.buffers.ssm_qkvz();
        let q = qkv_and_gate;
        let k = q.offset(width * bf16);
        let v = k.offset(width * bf16);
        let forget_or_gate = v.offset(width * bf16);
        for (weight, output) in [
            (&self.kda.q_proj, q),
            (&self.kda.k_proj, k),
            (&self.kda.v_proj, v),
        ] {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                normed,
                weight,
                output,
                width as u32,
                hidden_size,
                stream,
            )?;
        }

        // f_b(f_a(x)) supplies a log-decay for each key component.  The
        // temporary f_a and b-projection fit in the existing SSM BA buffer.
        let small = ctx.buffers.ssm_ba();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            normed,
            &self.kda.f_a_proj,
            small,
            head_dim,
            hidden_size,
            stream,
        )?;
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            small,
            &self.kda.f_b_proj,
            forget_or_gate,
            width as u32,
            head_dim,
            stream,
        )?;
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            normed,
            &self.kda.b_proj,
            small,
            heads,
            hidden_size,
            stream,
        )?;

        let conv_out = ctx.buffers.ssm_deinterleaved();
        ops::conv1d_update(
            ctx.gpu,
            self.conv1d_k,
            state.conv_state,
            qkv_and_gate,
            &self.conv_weight,
            conv_out,
            (width * 3) as u32,
            config.linear_conv_kernel_dim as u32,
            1,
            stream,
        )?;
        ops::kda_recurrent_decode(
            ctx.gpu,
            self.kda_recurrent_k,
            state.h_state,
            conv_out,
            forget_or_gate,
            small,
            self.kda.a_log.weight,
            self.kda.dt_bias.weight,
            qkv_and_gate,
            1,
            heads,
            head_dim,
            config.linear_gate_lower_bound,
            stream,
        )?;

        // g_b(g_a(x)) is a separate output gate and must not be folded into
        // the KDA decay path.  Reuse the now-dead small/forget regions.
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            normed,
            &self.kda.g_a_proj,
            small,
            head_dim,
            hidden_size,
            stream,
        )?;
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            small,
            &self.kda.g_b_proj,
            forget_or_gate,
            width as u32,
            head_dim,
            stream,
        )?;
        ops::kda_gated_rms_norm(
            ctx.gpu,
            self.kda_gated_norm_k,
            qkv_and_gate,
            forget_or_gate,
            self.kda.o_norm.weight,
            conv_out,
            1,
            heads,
            head_dim,
            eps,
            stream,
        )?;
        let attn_out = ctx.buffers.attn_output();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            conv_out,
            &self.kda.o_proj,
            attn_out,
            hidden_size,
            width as u32,
            stream,
        )?;

        Ok(attn_out)
    }

    fn forward_one_hc(
        &self,
        hidden: DevicePtr,
        state: &mut SsmLayerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_one_hc_row(hidden, state, ctx, stream, 0)
    }

    fn forward_one_hc_row(
        &self,
        hidden: DevicePtr,
        state: &mut SsmLayerState,
        ctx: &ForwardContext,
        stream: u64,
        row: usize,
    ) -> Result<()> {
        let hidden_size = ctx.config.hidden_size as u32;
        let hc_mult = self.hc.hc_mult as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let row_bytes = hc_mult as usize * hidden_size as usize * 2;
        let streams = ctx.buffers.hc_streams().offset(row * row_bytes);
        let post = ctx.buffers.hc_post().offset(row * row_bytes);
        let comb = ctx.buffers.hc_comb().offset(row * row_bytes);

        if self.layer_idx == 0 {
            ops::hc_expand(
                ctx.gpu,
                self.hc_expand_k,
                hidden,
                streams,
                1,
                hidden_size,
                hc_mult,
                stream,
            )?;
        }

        ops::hc_pre(
            ctx.gpu,
            self.hc_pre_k,
            streams,
            self.hc.attn.hc_fn,
            self.hc.attn.hc_scale,
            self.hc.attn.hc_base,
            hidden,
            post,
            comb,
            1,
            hidden_size,
            hc_mult,
            self.hc.sinkhorn_iters as u32,
            eps,
            self.hc.hc_eps,
            stream,
        )?;
        let normed = ctx.buffers.norm_output();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            hidden,
            &self.input_norm,
            normed,
            1,
            hidden_size,
            eps,
            stream,
        )?;
        let attn_out = self.attention_forward(normed, state, ctx, stream)?;
        ops::hc_post(
            ctx.gpu,
            self.hc_post_k,
            attn_out,
            streams,
            post,
            comb,
            streams,
            1,
            hidden_size,
            hc_mult,
            stream,
        )?;

        ops::hc_pre(
            ctx.gpu,
            self.hc_pre_k,
            streams,
            self.hc.ffn.hc_fn,
            self.hc.ffn.hc_scale,
            self.hc.ffn.hc_base,
            hidden,
            post,
            comb,
            1,
            hidden_size,
            hc_mult,
            self.hc.sinkhorn_iters as u32,
            eps,
            self.hc.hc_eps,
            stream,
        )?;
        let ffn_input = ctx.buffers.norm_output();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            hidden,
            &self.post_attn_norm,
            ffn_input,
            1,
            hidden_size,
            eps,
            stream,
        )?;
        let ffn_out = self.ffn.forward(ffn_input, ctx, stream)?;
        ops::hc_post(
            ctx.gpu,
            self.hc_post_k,
            ffn_out,
            streams,
            post,
            comb,
            streams,
            1,
            hidden_size,
            hc_mult,
            stream,
        )?;

        if self.layer_idx + 1 == ctx.config.num_hidden_layers {
            ops::hc_mean(
                ctx.gpu,
                self.hc_mean_k,
                streams,
                hidden,
                1,
                hidden_size,
                hc_mult,
                stream,
            )?;
        }
        Ok(())
    }
}

impl TransformerLayer for Glm5KdaLayer {
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    #[allow(clippy::too_many_arguments)]
    fn decode(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("GLM KDA requires SsmLayerState"))?;
        self.forward_one_hc(hidden, state, ctx, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_multi_seq<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        _kv_cache: &mut PagedKvCache,
        _seq_lens: &[usize],
        _block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ensure!(states.len() >= num_seqs, "GLM KDA state count is too small");
        let row_bytes = ctx.config.hidden_size * std::mem::size_of::<u16>();
        for (row, state) in states.iter_mut().take(num_seqs).enumerate() {
            let state = state
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("GLM KDA requires SsmLayerState"))?;
            self.forward_one_hc_row(hidden.offset(row * row_bytes), state, ctx, stream, row)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len_start: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("GLM KDA requires SsmLayerState"))?;
        let row_bytes = ctx.config.hidden_size * std::mem::size_of::<u16>();
        for token in 0..num_tokens {
            self.forward_one_hc_row(hidden.offset(token * row_bytes), state, ctx, stream, token)?;
        }
        Ok(())
    }

    fn is_ssm_layer(&self) -> bool {
        true
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        let h_state = gpu.alloc(self.h_state_bytes)?;
        gpu.memset(h_state, 0, self.h_state_bytes)?;
        let conv_state = gpu.alloc(self.conv_state_bytes)?;
        gpu.memset(conv_state, 0, self.conv_state_bytes)?;
        Ok(Box::new(SsmLayerState {
            h_state,
            conv_state,
            h_state_checkpoint: None,
            conv_state_checkpoint: None,
            h_state_intermediates: Vec::new(),
            conv_state_intermediates: Vec::new(),
            h_is_f16: false,
            h_prefill_stage: None,
        }))
    }
}
