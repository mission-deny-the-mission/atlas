// SPDX-License-Identifier: AGPL-3.0-only

//! Absorbed-MLA decode path of `attention_forward`. Single-token GEMV
//! chain (Q latent → norm → expand → absorbed-Q via batched GEMV →
//! Q_rope scatter → K_latent → K_rope+RoPE → cache assemble + write →
//! paged decode → V extract → O proj). Returns early — caller short-
//! circuits on the result. Extracted from `attention_forward.rs` to
//! keep that file under 500 LoC.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

#[allow(clippy::too_many_arguments, dead_code)]
pub(in crate::layers::qwen3_attention) struct DecodeMlaArgs {
    pub normed: DevicePtr,
    pub q_out: DevicePtr,
    pub k_out: DevicePtr,
    pub v_out: DevicePtr,
    pub q_dim: u32,
    pub h: u32,
    pub nq: u32,
    pub hd: u32,
    pub eps: f32,
    pub bs: usize,
    pub stream: u64,
    /// 4b inc-3: absolute position of this decode token (= seq_len − 1), host-side.
    /// `Some` only on the standard single-sequence decode path (drives the
    /// compressed-pool append); `None` on the batched / MTP-verify path, where
    /// append is skipped (frozen inc-2 pool) to avoid a shared per-layer counter.
    pub pos: Option<u32>,
}

impl Qwen3AttentionLayer {
    /// Run the absorbed MLA decode chain, returning the O-projection
    /// output (`ctx.buffers.qkv_output()`).
    pub(super) fn attention_forward_mla(
        &self,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        args: &DecodeMlaArgs,
    ) -> Result<DevicePtr> {
        let DecodeMlaArgs {
            normed,
            q_out,
            k_out,
            v_out,
            q_dim,
            h,
            nq,
            hd,
            eps,
            bs,
            stream,
            pos,
        } = *args;
        let mla = self
            .mla
            .as_ref()
            .expect("attention_forward_mla called without MLA config");
        // Keep sparse IndexPool opt-in during text-only bring-up. Dense MLA
        // is the correctness baseline and avoids consuming an IndexPool
        // sidecar that dense prefill did not populate.
        let glm_indexer = mla
            .glm_indexer
            .as_ref()
            .filter(|_| std::env::var_os("ATLAS_GLM_SPARSE_ATTENTION").is_some());
        let meta = ctx
            .attn_metadata
            .expect("MLA decode requires pre-uploaded metadata");

        let q_lora = mla.q_lora_rank as u32;
        let kv_lora = mla.kv_lora_rank as u32;
        let mla_nope = mla.nope as u32;
        let mla_v_dim = mla.v_dim as u32;
        let mla_rope = mla.rope as u32;
        let profile = ctx.profile;
        macro_rules! prof {
            ($label:expr, $body:expr) => {{
                if profile {
                    let _t = std::time::Instant::now();
                    let _r = $body;
                    ctx.gpu.synchronize(stream)?;
                    tracing::info!("    MLA {}: {:.0}μs", $label, _t.elapsed().as_micros());
                    _r
                } else {
                    $body
                }
            }};
        }

        // Step 1: Q latent → norm → expand
        let q_latent = ctx.buffers.ssm_ba();
        prof!("wq_a", {
            if let Some(ref wqa_nvfp4) = mla.wq_a_nvfp4 {
                self.nvfp4_decode_gemv(
                    ctx.gpu,
                    ctx.levers.gemv_sw,
                    normed,
                    wqa_nvfp4,
                    q_latent,
                    q_lora,
                    h,
                    stream,
                )
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed,
                    &mla.wq_a,
                    q_latent,
                    q_lora,
                    h,
                    stream,
                )
            }
        })?;
        prof!("q_norm", {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                q_latent,
                &mla.q_a_norm,
                q_latent,
                1,
                q_lora,
                eps,
                stream,
            )
        })?;
        let q_full = ctx.buffers.ssm_deinterleaved();
        prof!("wq_b", {
            if let Some(ref wqb_nvfp4) = mla.wq_b_nvfp4 {
                self.nvfp4_decode_gemv(
                    ctx.gpu,
                    ctx.levers.gemv_sw,
                    q_latent,
                    wqb_nvfp4,
                    q_full,
                    q_dim,
                    q_lora,
                    stream,
                )
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    q_latent,
                    &mla.wq_b,
                    q_full,
                    q_dim,
                    q_lora,
                    stream,
                )
            }
        })?;

        // Step 2: Q_absorbed via batched GEMV
        let mla_cache_dim = kv_lora + mla_rope;
        let q_absorbed_buf = ctx.buffers.expert_up_out();
        prof!("q_absorb", {
            if self.mla_batched_gemv_k.0 != 0 {
                ops::mla_batched_gemv(
                    ctx.gpu,
                    self.mla_batched_gemv_k,
                    q_full,
                    mla.w_uk_t.weight,
                    q_absorbed_buf,
                    kv_lora,
                    mla_nope,
                    nq,
                    hd,
                    mla_cache_dim,
                    stream,
                )
            } else {
                for head_idx in 0..nq as usize {
                    let q_nope_ptr = q_full.offset(head_idx * hd as usize * 2);
                    let q_abs_dst = q_absorbed_buf.offset(head_idx * mla_cache_dim as usize * 2);
                    let w_uk_head = mla
                        .w_uk_t
                        .weight
                        .offset(head_idx * mla.nope * mla.kv_lora_rank * 2);
                    let w_uk_dense = crate::weight_map::DenseWeight { weight: w_uk_head };
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        q_nope_ptr,
                        &w_uk_dense,
                        q_abs_dst,
                        kv_lora,
                        mla_nope,
                        stream,
                    )?;
                }
                Ok(())
            }
        })?;

        // Q_rope scatter
        let q_rope_direct = ctx.buffers.ssm_conv_out_f32();
        prof!("q_rope_scatter", {
            if self.mla_q_rope_scatter_k.0 != 0 {
                ops::mla_q_rope_scatter(
                    ctx.gpu,
                    self.mla_q_rope_scatter_k,
                    q_full,
                    q_absorbed_buf,
                    q_rope_direct,
                    nq,
                    hd,
                    mla_nope,
                    mla_rope,
                    kv_lora,
                    mla_cache_dim,
                    stream,
                )
            } else {
                for head_idx in 0..nq as usize {
                    let src = q_full.offset((head_idx * hd as usize + mla.nope) * 2);
                    ctx.gpu.copy_d2d_async(
                        src,
                        q_rope_direct.offset(head_idx * mla.rope * 2),
                        mla.rope * 2,
                        stream,
                    )?;
                    ctx.gpu.copy_d2d_async(
                        src,
                        q_absorbed_buf
                            .offset((head_idx * mla_cache_dim as usize + mla.kv_lora_rank) * 2),
                        mla.rope * 2,
                        stream,
                    )?;
                }
                Ok(())
            }
        })?;

        // GLM DSA IndexPool query projections use the post-QA-norm latent and
        // the same normalized layer input as MLA. GLM-5.3 has no RoPE, so the
        // regular Q buffer is dead after absorption and can stage this query.
        if let Some(indexer) = glm_indexer {
            prof!("glm_index_query", {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    q_latent,
                    &indexer.wq_b,
                    q_full,
                    (ctx.config.index_n_heads * ctx.config.index_head_dim) as u32,
                    q_lora,
                    stream,
                )?;
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed,
                    &indexer.weights_proj,
                    q_full.offset(ctx.config.index_n_heads * ctx.config.index_head_dim * 2),
                    ctx.config.index_n_heads as u32,
                    h,
                    stream,
                )
            })?;
        }

        // Step 3: KV latent → norm
        let kv_latent = ctx.buffers.expert_gate_out();
        prof!("wkv_a+norm", {
            if let Some(ref wkva_nvfp4) = mla.wkv_a_nvfp4 {
                self.nvfp4_decode_gemv(
                    ctx.gpu,
                    ctx.levers.gemv_sw,
                    normed,
                    wkva_nvfp4,
                    kv_latent,
                    kv_lora,
                    h,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed,
                    &mla.wkv_a,
                    kv_latent,
                    kv_lora,
                    h,
                    stream,
                )?;
            }
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                kv_latent,
                &mla.kv_a_norm,
                kv_latent,
                1,
                kv_lora,
                eps,
                stream,
            )
        })?;

        // Step 4: K_rope + RoPE + writeback
        let k_rope_single = ctx.buffers.ssm_ba();
        prof!("k_rope+RoPE+wb", {
            if mla_rope > 0 {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed,
                    &mla.wkv_a_rope,
                    k_rope_single,
                    mla_rope,
                    h,
                    stream,
                )?;
                ops::rope_yarn(
                    ctx.gpu,
                    self.rope_yarn_k,
                    q_rope_direct,
                    k_rope_single,
                    meta.positions,
                    1,
                    nq,
                    1,
                    mla_rope,
                    mla_rope,
                    mla.yarn_inv_freq,
                    ctx.config.rope_theta as f32,
                    stream,
                )?;
                if self.mla_q_rope_writeback_k.0 != 0 {
                    ops::mla_q_rope_writeback(
                        ctx.gpu,
                        self.mla_q_rope_writeback_k,
                        q_rope_direct,
                        q_absorbed_buf,
                        nq,
                        mla_rope,
                        kv_lora,
                        mla_cache_dim,
                        stream,
                    )
                } else {
                    for head_idx in 0..nq as usize {
                        let src = q_rope_direct.offset(head_idx * mla.rope * 2);
                        let dst = q_absorbed_buf
                            .offset((head_idx * mla_cache_dim as usize + mla.kv_lora_rank) * 2);
                        ctx.gpu.copy_d2d_async(src, dst, mla.rope * 2, stream)?;
                    }
                    Ok(())
                }
            } else {
                Ok(())
            }
        })?;

        // Step 6: Cache assemble + write
        let k_cache_entry = k_out;
        let v_cache_entry = v_out;
        prof!("cache_asm+write", {
            if self.mla_cache_assemble_k.0 != 0 {
                ops::mla_cache_assemble(
                    ctx.gpu,
                    self.mla_cache_assemble_k,
                    kv_latent,
                    k_rope_single,
                    k_cache_entry,
                    v_cache_entry,
                    kv_lora,
                    mla_rope,
                    mla_cache_dim,
                    stream,
                )?;
            } else {
                ctx.gpu
                    .copy_d2d_async(kv_latent, k_cache_entry, mla.kv_lora_rank * 2, stream)?;
                ctx.gpu.copy_d2d_async(
                    k_rope_single,
                    k_cache_entry.offset(mla.kv_lora_rank * 2),
                    mla.rope * 2,
                    stream,
                )?;
                ctx.gpu
                    .copy_d2d_async(kv_latent, v_cache_entry, mla.kv_lora_rank * 2, stream)?;
                ctx.gpu.memset_async(
                    v_cache_entry.offset(mla.kv_lora_rank * 2),
                    0,
                    mla.rope * 2,
                    stream,
                )?;
            }
            self.write_kv_cache(
                ctx.gpu,
                k_cache_entry,
                v_cache_entry,
                kv_cache,
                meta.slot,
                1,
                1,
                mla_cache_dim,
                bs as u32,
                mla_cache_dim,
                mla_cache_dim,
                stream,
                ctx.graph_capture,
            )
        })?;

        if let Some(indexer) = glm_indexer {
            let raw_pool = kv_cache
                .index_raw_pool_ptr(self.attn_layer_idx)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "GLM IndexPool sidecar missing for attention layer {}",
                        self.attn_layer_idx
                    )
                })?;
            let raw_stride = kv_cache
                .index_raw_block_stride(self.attn_layer_idx)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "GLM IndexPool raw stride missing for attention layer {}",
                        self.attn_layer_idx
                    )
                })?;
            // q_out is unused by absorbed MLA. Use its first two 128-wide rows
            // as temporary K and per-channel compression-gate outputs.
            prof!("glm_index_cache", {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed,
                    &indexer.wk,
                    q_out,
                    ctx.config.index_head_dim as u32,
                    h,
                    stream,
                )?;
                ops::index_key_layer_norm(
                    ctx.gpu,
                    self.glm_index_norm_k,
                    q_out,
                    indexer.k_norm_weight.weight,
                    indexer.k_norm_bias.weight,
                    q_out,
                    ctx.config.index_head_dim as u32,
                    1.0e-6,
                    stream,
                )?;
                let gate = q_out.offset(ctx.config.index_head_dim * 2);
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed,
                    &indexer.kpool_compress_gate,
                    gate,
                    ctx.config.index_head_dim as u32,
                    h,
                    stream,
                )?;
                ops::index_pool_write_raw(
                    ctx.gpu,
                    self.glm_index_write_k,
                    q_out,
                    gate,
                    meta.slot,
                    raw_pool,
                    1,
                    ctx.config.index_head_dim as u32,
                    bs as u32,
                    u32::try_from(raw_stride)
                        .map_err(|_| anyhow::anyhow!("GLM IndexPool raw stride exceeds u32"))?,
                    stream,
                )
            })?;
        }

        // Step 8: Paged decode attention
        let attn_out = ctx.buffers.attn_output();
        let inv_sqrt_d = self.effective_attn_scale(hd);
        if let Some(indexer) = glm_indexer {
            let pos = pos.ok_or_else(|| {
                anyhow::anyhow!("GLM IndexPool requires a single-sequence decode position")
            })?;
            let seq_len = pos.saturating_add(1);
            let kpool = u32::try_from(ctx.config.index_kpool)
                .map_err(|_| anyhow::anyhow!("GLM IndexPool kpool exceeds u32"))?;
            let candidates = seq_len / kpool;
            let select_count = candidates.min(
                u32::try_from(ctx.config.index_topk / ctx.config.index_kpool)
                    .map_err(|_| anyhow::anyhow!("GLM IndexPool top-k exceeds u32"))?,
            );
            let tail_count = if ctx.config.index_kpool_always_select_tail {
                seq_len % kpool
            } else {
                0
            };
            // q_full contains [index query | query-local index-head weights].
            // Its former MLA-Q contents have been consumed by Q absorption.
            if select_count > 0 {
                prof!("glm_index_select", {
                    ops::index_pool_select_weighted(
                        ctx.gpu,
                        self.glm_index_select_k,
                        q_full,
                        kv_cache
                            .index_raw_pool_ptr(self.attn_layer_idx)
                            .ok_or_else(|| anyhow::anyhow!("GLM IndexPool sidecar missing"))?,
                        q_full.offset(ctx.config.index_n_heads * ctx.config.index_head_dim * 2),
                        indexer.kpool_compress_ape.weight,
                        meta.block_table,
                        q_out,
                        candidates,
                        bs as u32,
                        u32::try_from(
                            kv_cache
                                .index_raw_block_stride(self.attn_layer_idx)
                                .ok_or_else(|| {
                                    anyhow::anyhow!("GLM IndexPool raw stride missing")
                                })?,
                        )
                        .map_err(|_| anyhow::anyhow!("GLM IndexPool raw stride exceeds u32"))?,
                        ctx.config.index_n_heads as u32,
                        ctx.config.index_head_dim as u32,
                        kpool,
                        select_count,
                        stream,
                    )
                })?;
            }
            let selected_tokens = select_count
                .checked_mul(kpool)
                .and_then(|n| n.checked_add(tail_count))
                .ok_or_else(|| anyhow::anyhow!("GLM IndexPool selected-token width overflow"))?;
            prof!("glm_index_expand", {
                ops::index_pool_expand(
                    ctx.gpu,
                    self.glm_index_expand_k,
                    q_out,
                    q_full,
                    select_count,
                    kpool,
                    seq_len,
                    tail_count,
                    stream,
                )
            })?;
            prof!("glm_selected_mla", {
                ops::selected_mla_bf16(
                    ctx.gpu,
                    self.glm_selected_mla_k,
                    q_absorbed_buf,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    meta.block_table,
                    q_full,
                    attn_out,
                    selected_tokens,
                    bs as u32,
                    kv_lora,
                    nq,
                    inv_sqrt_d,
                    stream,
                )
            })?;
        } else {
            prof!("paged_attn", {
                ops::paged_decode_attn_bf16(
                    ctx.gpu,
                    self.paged_decode_mla_k,
                    q_absorbed_buf,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    meta.block_table,
                    meta.seq_len,
                    meta.max_blocks_per_seq,
                    1,
                    nq,
                    1,
                    mla_cache_dim,
                    bs as u32,
                    inv_sqrt_d,
                    nq * mla_cache_dim,
                    0,
                    stream,
                )
            })?;
        }

        // Step 9: V extraction (batched GEMV)
        let v_extracted = ctx.buffers.norm_output();
        prof!("v_extract", {
            if self.mla_batched_gemv_k.0 != 0 {
                ops::mla_batched_gemv(
                    ctx.gpu,
                    self.mla_batched_gemv_k,
                    attn_out,
                    mla.w_uv.weight,
                    v_extracted,
                    mla_v_dim,
                    kv_lora,
                    nq,
                    mla_cache_dim,
                    mla_v_dim,
                    stream,
                )
            } else {
                for head_idx in 0..nq as usize {
                    let attn_head = attn_out.offset(head_idx * mla_cache_dim as usize * 2);
                    let w_uv_head = mla
                        .w_uv
                        .weight
                        .offset(head_idx * mla.v_dim * mla.kv_lora_rank * 2);
                    let v_dst = v_extracted.offset(head_idx * mla.v_dim * 2);
                    let w_uv_dense = crate::weight_map::DenseWeight { weight: w_uv_head };
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        attn_head,
                        &w_uv_dense,
                        v_dst,
                        mla_v_dim,
                        kv_lora,
                        stream,
                    )?;
                }
                Ok(())
            }
        })?;

        // Step 10: O projection
        let o_out = ctx.buffers.qkv_output();
        prof!("wo", {
            if let Some(ref wo_nvfp4) = mla.wo_nvfp4 {
                self.nvfp4_decode_gemv(
                    ctx.gpu,
                    ctx.levers.gemv_sw,
                    v_extracted,
                    wo_nvfp4,
                    o_out,
                    h,
                    nq * mla_v_dim,
                    stream,
                )
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    v_extracted,
                    &mla.wo,
                    o_out,
                    h,
                    nq * mla_v_dim,
                    stream,
                )
            }
        })?;

        Ok(o_out)
    }
}
