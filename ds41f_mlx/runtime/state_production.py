# SPDX-License-Identifier: MIT
# Copyright (c) 2023 DeepSeek
# Derived from oMLX deepseek_v41/language.py (MIT), modified for owned decode.
# License: ../prefill_fp8_mlx/OMLX_MATH_LICENSE.
"""Standard-off block state production on the sole P7 packed cache.

Only this producer receives mutable slots/shared publications. Loaded modules
supply weights and stateless projections, norms and MoE; their Block, Attention,
Compressor and Indexer calls are never invoked. Writes are tentative under the
parent all-layer pending lease, not commits. No CED, replay, verification state,
cache creation, recovery or alternate representation exists here.
"""


class DecodeStateProducer:
    def __init__(self, mx, math):
        self.mx = mx
        self.math = math  # Parent supplies the admitted implementation handle.

    @staticmethod
    def validate_completion(cache, c, frontier):
        """Structural lifecycle gate after materialization, before lease commit.

        Storage widths/dtypes stay the qualified packed representation; this gate
        checks semantic lengths, including non-source empties, not a new format.
        """
        for i, item in enumerate(cache):
            source = i in c.kv_source_layers
            ratio = c.compress_ratios[i] if source else 0
            expected = {
                1: min(frontier, c.window_size),
                2: frontier // ratio if ratio else 0,
                3: frontier // ratio if ratio and i in c.index_source_layers else 0,
                4: frontier % ratio if ratio > 1 else 0,
                5: frontier % ratio if ratio > 1 else 0,
                6: c.engram_max_ngram_size - 1 if i == 0 and c.engram_layer_ids else 0,
            }
            for slot, length in expected.items():
                value = item[slot]
                if value is None or value.shape[0] != 1 or value.shape[1] != length:
                    raise RuntimeError(f'owned state lifecycle mismatch layer {i} slot {slot}')

    def compressor(self, module, x, cache, start):
        mx, r = self.mx, module.ratio
        if r == 1:
            return module.norm(module.wkv(x))
        kv, gate = module.wkv(x.astype(mx.float32)), module.wgate(x.astype(mx.float32))
        rem = start % r
        if rem:
            kv = mx.concatenate([cache[4][:, :rem], kv], 1)
            gate = mx.concatenate([cache[5][:, :rem], gate], 1)
        cutoff = kv.shape[1] // r * r
        cache[4], cache[5] = kv[:, cutoff:], gate[:, cutoff:]
        if not cutoff:
            return None
        pooled = mx.sum(
            kv[:, :cutoff].reshape(1, -1, r, kv.shape[-1])
            * mx.softmax(gate[:, :cutoff].reshape(1, -1, r, gate.shape[-1]), axis=2),
            axis=2,
        )
        return module.norm(pooled.astype(x.dtype))

    def indexer(self, module, x, qr, latent, cache, shared, start, ratio):
        mx, m = self.mx, self.math
        c, layer = module._config, module._layer
        end = start + x.shape[1]
        if layer in c.kv_source_layers:
            previous = cache[3]
            previous = (previous[:, :start // ratio] if previous is not None else
                        m.pack_activation(mx.zeros((1, 0, c.index_head_dim), x.dtype), bits=4))
            if latent is not None:
                pos = mx.arange(start // ratio, end // ratio) * ratio
                key = m.pack_activation(m.rope(module.k_norm(module.wk(latent)), pos, c, True), bits=4)
                previous = mx.concatenate([previous, key], 1)
            shared['index_k'] = cache[3] = previous
        key = shared['index_k']
        q = module.wq_b(qr).reshape(1, x.shape[1], c.index_n_heads, c.index_head_dim)
        q = m.quantize_activation(m.rope(q, mx.arange(start, end), c, True), bits=4)
        weights = module.weights_proj(x).astype(mx.float32) * (
            c.index_head_dim**-0.5 * c.index_n_heads**-0.5)
        candidates = None
        if 0 <= c.candidate_source_layer < layer:
            blocks = shared['candidates']
            candidates = blocks[..., None] * c.candidate_block_size + mx.arange(c.candidate_block_size)
            candidates = mx.where(blocks[..., None] >= 0, candidates, -1).reshape(
                1, x.shape[1], blocks.shape[-1] * c.candidate_block_size)
        if candidates is None:
            idx, blocks = m.packed_index_topk(
                q, key, weights, start, ratio, c.index_topk,
                block_count=c.candidate_topk_blocks if layer == c.candidate_source_layer else 0,
                block_size=c.candidate_block_size)
            if layer == c.candidate_source_layer:
                shared['candidates'] = blocks
            return idx
        scores = m.packed_index_scores(q, key, weights, start, ratio, candidates)
        count = min(c.index_topk, scores.shape[-1])
        order = mx.argsort(-scores, axis=-1)[..., :count].astype(mx.int32)
        valid = mx.take_along_axis(scores, order, -1) > -float('inf')
        idx = mx.take_along_axis(candidates, order, -1)
        return mx.sort(mx.where(valid, idx, -1), -1)

    def attention(self, module, x, cache, shared, start):
        mx, m = self.mx, self.math
        c, layer = module._config, module._layer
        ratio, length = c.compress_ratios[layer], x.shape[1]
        positions = mx.arange(start, start + length)
        # This loaded helper performs projections only and receives no state.
        query, kv_input = module._input_projections(x)
        qr = module.q_norm(query)
        q = m.rope(module.wq_b(qr).reshape(1, length, c.n_heads, c.head_dim),
                   positions, c, bool(ratio))
        new = m.pack_activation(m.rope(module.kv_norm(kv_input), positions, c, bool(ratio)))
        old = cache[1]
        old_len = min(start, c.window_size, 0 if old is None else int(old.shape[1]))
        kv = mx.concatenate([old[:, :old_len], new], 1) if old_len else new
        cache[1] = kv[:, -c.window_size:]
        if start == 0:
            local = mx.maximum(positions[:, None] - c.window_size + 1, 0)
            local = local + mx.arange(min(length, c.window_size))
        else:
            local = positions[:, None] - c.window_size + 1 + mx.arange(c.window_size)
        valid = (local >= max(0, start - old_len)) & (local <= positions[:, None])
        idx = mx.where(valid, local - (start - old_len), -1)[None]
        pooled = mx.zeros((1, 0, c.head_dim // 2 + c.head_dim // 16), mx.uint8)
        ci = mx.zeros((1, length, 0), mx.int32)
        if ratio:
            latent = None
            if layer in c.kv_source_layers:
                latent = self.compressor(module.compressor, x, cache, start)
                if cache[2] is None:
                    cache[2] = m.pack_activation(mx.zeros((1, 0, c.head_dim), x.dtype),
                                                bits=4, group_size=16, e4m3_scale=True)
                shared['kv'] = cache[2][:, :start // ratio]
            if layer in c.index_source_layers:
                shared['idx'] = self.indexer(module.indexer, x, qr, latent, cache, shared, start, ratio)
            if latent is not None:
                compressed = m.rope(latent, mx.arange(start // ratio, (start + length) // ratio)
                                    * ratio, c, True)
                compressed = m.pack_activation(compressed, bits=4, group_size=16, e4m3_scale=True)
                shared['kv'] = cache[2] = mx.concatenate([shared['kv'], compressed], 1)
            ci, pooled = shared['idx'], shared['kv']
        out = m.packed_sparse_attention(q, kv, pooled, idx, ci, module.attn_sink, c.head_dim**-0.5)
        out = m.rope(out, positions, c, bool(ratio), inverse=True)
        grouped = out.reshape(1, length, c.o_groups, -1)
        weight = module.wo_a.weight.reshape(c.o_groups, c.o_lora_rank, -1)
        return module.wo_b(mx.einsum('bsgd,grd->bsgr', grouped, weight).flatten(-2))

    def block(self, module, h, pre, cache, shared, start):
        if not getattr(cache, '_p6_append_pending', False):
            raise RuntimeError('state production requires pending target lease')
        if getattr(cache, '_mtp_verify_state', None) is not None:
            raise RuntimeError('standard-off producer cannot mutate MTP verification state')
        if h.shape[:2] != (1, 1) or start < 0:
            raise ValueError('state production requires one unpadded decode token')
        m, c = self.math, module._config
        ap, ao, ac = m.hc_mixes(h, module.hc_attn_fn, module.hc_attn_scale, module.hc_attn_base, c)
        x = m.hc_pre_norm(h, pre, module.attn_norm.weight, module.attn_norm.eps)
        h = m.hc_post(self.attention(module.attn, x, cache, shared, start), h, ao, ac)
        fp, fo, fc = m.hc_mixes(h, module.hc_ffn_fn, module.hc_ffn_scale, module.hc_ffn_base, c)
        h = m.hc_post(module.ffn(m.hc_pre_norm(h, ap, module.ffn_norm.weight,
                                            module.ffn_norm.eps), None), h, fo, fc)
        return h, fp
