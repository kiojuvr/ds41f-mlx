# SPDX-License-Identifier: MIT
# Copyright (c) 2023 DeepSeek
# Decode sequencing derived from oMLX deepseek_v41/language.py (MIT), modified.
# License: ../prefill_fp8_mlx/OMLX_MATH_LICENSE; artifacts/m45/provenance.json.
"""Owned standard-off, one-token/all-40-layer decode transaction.

No LanguageModel call, row extraction/merge, replay, or alternate cache. Owned
state production uses admitted numerical/storage/SSD Engram primitive handles.
A failed transaction burns every layer, including passive aliases; there is no
resumable partially mutated state. OFF execute retains its one-token contract;
begin_prefix_journal supplies a separate bounded producer/settlement primitive,
not a generation acceptance loop. M44 publishes history only after execute
returns. Single-flight OFF cancellation is observed between transactions.
"""
from contextlib import nullcontext

from ds41f_mlx.runtime.state_production import DecodeStateProducer
from ds41f_mlx.runtime.resource_admission import resources_for


class TargetForwardTransaction:
    def __init__(self, model, mx):
        self.model, self.mx = model, mx
        self.resources = resources_for(model)
        self.resources.require_backend(mx)
        self.producer = DecodeStateProducer(mx, self.resources.math)
        self.proposal_child = getattr(model, '_ds41f_proposal_child', None)
        self.tap_rows = None
        # Stable admitted primitives, never a fresh ambient import.
        math = self.resources.math
        self.hc_pre = math.hc_pre
        self.project_logits = math.project_logits
        self.pack_activation = math.pack_activation

    @staticmethod
    def invalidate(cache):
        for item in cache:
            item._p6_append_failed = True
            item._p6_append_invalid = True

    def validate(self, token, cache, frontier):
        self.resources.assert_active()
        c = self.model._config
        if token.shape != (1,) or len(cache) != 40 or len(self.model.layers) != 40:
            raise ValueError('owned target requires one token and all 40 layers')
        if c.n_layers != 40:
            raise ValueError('owned target requires the qualified 40-layer model')
        if c.engram_layer_ids and self.model._hasher is None:
            raise ValueError('owned target requires tokenizer-derived Engram map')
        for i, item in enumerate(cache):
            if type(item) is not self.resources.cache_type:
                raise ValueError('unadmitted packed-cache implementation')
            if (getattr(item, '_p6_append_invalid', False)
                or getattr(item, '_p6_append_failed', False)
                or getattr(item, '_p6_append_pending', False)):
                raise RuntimeError('owned target cache is invalid or pending')
            if getattr(item, '_mtp_verify_state', None) is not None:
                raise ValueError('owned target cannot consume MTP verification state')
            if len(item.cache) != 7:
                raise ValueError('owned target requires seven packed slots')
            ratio = c.compress_ratios[i] if i in c.kv_source_layers else 0
            if item.compress_ratio != ratio:
                raise ValueError('owned target cache compression layout mismatch')
            if item.size() != frontier or item.batch_size != 1:
                raise ValueError('owned target cache frontier/row mismatch')
            mask = item.make_mask(1)
            if mask is not None and not bool(self.mx.all(mask).item()):
                raise ValueError('owned target cannot consume padded input')

    def begin_prefix_journal(self, cache, frontier, bound, stream, *, fault=None):
        from ds41f_mlx.runtime.accepted_prefix import AcceptedPrefixJournal
        return AcceptedPrefixJournal(self, cache, frontier, bound, stream, fault)

    def forward(self, token, cache, frontier, journal=None):
        """Sequence qualified primitives directly on the sole live cache list."""
        mx, model = self.mx, self.model
        c = model._config
        ids = token[:, None]
        h = model.embed(ids)
        hashes, history = (None, None)
        if model._hasher is not None:
            hashes, history = model._hasher(ids, cache[0][6], None)
        h = mx.repeat(h[..., None, :], c.hc_mult, -2)
        pre = mx.broadcast_to((mx.arange(c.hc_mult) == 0).astype(mx.float32), h.shape[:-1])
        shared = {}
        captured = {}
        child = self.proposal_child
        if child is not None:
            child.assert_active()
        self.tap_rows = None
        prefetch = getattr(model, '_engram_prefetch', None)
        with prefetch.forward() if prefetch is not None else nullcontext():
            if prefetch is not None and c.engram_layer_ids:
                prefetch.submit(model.layers[c.engram_layer_ids[0]].engram.embed, hashes[:, :, 0])
            for i, layer in enumerate(model.layers):
                if 'engram' in layer:
                    ix = list(c.engram_layer_ids).index(i)
                    if prefetch is not None:
                        mx.async_eval(h, pre)
                    h = layer.engram(h, hashes[:, :, ix], None)
                    if prefetch is not None and ix + 1 < len(c.engram_layer_ids):
                        prefetch.submit(model.layers[c.engram_layer_ids[ix + 1]].engram.embed,
                                        hashes[:, :, ix + 1])
                if journal is None:
                    h, pre = self.producer.block(layer, h, pre, cache[i], shared, frontier)
                else:
                    h, pre = self.producer.block(layer, h, pre, cache[i], shared, frontier, journal)
                if child is not None and i in c.dspark_target_layer_ids:
                    captured[i] = mx.mean(h, axis=2)
                if prefetch is not None and 'engram' in layer:
                    mx.async_eval(h, pre)
                cache[i][0] = mx.array([frontier + 1], mx.int32)
                if history is not None and i == 0:
                    cache[i][6] = mx.array(history, mx.int64)
                # Preserve the qualified seven-slot empty representation. This
                # initializes absent slots only, never repacks existing state.
                for slot in range(1, 7):
                    if cache[i][slot] is None:
                        width = c.index_head_dim if slot == 3 else c.head_dim
                        empty = mx.zeros((1, 0, width), h.dtype)
                        if slot == 1:
                            empty = self.pack_activation(empty)
                        elif slot == 2:
                            empty = self.pack_activation(empty, 4, 16, True)
                        elif slot == 3:
                            empty = self.pack_activation(empty, 4)
                        cache[i][slot] = mx.zeros((1, 0), mx.int64) if slot == 6 else empty
                # Admission metadata is state too; no subordinate advance call.
                if cache[i].lengths is not None:
                    cache[i].lengths -= 1
                if cache[i].left_padding is not None:
                    cache[i].left_padding -= 1
        if child is not None:
            self.tap_rows = mx.concatenate([captured[i] for i in c.dspark_target_layer_ids], -1)
            # The caller materializes and detaches this bounded same-forward
            # receipt with the row completion barrier, not at each tapped layer.
        return self.project_logits(model.norm(self.hc_pre(h, pre)), model.head.weight)[:, -1, :]

    def execute(self, token, cache, frontier, sampler, stream):
        mx = self.mx
        objects = tuple(cache)
        try:
            with mx.stream(stream):
                self.validate(token, cache, frontier)
                for item in cache:
                    item._p6_append_pending = True
                logits = self.forward(token, cache, frontier)
                logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                pending = sampler(logprobs)
                # Materialize ALL mutated state, not just dependencies of logits.
                # Completion is the commit barrier, including compressor/history
                # writes that may otherwise remain lazy after sampling.
                mx.async_eval(pending, logprobs,
                    *(() if self.tap_rows is None else (self.tap_rows,)), *(x for c in cache
                    for x in (*c.cache, c.lengths, c.left_padding) if x is not None))
                mx.eval(token)
                consumed = int(token.item())
            mx.synchronize(stream)
            if self.tap_rows is not None:
                from ds41f_mlx.runtime.hidden_taps import detach
                with mx.stream(stream):
                    self.tap_rows = detach(mx, self.tap_rows)
            if len(cache) != len(objects) or any(a is not b for a, b in zip(cache, objects)):
                raise RuntimeError('owned target cache objects replaced')
            if any(item.size() != frontier + 1 for item in cache):
                raise RuntimeError('owned target commit frontier mismatch')
            self.producer.validate_completion(cache, self.model._config, frontier + 1)
            for item in cache:
                item._p6_append_pending = False
            return consumed, pending
        except BaseException:
            self.tap_rows = None
            self.invalidate(objects)
            self.invalidate(cache)
            raise
