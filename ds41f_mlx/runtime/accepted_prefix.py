"""Bounded producer undo journal; no token acceptance or generation authority.

The caller owns the exact-list lease and committed frontier. While borrowed,
all objects remain pending and unusable by OFF/admission/idle transfer. A returned
settled frontier is a publication receipt, NOT a history/RNG commit.
"""
import numpy as np


class AcceptedPrefixJournal:
    MAX_SPAN = 32

    def __init__(self, target, cache, frontier, bound, stream, fault=None):
        if not isinstance(bound, int) or not 1 <= bound <= self.MAX_SPAN:
            raise ValueError('journal span must be in [1, 32]')
        target.validate(target.mx.array([0], target.mx.int32), cache, frontier)
        target.producer.validate_completion(cache, target.model._config, frontier)
        self.target, self.mx = target, target.mx
        self.cache, self.objects = cache, tuple(cache)
        self.frontier, self.bound, self.stream = frontier, bound, stream
        self.fault = fault or (lambda phase, layer, slot: None)
        self.phase, self.count, self.payload_bytes = 'begin', 0, 0
        self.settlement_packed_copy_bytes = 0
        self.evicted, self.projections, self.histories = [], {}, []
        self.initial = []
        self.logits = []
        self.tap_rows = []
        try:
            for item in self.objects:
                item._p6_append_pending = True
                item._accepted_prefix_journal = self
            with self.mx.stream(stream):
                for item in self.objects:
                    # No window/compressed/index snapshots, only bounded tails,
                    # history and row admission scalars.
                    self.initial.append({s: self.detach(item[s]) for s in (4, 5, 6)})
                    self.initial[-1]['lengths'] = self.detach(item.lengths)
                    self.initial[-1]['left_padding'] = self.detach(item.left_padding)
            self.mx.synchronize(stream)
            self.phase = 'tentative'
        except BaseException:
            self.burn()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.phase not in ('retired', 'burned'):
            if exc_type is not None:
                self.burn()
            else:
                self.cancel()
        return False

    def detach(self, value):
        if value is None:
            return None
        # contiguous/array(slice) are NOT a guaranteed allocation boundary in
        # MLX. A bounded byte-preserving host copy explicitly severs both parent
        # allocation and lazy graph. bfloat16 is copied through its uint16 view.
        self.mx.eval(value)
        raw = np.array(value.view(self.mx.uint8), copy=True)
        out = self.mx.array(raw).view(value.dtype).reshape(value.shape)
        self.mx.eval(out)
        self.payload_bytes += value.nbytes
        return out

    def check(self):
        self.target.resources.assert_active()
        if (len(self.cache) != len(self.objects)
            or any(a is not b for a, b in zip(self.cache, self.objects))
            or any(getattr(x, '_accepted_prefix_journal', None) is not self
                   or not x._p6_append_pending or len(x.cache) != 7
                   or getattr(x, '_p6_append_invalid', False)
                   or getattr(x, '_p6_append_failed', False) for x in self.objects)):
            raise RuntimeError('borrowed exact target lease changed')

    def window_write(self, layer, old, window):
        if layer in self.evicted[-1]:
            raise RuntimeError('duplicate journal window mutation')
        # One-token producer: at most one lost chronological row per layer.
        lost = old[:, :1] if old.shape[1] == window else None
        self.evicted[-1][layer] = self.detach(lost)

    def compressor_write(self, layer, kv, gate):
        rows = self.projections.setdefault(layer, [])
        if len(rows) != self.count or kv.shape[1] != 1 or gate.shape != kv.shape:
            raise RuntimeError('unbounded or duplicate journal projection')
        rows.append((self.detach(kv), self.detach(gate)))

    def advance(self, token):
        if self.phase != 'tentative' or self.count >= self.bound:
            raise RuntimeError('journal not open or span exhausted')
        try:
            self.check()
            if token.shape != (1,):
                raise ValueError('journal consumes one target input per tentative position')
            for item in self.objects:
                mask = item.make_mask(1)
                if mask is not None and not bool(self.mx.all(mask).item()):
                    raise ValueError('journal cannot advance past unpadded admission')
            self.evicted.append({})
            with self.mx.stream(self.stream):
                logits = self.target.forward(token, self.cache, self.frontier + self.count,
                                             journal=self)
                if getattr(self.target, 'tap_rows', None) is not None:
                    self.tap_rows.append(self.target.tap_rows)
                    self.target.tap_rows = None
                self.histories.append(self.detach(self.cache[0][6]))
                # B lazy logits could otherwise pin B context-sized packed
                # parents. Retain bounded outputs, never their execution graphs.
                self.mx.eval(logits)
                self.logits.append(logits)
            if len(self.evicted[-1]) != len(self.objects):
                raise RuntimeError('incomplete all-layer journal window coverage')
            self.count += 1
            return logits
        except BaseException:
            self.burn()
            raise

    def complete(self):
        if self.phase != 'tentative':
            raise RuntimeError('journal completion out of order')
        try:
            self.check()
            c = self.target.model._config
            expected = {i for i in c.kv_source_layers if c.compress_ratios[i] > 1}
            if self.count and (set(self.projections) != expected
                               or any(len(rows) != self.count for rows in self.projections.values())):
                raise RuntimeError('incomplete source projection journal')
            self.fault('materialize-before', -1, -1)
            with self.mx.stream(self.stream):
                for i, item in enumerate(self.objects):
                    self.mx.eval(*(x for x in (*item.cache, item.lengths, item.left_padding)
                                   if x is not None))
                    self.fault('materialize-state', i, -1)
                self.mx.eval(*self.logits)
            self.mx.synchronize(self.stream)
            if any(item[0].shape != (1,) or item[0].dtype != self.mx.int32
                   or item.size() != self.frontier + self.count for item in self.objects):
                raise RuntimeError('tentative all-layer frontier mismatch')
            for i, item in enumerate(self.objects):
                for name in ('lengths', 'left_padding'):
                    initial, actual = self.initial[i][name], getattr(item, name)
                    if ((initial is None) != (actual is None)
                        or initial is not None and (actual.dtype != initial.dtype
                            or actual.shape != initial.shape or not bool(self.mx.array_equal(
                                actual, initial - self.count).item()))):
                        raise RuntimeError('tentative admission metadata mismatch')
            self.target.producer.validate_completion(
                self.objects, self.target.model._config, self.frontier + self.count)
            self.fault('materialize-after', -1, -1)
            self.phase = 'materialized'
        except BaseException:
            self.burn()
            raise

    def settle(self, accepted, *, publish):
        """Prepare/evaluate ALL state, then publish under the pending barrier.

        Acceptance is supplied by the generation owner in consumed-input units.
        publish(end) is the parent's frontier-only publication hook, invoked with
        ALL aliases still pending. It must not sample, emit responses, or start
        another execution. Failure burns even if the hook advanced owner metadata.
        Any failure (including during publication) burns originals and replacements.
        """
        if self.phase != 'materialized':
            raise RuntimeError('settlement requires materialized journal')
        if not isinstance(accepted, int) or not 0 <= accepted <= self.count:
            raise ValueError('accepted prefix outside tentative span')
        try:
            self.check()
            mx, c = self.mx, self.target.model._config
            end = self.frontier + accepted
            prepared = []
            with mx.stream(self.stream):
                for i, item in enumerate(self.objects):
                    states = list(item.cache)
                    window = states[1]
                    for step in range(self.count - 1, accepted - 1, -1):
                        window = window[:, :-1]
                        lost = self.evicted[step][i]
                        if lost is not None:
                            window = mx.concatenate([lost, window], 1)
                    # take is an explicit output allocation, unlike slice or
                    # contiguous (which can retain an oversized parent bucket).
                    states[1] = mx.take(window, mx.arange(window.shape[1]), axis=1)
                    self.settlement_packed_copy_bytes += states[1].nbytes
                    r = item.compress_ratio
                    for s in (2, 3):
                        n = end // r if r else 0
                        if s == 3 and i not in c.index_source_layers:
                            n = 0
                        if n < states[s].shape[1]:
                            # Publish selected packed BYTES without quantization,
                            # cache creation, replay, or full-cache repacking.
                            states[s] = mx.take(states[s], mx.arange(n), axis=1)
                            self.settlement_packed_copy_bytes += states[s].nbytes
                    for s in (4, 5):
                        if r > 1:
                            rows = [self.initial[i][s]] + [p[s - 4] for p in
                                self.projections.get(i, [])[:accepted]]
                            joined = mx.concatenate(rows, 1)
                            used = (self.frontier % r + accepted) // r * r
                            # Detached remainder prevents retaining the journal.
                            states[s] = self.detach(joined[:, used:])
                        else:
                            states[s] = self.initial[i][s]
                    states[6] = (self.histories[accepted - 1] if i == 0 and accepted
                                 else self.initial[i][6])
                    states[0] = mx.array([end], mx.int32)
                    meta = {name: None if self.initial[i][name] is None else
                            self.initial[i][name] - accepted
                            for name in ('lengths', 'left_padding')}
                    self.fault('prepare', i, -1)
                    prepared.append((states, meta))
                mx.eval(*(x for states, meta in prepared
                          for x in (*states, *meta.values()) if x is not None))
            mx.synchronize(self.stream)
            # Validate prepared arrays without manufacturing executable cache
            # objects or giving a shadow model authority.
            for i, (states, _) in enumerate(prepared):
                r = self.objects[i].compress_ratio
                lengths = {1: min(end, c.window_size), 2: end // r if r else 0,
                           3: end // r if r and i in c.index_source_layers else 0,
                           4: end % r if r > 1 else 0, 5: end % r if r > 1 else 0,
                           6: c.engram_max_ngram_size - 1 if i == 0 and c.engram_layer_ids else 0}
                if int(states[0].item()) != end or any(states[s].shape[1] != n
                                                      for s, n in lengths.items()):
                    raise RuntimeError('prepared accepted state mismatch')
            self.check()
            self.phase = 'publishing'
            # Pending remains set on ALL aliases until the entire publication
            # succeeds. A mid-publication exception cannot yield continuation.
            for i, (states, meta) in enumerate(prepared):
                for slot, value in enumerate(states):
                    self.fault('publish', i, slot)
                    self.objects[i][slot] = value
                for name, value in meta.items():
                    self.fault('publish', i, name)
                    setattr(self.objects[i], name, value)
            self.check()
            if any(item.size() != end for item in self.objects):
                raise RuntimeError('published accepted frontier mismatch')
            self.target.producer.validate_completion(self.objects, c, end)
            self.fault('publication-boundary', -1, -1)
            self.check()
            publish(end)
            self.check()
            for item in self.objects:
                item._p6_append_pending = False
                del item._accepted_prefix_journal
            self.phase = 'published'
            self.retire()
            return end
        except BaseException:
            self.burn()
            raise

    def cancel(self):
        # Cancellation never chooses a token prefix: restore the committed F.
        if self.phase == 'tentative':
            self.complete()
        return self.settle(0, publish=lambda end: None)

    def retire(self):
        self.evicted.clear()
        self.projections.clear()
        self.histories.clear()
        self.initial.clear()
        self.logits.clear()
        self.tap_rows.clear()
        if self.target is not None:
            self.target.tap_rows = None
        self.fault = None
        self.cache = None
        self.objects = ()
        self.target = None
        self.mx = self.stream = None
        self.phase = 'retired'

    def burn(self):
        if self.target is not None:
            self.target.invalidate(self.objects)
            self.target.invalidate(self.cache)
            try:
                self.mx.synchronize(self.stream)
            except BaseException:
                # No continuation survives an uncertain device drain either.
                self.drain_failed = True
            for item in (*self.objects, *self.cache):
                if getattr(item, '_accepted_prefix_journal', None) is self:
                    del item._accepted_prefix_journal
            self.retire()
            self.phase = 'burned'
