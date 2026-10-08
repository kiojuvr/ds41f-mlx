"""M47 child resources and generation-scoped DSpark derived acceleration.

No target execution, scheduler, acceptance, canonical sampling or history here.
Explicitly opt-in qualification API; standard runtime defaults are unchanged.
"""
import json
from contextlib import closing
from dataclasses import asdict

import mlx.nn as nn
from mlx.utils import tree_flatten

from ds41f_mlx.model_execution import dspark
from ds41f_mlx.model_execution.checkpoint import iter_source_weights
from ds41f_mlx.model_execution.loading import set_module
from ds41f_mlx.model_execution.quantization import QuantizedProjection
from ds41f_mlx.runtime.resource_admission import require


class ProposalParameters(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mtp = dspark.make_stages(config)


class ProposalResources:
    def __init__(self, parent, model):
        parent.validate_binding(model)
        self.parent, self.model = parent, model
        self.mx = parent.modules['mlx.core']
        self.config = model._config
        self.active = True
        self.receipts, self.producers = set(), set()
        self.identity = parent.identity
        self.parameters = ProposalParameters(self.config)
        self._config = asdict(self.config)
        try:
            root = parent._checkpoint
            raw = json.loads((root / 'config.json').read_text())
            mapping = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
            mapping = {key: value for key, value in mapping.items() if key.startswith('mtp.')}
            require(len(mapping) == 2401, 'unqualified proposal tensor inventory')
            self.source_keys = tuple(sorted(mapping))
            self.shards = tuple(sorted(set(mapping.values())))
            logical = {name: value.shape for name, value in tree_flatten(self.parameters.parameters())}
            seen = set()
            with closing(iter_source_weights(root, raw, mapping, preserve_mtp=True)) as batches:
                for values, specs in batches:
                    values = {key.removeprefix('language_model.'): value for key, value in values.items()}
                    specs = {key.removeprefix('language_model.'): value for key, value in specs.items()}
                    require(not seen.intersection(values), 'duplicate proposal tensor')
                    for name, spec in specs.items():
                        shape = logical[name + '.weight']
                        bits, group = spec['bits'], spec.get('group_size', 32)
                        require(values[name + '.weight'].shape == (*shape[:-1], shape[-1] * bits // 32)
                                and values[name + '.scales'].shape == (*shape[:-1], shape[-1] // group),
                                'proposal packed shape mismatch')
                        set_module(self.parameters, name, QuantizedProjection(
                            values[name + '.weight'], values[name + '.scales'],
                            biases=values.get(name + '.biases'), **spec))
                    for name, value in values.items():
                        if name.rsplit('.', 1)[0] not in specs:
                            require(name in logical and value.shape == logical[name], 'proposal tensor shape mismatch')
                    self.parameters.load_weights(list(values.items()), strict=False)
                    self.mx.eval(values)
                    seen.update(values)
            require(seen == {name for name, _ in tree_flatten(self.parameters.parameters())},
                    'incomplete proposal parameters')
            self.parameters.eval()
            for _, module in self.parameters.named_modules():
                cls = type(module)
                origin = parent.modules.get(cls.__module__)
                require(origin is not None and vars(origin).get(cls.__name__) is cls,
                        'unadmitted proposal module class')
            self._binding = self.binding()
            object.__setattr__(model, '_ds41f_proposal_child', self)
        except BaseException:
            self.retire()
            raise

    def binding(self):
        return (tuple((name, id(module), type(module)) for name, module in self.parameters.named_modules()),
                tuple((name, id(value)) for name, value in tree_flatten(self.parameters.parameters())),
                id(self.model.embed), id(self.model.head), id(self.model.head.weight))

    def assert_active(self):
        self.parent.assert_active()
        require(self.active, 'proposal resources retired')
        require(asdict(self.config) == self._config and self.binding() == self._binding,
                'proposal resources substituted')

    def retire(self):
        self.active = False
        for producer in tuple(self.producers):
            producer.retire()
        for receipt in tuple(self.receipts):
            receipt.retire()
        self.mx.synchronize()
        self.parameters = None
        if getattr(self.model, '_ds41f_proposal_child', None) is self:
            object.__delattr__(self.model, '_ds41f_proposal_child')
        self.model = None  # release shared target read handles after revocation


class DSparkProposalProducer:
    def __init__(self, session, seed=None):
        self.session = session
        self.child = session.target_forward.proposal_child
        self.child.assert_active()
        if getattr(session, '_proposal_producer', None) is not None:
            raise RuntimeError('one proposal producer per generation')
        self.active = True
        self.rings = dspark.make_cache(self.child.config)
        bootstrap = getattr(session, '_tap_receipt', None)
        if bootstrap is None or bootstrap.retired:
            raise RuntimeError('same-forward bootstrap receipt required')
        self.frontier = seed.start if seed is not None else bootstrap.start
        self.child.producers.add(self)
        session._proposal_producer = self
        try:
            if seed is not None:
                self.advance(seed)
            self.advance(session.take_tap_receipt())
        except BaseException:
            self.retire()
            raise

    def advance(self, receipt, *, _fault=None):
        if not self.active:
            raise RuntimeError('proposal producer retired')
        try:
            self.child.assert_active()
            rows = receipt.take(self.child, self.session, self.frontier)
            first = self.child.parameters.mtp[0]
            x = first.main_norm(first.main_proj(rows))
            for stage, ring in zip(self.child.parameters.mtp, self.rings):
                stage.attn.append_context(x, ring, start_offset=self.frontier)
                if _fault is not None:
                    _fault('ring-appended')
            self.child.mx.eval(*(ring.keys for ring in self.rings))
            self.frontier = receipt.end
            if {ring.offset for ring in self.rings} != {self.frontier}:
                raise RuntimeError('proposal ring frontier mismatch')
        except BaseException:
            receipt.retire()
            self.retire()
            raise

    def propose(self, width=None, *, _fault=None):
        self.session._require_unborrowed()
        try:
            self.child.assert_active()
            if not self.active or self.session._stopped or self.session._failed:
                raise RuntimeError('live proposal session required')
            if self.session.token_frontier > 8192:
                raise RuntimeError('proposal context outside qualified 8K reference envelope')
            if self.frontier != self.session.token_frontier:
                raise RuntimeError('proposal ring is stale')
            c, mx = self.child.config, self.child.mx
            if any(r.keys is None or r.keys.shape[2] != c.window_size for r in self.rings):
                raise RuntimeError('full proposal priming window required')
            width = c.dspark_block_size if width is None else int(width)
            if not 1 <= width <= c.dspark_block_size:
                raise ValueError('proposal width outside checkpoint bound')
            # A read-only numerical view. No target cache or LanguageModel call.
            view = type('ProposalMathView', (), {})()
            view._config, view.mtp = c, self.child.parameters.mtp
            view.embed, view.head = self.child.model.embed, self.child.model.head
            anchor = self.session._pending.reshape(1, 1)
            if _fault is not None:
                _fault('proposal-before')
            logits, _ = dspark.proposal_forward(view, anchor, self.rings, width)
            previous, result = anchor[:, 0], []
            for i in range(width):
                bias, _ = view.mtp[-1].markov_head(previous)
                previous = mx.argmax(logits[:, i] + bias, axis=-1).astype(mx.uint32)
                mx.eval(previous)
                result.append(int(previous.item()))
            return tuple(result)
        except BaseException:
            self.retire()
            raise

    def cycle(self, *, width=None, cancelled=None, _fault=None):
        drafts = self.propose(width)
        try:
            result = self.session.speculative_cycle(drafts, cancelled=cancelled, _fault=_fault)
        except BaseException:
            self.retire()
            raise
        if self.session._stopped:
            self.retire()
        elif result['consumed_positions']:
            self.advance(self.session.take_tap_receipt())
        return result

    def retire(self):
        self.active = False
        for ring in self.rings:
            ring.retire()
        self.child.producers.discard(self)
        if getattr(self.session, '_proposal_producer', None) is self:
            self.session._proposal_producer = None
