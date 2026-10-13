"""Investigation-only launcher: token LCP + paired retention/lookup evidence.

Usage: <admitted-python> -I /absolute/path/tools/observe_prefix_frontier.py \
    /absolute/path/evidence.jsonl [normal ds41f_mlx.serve arguments]

No wire/body logging, decoded text, cache policy changes, or second cache owner.
Token tuples are diagnostic-only and bounded to 256 publication records. Run
before the OpenCode session; history evicted before launch cannot be inferred.
"""
from collections import deque
import json
from pathlib import Path
import sys


def lcp(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def comparison(old, new):
    frontier = lcp(old, new)
    return dict(lcp=frontier, previous_tokens=len(old), next_tokens=len(new),
                previous_frontier_id=old[frontier] if frontier < len(old) else None,
                next_frontier_id=new[frontier] if frontier < len(new) else None)


def install(output):
    from ds41f_mlx.serving.paired_checkpoint import PairedCheckpointAuthority, image_keys
    from ds41f_mlx.serving.production_mtp import ProductionScheduler

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    history = deque(maxlen=256)
    previous = None
    active = None
    publication = None
    lookup = None

    def emit(event, **data):
        with output.open('a') as stream:
            stream.write(json.dumps(dict(event=event, **data)) + '\n')

    drop_original = PairedCheckpointAuthority._drop_payload
    clear_original = PairedCheckpointAuthority.clear
    publish_original = PairedCheckpointAuthority.publish
    acquire_original = PairedCheckpointAuthority.acquire
    retire_original = ProductionScheduler._retire
    schedule_original = ProductionScheduler._schedule_waiting

    def drop(self, key):
        checkpoint = self._payloads.get(key)
        if checkpoint is not None:
            emit('payload_dropped', request_id=active, frontier=len(checkpoint.tokens),
                 phase='lookup' if lookup_in_progress[0] else 'publication_or_external_eviction')
        return drop_original(self, key)

    def clear(self):
        emit('authority_cleared', request_id=active,
             frontiers=sorted(len(c.tokens) for c in self._payloads.values()))
        return clear_original(self)

    lookup_in_progress = [False]

    def publish(self, checkpoint):
        nonlocal publication
        result = publish_original(self, checkpoint)
        # Token-only evidence, never retain checkpoint arrays/cache handles.
        tokens = checkpoint.tokens
        block = self.paged.find_cached_block(list(tokens), extra_keys=image_keys(checkpoint.images))
        present = block is not None and block.block_hash in self._payloads
        if present:
            history.append((tokens, checkpoint.identity, checkpoint.images))
        if active is not None:
            publication = tokens
        emit('publication', request_id=active, frontier=len(tokens), retained=present,
             live_frontiers=sorted(len(c.tokens) for c in self._payloads.values()))
        return result

    def acquire(self, prompt, **kwargs):
        nonlocal lookup
        prompt = tuple(prompt)
        images = kwargs.get('images', ())

        def compatible(tokens, identity, old_images):
            return (identity == self.identity and len(tokens) <= len(prompt)-2
                    and prompt[:len(tokens)] == tokens
                    and tuple(s for s in images if s['start'] < len(tokens)) == old_images)

        live = sorted(len(c.tokens) for c in self._payloads.values()
                      if compatible(c.tokens, c.identity, c.images))
        past = sorted(set(len(t) for t, identity, im in history if compatible(t, identity, im)))
        before = set(self._payloads)
        lookup_in_progress[0] = True
        try:
            result = acquire_original(self, prompt, **kwargs)
        finally:
            lookup_in_progress[0] = False
        restored = len(result[2]) if result is not None else 0
        lookup = dict(cached_tokens=restored, compatible_live_frontiers=live,
                      compatible_observed_frontiers=past,
                      lookup_removed_payloads=len(before-set(self._payloads)),
                      checkpoint_status=('restored' if restored else
                          'present_but_not_restored' if live else
                          'previously_published_now_absent' if past else
                          'no_compatible_checkpoint_observed'),
                      history_window=len(history))
        if previous is not None:
            lookup.update(previous_request_id=previous['request_id'],
                          previous_finish_reason=previous['finish_reason'],
                          previous_cached_tokens=previous['cached_tokens'],
                          request_lcp=comparison(previous['prompt'], prompt),
                          canonical_lcp=comparison(previous['canonical'], prompt))
        emit('lookup', request_id=active, **lookup)
        return result

    def schedule(self):
        nonlocal active, lookup
        if not self.running and self.waiting:
            active = self.waiting[0].request_id
            lookup = None
        return schedule_original(self)

    def retire(self, request, **kwargs):
        nonlocal previous, active, publication
        active = request.request_id
        publication = None
        prompt = tuple(request.prompt_token_ids)
        result = retire_original(self, request, **kwargs)
        trace = request.trace
        choices = (trace.get('response') or {}).get('choices') or []
        finish = choices[0].get('finish_reason') if choices else None
        # Publication happens after canonical settlement, before native cleanup.
        # If no final publication was possible, do not fabricate canonical IDs.
        canonical = publication
        if canonical is not None and len(canonical) == trace.get('canonical_frontier'):
            if (previous is not None and lookup is not None
                    and lookup.get('previous_request_id') == previous['request_id']
                    and previous['finish_reason'] == 'stop'
                    and previous['cached_tokens'] > 0 and request.cached_tokens == 0):
                emit('final_to_next_miss_candidate', request_id=active, **lookup,
                     note='Verify same OpenCode session and new user turn externally; no body recorded.')
            previous = dict(request_id=active, prompt=prompt, canonical=canonical,
                            finish_reason=finish, cached_tokens=request.cached_tokens)
        else:
            emit('canonical_unavailable', request_id=active)
            previous = None
        active = None
        return result

    PairedCheckpointAuthority._drop_payload = drop
    PairedCheckpointAuthority.clear = clear
    PairedCheckpointAuthority.publish = publish
    PairedCheckpointAuthority.acquire = acquire
    ProductionScheduler._schedule_waiting = schedule
    ProductionScheduler._retire = retire
    emit('observer_started', history_limit=256, prompt_body_logged=False)


if __name__ == '__main__':
    # Do not import Scheduler/MLX-LM before admission. Transformers' lazy
    # import changes its ModuleSpec search locations from str to Path, which
    # the installed-origin admission deliberately checks before imports.
    from ds41f_mlx import mtp_identity
    original_inspect = mtp_identity.inspect
    installed = False

    def inspect_then_observe(*args, **kwargs):
        global installed
        report = original_inspect(*args, **kwargs)
        if not installed:
            install(sys.argv[1])
            installed = True
        return report

    mtp_identity.inspect = inspect_then_observe
    from ds41f_mlx.serve import main
    raise SystemExit(main(sys.argv[2:]))
