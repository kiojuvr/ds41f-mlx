"""Investigation-only launcher: token LCP + paired retention/lookup evidence.

Usage: <admitted-python> -I /absolute/path/tools/observe_prefix_frontier.py \
    /absolute/path/evidence.jsonl [normal ds41f_mlx.serve arguments]

No wire/body logging, decoded text, cache policy changes, or second cache owner.
Token tuples are diagnostic-only and bounded to 256 publication records. Run
before the OpenCode session; history evicted before launch cannot be inferred.
"""
from collections import deque
import json
import hashlib
import hmac
import secrets
import re
from pathlib import Path
import sys


def lcp(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def comparison(old, new):
    frontier = lcp(old, new)
    return dict(lcp=frontier, previous_tokens=len(old), next_tokens=len(new),
                previous_frontier_id=old[frontier] if frontier < len(old) else None,
                next_frontier_id=new[frontier] if frontier < len(new) else None)


def structural_snapshot(payload, key):
    """Comparable field fingerprints; the per-process key is never published."""
    def digest(value):
        data = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
        return hmac.new(key, data, hashlib.sha256).hexdigest()

    return {name: ([{field: digest(value) for field, value in message.items()}
                    for message in values] if name == 'messages' else digest(values))
            for name, values in payload.items()}


def first_structure_change(old, new):
    """Report field paths only, not fingerprints or field values."""
    for name in sorted(old.keys() | new.keys()):
        if name == 'messages':
            continue
        if old.get(name) != new.get(name):
            return dict(field=name)
    a, b = old.get('messages', []), new.get('messages', [])
    for index, (x, y) in enumerate(zip(a, b)):
        fields = sorted(k for k in x.keys() | y.keys() if x.get(k) != y.get(k))
        if fields:
            return dict(field='messages', message_index=index, changed_fields=fields)
    if len(b) < len(a):
        return dict(field='messages', message_index=len(b), removed=True)
    return None  # Exact historical extension (new messages are allowed).


def converted_payload(conversation):
    return dict(thinking_mode=conversation.thinking_mode,
                reasoning_effort=conversation.reasoning_effort,
                tool_choice=conversation.tool_choice,
                response_format=conversation.response_format,
                tools=[dict(name=t.name, description=t.description,
                            parameters=t.parameters, strict=t.strict) for t in conversation.tools],
                messages=[dict(role=m.role, content=m.content,
                               reasoning_content=m.reasoning_content,
                               tool_call_id=m.tool_call_id,
                               tool_calls=None if m.tool_calls is None else
                                   [dict(id=t.id, name=t.name, arguments=t.arguments) for t in m.tool_calls])
                          for m in conversation.messages])


def reminder_structure(conversation, key):
    """Diagnose harness reminder relocation without logging text or digests."""
    result = []
    for message in conversation.messages:
        blocks = re.findall(r'<system-reminder>.*?</system-reminder>', message.content, re.S)
        remainder = re.sub(r'<system-reminder>.*?</system-reminder>', '', message.content, flags=re.S).rstrip()
        result.append(dict(role=message.role, content_chars=len(message.content),
            plan_reminder=any('Plan Mode - System Reminder' in b for b in blocks),
            fingerprints=structural_snapshot(dict(blocks=blocks, remainder=remainder), key)))
    return result


def reminder_change(old, new, index):
    if index >= len(old) or index >= len(new):
        return None
    a, b = old[index], new[index]
    return dict(previous_role=a['role'], next_role=b['role'],
        previous_content_chars=a['content_chars'], next_content_chars=b['content_chars'],
        previous_plan_reminder=a['plan_reminder'], next_plan_reminder=b['plan_reminder'],
        non_reminder_text_equal=a['fingerprints']['remainder'] == b['fingerprints']['remainder'],
        identical_reminder_on_later_message=any(
            a['plan_reminder'] and z['fingerprints']['blocks'] == a['fingerprints']['blocks']
            for z in new[index + 1:]))


def mismatch_region(prepared, tokenizer, position, change):
    """Bracket a changed converted message using the actual recipe, not a local template.

    Truncated renders include a generation preamble. Their LCP with the full
    prompt is the stable boundary; do not treat their lengths as token offsets.
    Image-expanded sequences need separate span attribution and are not guessed.
    """
    if prepared.multimodal is not None or not change or change.get('field') != 'messages':
        return None
    from deepseek_recipe import Conversation, DeepseekV41Encoding
    c = prepared.conversation_request.conversation
    index = change['message_index']
    if index >= len(c.messages):
        return None
    encoding = DeepseekV41Encoding().with_tokenizer(tokenizer)
    def frontier(count):
        prefix = Conversation(c.messages[:count], thinking_mode=c.thinking_mode,
                              reasoning_effort=c.reasoning_effort, tools=c.tools,
                              tool_choice=c.tool_choice, response_format=c.response_format)
        return lcp(encoding.encode(prefix), prepared.token_ids)
    start, end = frontier(index), frontier(index + 1)
    return dict(message_index=index, role=c.messages[index].role,
                stable_prefix_before=start, stable_prefix_through=end,
                contains_mismatch=start <= position <= end,
                at_through_boundary=position == end,
                boundary_method='truncated_recipe_render_lcp')


def install(output):
    from ds41f_mlx.serving.paired_checkpoint import PairedCheckpointAuthority, image_keys
    from ds41f_mlx.serving.production_mtp import ProductionScheduler
    from ds41f_mlx.serving import server

    key = secrets.token_bytes(32)
    prepare_original = server.prepare_request

    def prepare(protocol, body, **kwargs):
        prepared = prepare_original(protocol, body, **kwargs)
        payload = json.loads(body)
        # Request-only authority fields; sampling/transport changes are not history.
        raw = {k: payload[k] for k in ('messages', 'tools', 'tool_choice',
               'response_format', 'reasoning_effort', 'thinking') if k in payload}
        prepared.prefix_structure = dict(
            wire=structural_snapshot(raw, key),
            converted=structural_snapshot(converted_payload(prepared.conversation_request.conversation), key),
            reminders=reminder_structure(prepared.conversation_request.conversation, key))
        return prepared

    server.prepare_request = prepare
    current_request = None
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
        if previous is not None and current_request is not None:
            structure = getattr(current_request.prepared, 'prefix_structure', None)
            old_structure = previous.get('structure')
            if structure is not None and old_structure is not None:
                wire_change = first_structure_change(old_structure['wire'], structure['wire'])
                converted_change = first_structure_change(old_structure['converted'], structure['converted'])
                lookup['structure'] = dict(wire_history_change=wire_change,
                    reminder_change=(reminder_change(old_structure['reminders'], structure['reminders'],
                        converted_change['message_index']) if converted_change and
                        converted_change.get('field') == 'messages' else None),
                    converted_history_change=converted_change,
                    public_history_change=first_structure_change(
                        previous.get('public_structure', old_structure['wire']), structure['wire']),
                    mismatch_region=mismatch_region(current_request.prepared,
                        current_request.recipe_tokenizer, lookup['request_lcp']['lcp'], converted_change))
        emit('lookup', request_id=active, **lookup)
        return result

    def schedule(self):
        nonlocal active, lookup, current_request
        if not self.running and self.waiting:
            current_request = self.waiting[0]
            active = current_request.request_id
            lookup = None
        return schedule_original(self)

    def retire(self, request, **kwargs):
        nonlocal previous, active, publication, current_request
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
        emit('settlement', request_id=active, finish_reason=finish,
             cached_tokens=request.cached_tokens, canonical_frontier=trace.get('canonical_frontier'),
             replay=trace.get('prompt_replay'), repack=trace.get('full_cache_repack'),
             cancelled=trace.get('cancelled'))
        if canonical is not None and len(canonical) == trace.get('canonical_frontier'):
            if (previous is not None and lookup is not None
                    and lookup.get('previous_request_id') == previous['request_id']
                    and previous['finish_reason'] == 'stop'
                    and previous['cached_tokens'] > 0 and request.cached_tokens == 0):
                emit('final_to_next_miss_candidate', request_id=active, **lookup,
                     note='Verify same OpenCode session and new user turn externally; no body recorded.')
            structure = getattr(request.prepared, 'prefix_structure', None)
            public_structure = None
            if structure is not None and choices and choices[0].get('message') is not None:
                public_structure = dict(structure['wire'])
                public_structure['messages'] = (structure['wire'].get('messages', []) +
                    structural_snapshot({'messages': [choices[0]['message']]}, key)['messages'])
            previous = dict(request_id=active, prompt=prompt, canonical=canonical,
                            finish_reason=finish, cached_tokens=request.cached_tokens,
                            structure=structure)
            if public_structure is not None:
                previous['public_structure'] = public_structure
        else:
            emit('canonical_unavailable', request_id=active)
            previous = None
        active = None
        current_request = None
        return result

    PairedCheckpointAuthority._drop_payload = drop
    PairedCheckpointAuthority.clear = clear
    PairedCheckpointAuthority.publish = publish
    PairedCheckpointAuthority.acquire = acquire
    ProductionScheduler._schedule_waiting = schedule
    ProductionScheduler._retire = retire
    emit('observer_started', history_limit=256, prompt_body_logged=False,
         structural_fingerprints_logged=False)


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
