"""Fit new tool results using actual runtime recipe admission, never char/token guesses.

Only current (not yet model-observed) fetch results may be excerpted. Effects are
already reserved/executed exactly once. Budget calls do not generate or retry them.
"""
import copy
import json
from urllib.parse import quote
from ds41f_mlx.web_client import RuntimeHTTPError
from ds41f_mlx.web_tools import ToolError


def fit_tool_results(runtime, session_id, count, request, messages, displays):
    base = copy.deepcopy(request)
    original = copy.deepcopy(messages)
    fetches = []
    for index, display in enumerate(displays):
        if display.get('tool') in {'fetch_url', 'fetch_pdf'} and not display.get('error'):
            content = original[index]['content']
            payload = json.loads(content[0]['text'] if isinstance(content, list) else content)
            if isinstance(payload.get('excerpt'), str) or any(isinstance(p.get('text'), str) and p['text'] for p in payload.get('pages', [])):
                fetches.append((index, payload))

    def candidate(size):
        out = copy.deepcopy(original)
        for index, payload in fetches:
            value = copy.deepcopy(payload)
            if 'excerpt' in payload:
                text = payload['excerpt']; excerpt = text[:size]
                value.update(excerpt=excerpt, context_truncated=len(excerpt) < len(text),
                             termination_reason='context_budget' if len(excerpt) < len(text) else payload.get('termination_reason'),
                             returned_chars=len(excerpt), next_offset=payload.get('offset', 0) + len(excerpt),
                             has_more=payload.get('total_chars', len(text)) > payload.get('offset', 0) + len(excerpt))
            else:
                for page in value['pages']:
                    text = page['text']
                    page.update(text=text[:size], context_truncated=len(text) > size,
                                returned_chars=min(len(text), size),
                                next_offset=page.get('offset', 0) + min(len(text), size))
            encoded = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
            if isinstance(out[index]['content'], list):
                # Never modify image bytes/parts, including newly rendered pages.
                out[index]['content'][0]['text'] = encoded
            else:
                out[index]['content'] = encoded
        return out

    def capacity_failure(error):
        try: return error.status == 400 and json.loads(error.body)['error']['code'] == 'context_capacity_exhausted'
        except (KeyError, TypeError, ValueError): return False

    def preview(out):
        body = dict(base, messages=[*base['messages'], *out])
        value = runtime.request('POST', f'/v1/sessions/{quote(session_id, safe="")}/budget', body)
        if value['request_count'] != count: raise ToolError('runtime frontier changed during tool budget; reconcile, no effect retry')
        return value

    # Even no-fetch batches must fit as actual full requests before consumption.
    try:
        budget = preview(original)
        return original, displays, budget
    except RuntimeHTTPError as error:
        if not capacity_failure(error) or not fetches: raise
    maximum = max(max([len(value.get('excerpt', '')), *[len(p['text']) for p in value.get('pages', [])]]) for _, value in fetches)
    minimal = candidate(0)
    budget = preview(minimal)  # if structural/non-fetch results alone cannot fit, no invented capacity
    low, high = 0, maximum
    while low + 1 < high:
        middle = (low + high) // 2
        out = candidate(middle)
        try:
            tested = preview(out)
        except RuntimeHTTPError as error:
            if not capacity_failure(error): raise
            high = middle
        else:
            low, budget = middle, tested
    out = candidate(low)
    # Every selected candidate was admitted by the exact tokenizer/expansion.
    for index, _ in fetches:
        content = out[index]['content']
        displays[index] = {**displays[index], **json.loads(content[0]['text'] if isinstance(content, list) else content)}
    return out, displays, budget
