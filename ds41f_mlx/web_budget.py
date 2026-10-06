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
        if display.get('tool') == 'fetch_url' and not display.get('error'):
            payload = json.loads(original[index]['content'])
            if isinstance(payload.get('excerpt'), str): fetches.append((index, payload))

    def candidate(size):
        out = copy.deepcopy(original)
        for index, payload in fetches:
            value = dict(payload)
            text = payload['excerpt']; excerpt = text[:size]
            value.update(excerpt=excerpt, context_truncated=len(excerpt) < len(text),
                         termination_reason='context_budget' if len(excerpt) < len(text) else payload.get('termination_reason'),
                         returned_chars=len(excerpt), next_offset=payload.get('offset', 0) + len(excerpt))
            out[index]['content'] = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
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
    maximum = max(len(value['excerpt']) for _, value in fetches)
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
        displays[index] = {**displays[index], **json.loads(out[index]['content'])}
    return out, displays, budget
