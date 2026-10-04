"""Explicit local capability policy. No environment or request selects a profile."""
import json
import math
import re

PROFILE = 'mtp-singleton-v1'
ALIASES = ('deepseek-v4.1-flash', 'deepseek-v41-flash', 'deepseek-flash')
LIMITS = dict(body_bytes=1048576, context_tokens=8192, output_tokens=768,
              live_sessions=1, preparation_slots=1, connections=8,
              body_timeout_s=30, send_stall_s=30, effects=128, tool_result_bytes=65536)
WEATHER = {'type': 'function', 'function': {'name': 'lookup_weather',
    'description': 'Return deterministic weather for a city.', 'strict': True,
    'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}},
                   'required': ['city'], 'additionalProperties': False}}}


def strict_json(raw):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError('duplicate JSON key')
            out[k] = v
        return out
    def invalid(_):
        raise ValueError('nonfinite JSON number')
    def number(raw_number):
        value = float(raw_number)
        if not math.isfinite(value):
            raise ValueError('nonfinite JSON number')
        return value
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=invalid, parse_float=number)
    except (UnicodeError, RecursionError) as exc:
        raise ValueError('invalid bounded UTF-8 JSON') from exc
    if not isinstance(value, dict):
        raise ValueError('JSON object required')
    return value


def string(value, *, limit=1048576):
    if not isinstance(value, str) or len(value.encode('utf-8')) > limit:
        raise ValueError('bounded string required')
    if '<|' in value or '｜' in value:
        raise ValueError('raw special-token source unavailable')


def validate_chat(raw):
    b = strict_json(raw)
    allowed = {'model', 'messages', 'temperature', 'reasoning_effort', 'tools',
               'tool_choice', 'stream', 'max_tokens'}
    if set(b) - allowed:
        raise ValueError('unsupported request field')
    if b.get('model') not in ALIASES:
        raise ValueError('fixed model alias required')
    n = b.get('max_tokens')
    if type(n) is not int or not 1 <= n <= 768:
        raise ValueError('max_tokens must be integer 1..768')
    if 'temperature' in b and (type(b['temperature']) not in (int, float) or b['temperature'] != 0):
        raise ValueError('temperature must be numeric zero')
    if 'reasoning_effort' in b and b['reasoning_effort'] != 'none':
        raise ValueError('reasoning_effort must be none')
    if 'stream' in b and type(b['stream']) is not bool:
        raise ValueError('stream must be boolean')
    if 'tools' in b:
        if json.dumps(b['tools'], sort_keys=True, separators=(',', ':')) != json.dumps([WEATHER], sort_keys=True, separators=(',', ':')):
            raise ValueError('only the pinned weather declaration is supported')
        if b.get('tool_choice', 'auto') not in ('auto',) and b.get('tool_choice') != {'type':'function','function':{'name':'lookup_weather'}}:
            raise ValueError('unsupported tool_choice')
    elif 'tool_choice' in b:
        raise ValueError('tool_choice requires tools')
    messages = b.get('messages')
    if not isinstance(messages, list) or not messages:
        raise ValueError('nonempty ordinary messages required')
    pending = []
    for m in messages:
        if not isinstance(m, dict):
            raise ValueError('ordinary message object required')
        role = m.get('role')
        if pending and role != 'tool':
            raise ValueError('ordered real tool results required')
        if role in ('system', 'user'):
            if set(m) != {'role', 'content'}:
                raise ValueError('unsupported message field')
            string(m['content'])
        elif role == 'assistant':
            if set(m) - {'role', 'content', 'tool_calls'}:
                raise ValueError('unsupported assistant field')
            calls = m.get('tool_calls')
            if 'tool_calls' in m and (not isinstance(calls, list) or not 1 <= len(calls) <= 2):
                raise ValueError('one/two ordinary completed calls required')
            if m.get('content') is not None:
                string(m['content'])
            if calls:
                if 'tools' not in b or not isinstance(calls, list) or not 1 <= len(calls) <= 2:
                    raise ValueError('one/two weather calls required')
                for c in calls:
                    if not isinstance(c, dict) or set(c) != {'id','type','function'} or c['type'] != 'function':
                        raise ValueError('ordinary function call required')
                    string(c['id'], limit=256)
                    f = c['function']
                    if not isinstance(f, dict) or set(f) != {'name','arguments'} or f['name'] != 'lookup_weather':
                        raise ValueError('weather function required')
                    string(f['arguments'], limit=65536)
                    args = strict_json(f['arguments'].encode())
                    if set(args) != {'city'}:
                        raise ValueError('weather city required')
                    string(args['city'], limit=65536)
                    pending.append(c['id'])
                if len(set(pending)) != len(pending):
                    raise ValueError('duplicate tool IDs')
            elif not isinstance(m.get('content'), str):
                raise ValueError('assistant text required')
        elif role == 'tool':
            if set(m) != {'role','tool_call_id','content'} or not pending or m['tool_call_id'] != pending.pop(0):
                raise ValueError('wrong/missing/duplicate tool result')
            string(m['content'], limit=65536)
        else:
            raise ValueError('unsupported role')
    if pending or messages[-1]['role'] not in ('user','tool'):
        raise ValueError('request must finish with user or real tool results')
    return b


def validate_completed_calls(message):
    if not isinstance(message, dict) or message.get('role') != 'assistant' or set(message) - {'role','content','tool_calls'}:
        raise ValueError('ordinary text/tool assistant required')
    calls = message.get('tool_calls', [])
    if 'tool_calls' in message and (not isinstance(calls, list) or not 1 <= len(calls) <= 2):
        raise ValueError('one/two ordinary completed calls required')
    if message.get('content') is not None:
        string(message['content'])
    elif not calls:
        raise ValueError('ordinary assistant text required')
    if not isinstance(calls, list) or len(calls) > 2:
        raise ValueError('at most two weather calls supported')
    ids = []
    for call in calls:
        if not isinstance(call, dict) or set(call) != {'id','type','function'} or call['type'] != 'function':
            raise ValueError('ordinary function call required')
        string(call['id'], limit=256)
        ids.append(call['id'])
        function = call['function']
        if not isinstance(function, dict) or set(function) != {'name','arguments'} or function['name'] != 'lookup_weather':
            raise ValueError('only weather effects are supported')
        string(function['arguments'], limit=65536)
        args = strict_json(function['arguments'].encode())
        if set(args) != {'city'}:
            raise ValueError('only a city argument is supported')
        string(args['city'], limit=65536)
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate completed call IDs')


def sequence(headers):
    values = [v for k,v in headers if k.lower() == b'x-ds41f-request-sequence']
    if len(values) != 1 or not re.fullmatch(rb'[1-9][0-9]{0,19}', values[0]):
        raise ValueError('one canonical request sequence required')
    n = int(values[0])
    if n > (1 << 64)-1:
        raise ValueError('request sequence exhausted')
    return n


def certificate_projection(cert, body=None, response=None):
    if cert is None:
        return None
    # Native witness and diagnostic error text are intentionally private.
    keys = ('version','frontier','canonical_sha256','representable','executable_tools',
            'exact_prefix','semantic_complete','first_mismatch','encoded_length','profile_supported')
    out = {k:cert[k] for k in keys if k in cert}
    out['projection'] = 'ds41f.mtp.certificate.v1'
    if body is not None and response is not None and len(response.get('choices', [])) == 1:
        import hashlib
        binding = dict(messages=body['messages'], assistant=response['choices'][0]['message'])
        out['ordinary_binding_sha256'] = hashlib.sha256(json.dumps(binding, ensure_ascii=False,
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return out
