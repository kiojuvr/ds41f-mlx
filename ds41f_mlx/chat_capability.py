"""Frontend contract consumption, never sampler/default authority.

Application preferences are sent explicitly. Runtime omission semantics are not
inheritance: a tool workflow keeps its application snapshot, not server defaults.
"""
from copy import deepcopy
import math

PARAMETERS = ('temperature', 'top_p', 'top_k', 'min_p', 'repetition_penalty',
              'presence_penalty', 'frequency_penalty', 'seed')


def vision_available(runtime):
    """Execution capability, not application preview or retained-state authority."""
    return (runtime.get('status') == 'ready' and
            (runtime.get('capabilities') or {}).get('chat_vision') == 'bounded-inline-v1')


def negotiate(runtime):
    if runtime.get('status') != 'ready':
        raise ValueError('Runtime is not READY: ' + str(runtime.get('status')))
    capability = (runtime.get('capabilities') or {}).get('stateful_chat')
    if not (capability == 'production-sessions-v1' or
            (capability is None and runtime.get('profile') == 'standard-off')):
        raise ValueError('Unsupported runtime stateful_chat capability: ' + str(capability))
    contract = runtime.get('generation_policy')
    defaults = runtime.get('chat_defaults')
    if (not isinstance(contract, dict) or contract.get('version') != 'generation-policy-v1'
            or contract.get('scope') != 'request'
            or set(contract.get('parameters', {})) != set(PARAMETERS)
            or set(contract.get('defaults', {})) != set(PARAMETERS)
            or not isinstance(defaults, dict)):
        raise ValueError('Unsupported generation policy contract')
    reasoning = contract.get('reasoning', {})
    if (reasoning.get('scope') != 'session' or reasoning.get('stateless_scope') != 'request'
            or reasoning.get('change') != 'reject' or reasoning.get('thinking_budget_tokens') is not False
            or reasoning.get('default_thinking_mode') is not (defaults.get('reasoning_effort') != 'none')
            or not isinstance(reasoning.get('efforts'), list) or 'none' not in reasoning['efforts']):
        raise ValueError('Unsupported reasoning ownership contract')
    if (set(defaults) != set(PARAMETERS) | {'reasoning_effort', 'max_tokens'}
            or any(defaults[key] != value for key, value in contract['defaults'].items())
            or defaults['reasoning_effort'] != reasoning.get('default_reasoning_effort', 'none')):
        raise ValueError('Generation defaults disagree with runtime authority')
    validate(contract, defaults)
    return deepcopy(defaults)


def validate(contract, values):
    for name in PARAMETERS:
        spec, value = contract['parameters'][name], values[name]
        if name == 'seed' and value is None:
            continue
        try:
            finite = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or (spec.get('type') == 'integer' and type(value) is not int):
            raise ValueError(name + ' must be a finite ' + spec.get('type', 'number'))
        for bound, invalid in (('minimum', lambda b: value < b), ('maximum', lambda b: value > b),
                               ('exclusiveMinimum', lambda b: value <= b)):
            if bound in spec and invalid(spec[bound]):
                raise ValueError(name + ' out of runtime range')
    if values['reasoning_effort'] not in contract['reasoning']['efforts']:
        raise ValueError('Unsupported reasoning effort')
