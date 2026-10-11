"""Frontend negotiation only; no execution policy or server normalization."""

LEGACY_DEFAULTS = dict(temperature=1.0, top_p=.95, reasoning_effort='none', max_tokens='auto')
PRODUCTION_DEFAULTS = dict(temperature=0, top_p=1, reasoning_effort='none', max_tokens='auto')


def negotiate(runtime):
    if runtime.get('status') != 'ready':
        raise ValueError('Runtime is not READY: ' + str(runtime.get('status')))
    capability = (runtime.get('capabilities') or {}).get('stateful_chat')
    if capability == 'production-sessions-v1':
        defaults = runtime.get('chat_defaults')
        if defaults != PRODUCTION_DEFAULTS:
            raise ValueError('Unsupported production chat_defaults; no silent normalization')
        return dict(defaults)
    if capability is None and runtime.get('profile') == 'standard-off':
        # Explicit compatibility with the existing standard-OFF contract.
        return dict(LEGACY_DEFAULTS)
    raise ValueError('Unsupported runtime stateful_chat capability: ' + str(capability))
