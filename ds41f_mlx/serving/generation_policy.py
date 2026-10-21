"""Request policy normalization; sampling uses the attributed owned source copy."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class GenerationPolicy:
    # Official DeepSeek-V4.1-Flash recommendation; greedy is explicit only.
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None

    def to_json(self):
        return asdict(self)

    def processors(self):
        from mlx_lm.sample_utils import make_logits_processors
        return make_logits_processors(
            repetition_penalty=None if self.repetition_penalty == 1 else self.repetition_penalty,
            presence_penalty=None if self.presence_penalty == 0 else self.presence_penalty,
            frequency_penalty=None if self.frequency_penalty == 0 else self.frequency_penalty)

    def sampling_params(self, maximum):
        from omlx.request import SamplingParams
        return SamplingParams(max_tokens=maximum, **self.to_json())


def with_sampling_defaults(payload):
    """Resolve omission before recipe conversion, retaining its numeric authority."""
    if not isinstance(payload, dict):
        return payload  # Let the protocol converter reject the original shape.
    defaults = GenerationPolicy()
    return payload | {name: getattr(defaults, name) for name in ('temperature', 'top_p')
                      if payload.get(name) is None}


def normalize_policy(converted, payload):
    thinking = payload.get('thinking')
    if isinstance(thinking, dict) and thinking.get('budget_tokens') is not None:
        raise ValueError('thinking budget_tokens is unsupported')
    options = converted.inference_options
    values = GenerationPolicy().to_json()
    for name in values:
        value = getattr(options, name, None) if name in ('temperature', 'top_p') else payload.get(name)
        if value is not None:
            values[name] = value
    for name, low, high in (('temperature', 0, 2), ('top_p', 0, 1),
                            ('min_p', 0, 1), ('repetition_penalty', 0, None),
                            ('presence_penalty', -2, 2), ('frequency_penalty', -2, 2)):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f'{name} must be a finite number')
        if value < low or (high is not None and value > high) or (name in ('top_p', 'repetition_penalty') and value == 0):
            raise ValueError(f'{name} out of range')
    # Fixed DeepSeek-V4.1 checkpoint vocab_size=129280; upstream apply_top_k
    # requires k < vocab_size. Reject before any executable cache mutation.
    if type(values['top_k']) is not int or not 0 <= values['top_k'] < 129280:
        raise ValueError('top_k must be an integer in [0, 129279]')
    seed = values['seed']
    if seed is not None and (type(seed) is not int or not 0 <= seed <= 2**32-1):
        raise ValueError('seed must be an integer in [0, 4294967295]')
    return GenerationPolicy(**values)


def policy_for(request):
    return getattr(request, 'generation_policy', None) or normalize_policy(request, {})


def reasoning_policy(request):
    """Canonical effective state after recipe conversion, not wire spellings.

    The recipe already normalizes minimal/medium. Its retained xhigh spelling
    is DeepSeek's compatibility alias for high, not a separate model effort.
    Accept both converted and prepared requests so production policy shares
    this authority without changing the recipe's protocol/rendering objects.
    """
    converted = getattr(request, 'conversation_request', request)
    conversation = converted.conversation
    effort = conversation.reasoning_effort
    if effort == 'xhigh':
        effort = 'high'
    return dict(thinking_mode=conversation.thinking_mode,
                reasoning_effort=effort)


def capability_contract(default_reasoning_effort='none'):
    # Bounded MTP real-model acceptance; not distributional equivalence.
    return dict(version='generation-policy-v1', qualification='real-model-qualified',
                execution_profiles=['mtp-serving-v1'],
                sampler_authority='ds41f_mlx.sampling.make_sampler',
                defaults=GenerationPolicy().to_json(), scope='request',
                parameters=dict(temperature={'minimum': 0, 'maximum': 2},
                    top_p={'exclusiveMinimum': 0, 'maximum': 1},
                    top_k={'type': 'integer', 'minimum': 0, 'maximum': 129279}, min_p={'minimum': 0, 'maximum': 1},
                    repetition_penalty={'exclusiveMinimum': 0, 'context_tokens': 20},
                    presence_penalty={'minimum': -2, 'maximum': 2, 'context_tokens': 20},
                    frequency_penalty={'minimum': -2, 'maximum': 2, 'context_tokens': 20},
                    seed={'type': 'integer', 'minimum': 0, 'maximum': 2**32-1, 'semantics': 'best-effort'}),
                reasoning=dict(authority='deepseek-recipe', scope='session', stateless_scope='request', default_thinking_mode=default_reasoning_effort != 'none',
                               default_reasoning_effort=default_reasoning_effort,
                               efforts=['none', 'low', 'high', 'xhigh', 'max'],
                               effective_efforts=['none', 'low', 'high', 'max'],
                               compatibility_aliases={'xhigh': 'high'},
                               effective_authority='ds41f_mlx.serving.generation_policy.reasoning_policy',
                               change='reject', thinking_budget_tokens=False))
