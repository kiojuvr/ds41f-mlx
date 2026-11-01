"""Read-only verification with the installed Production converter; no generation."""
import json
from ds41f_mlx.serving.ordinary_admission import convert_ordinary
from ds41f_mlx.serving.generation_policy import normalize_policy, reasoning_policy

payload = dict(model='deepseek-v4.1-flash', messages=[dict(role='user', content='test')],
               max_tokens=128, temperature=0.0, reasoning_effort='none', stream=True,
               stream_options=dict(include_usage=True))
c, usage, auto = convert_ordinary(json.dumps(payload).encode(), default_reasoning_effort='max')
print(json.dumps(dict(sampling=normalize_policy(c, payload).to_json(),
                     reasoning=reasoning_policy(c), include_usage=usage,
                     max_tokens=c.inference_options.max_tokens), indent=2))
