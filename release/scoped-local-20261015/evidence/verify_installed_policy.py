"""Release-only installed policy check; run with each deployment Python -I."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import ds41f_mlx
from ds41f_mlx.config import load_runtime_config
from ds41f_mlx.serving.server import load_v41_tokenizer, prepare_request
from ds41f_mlx.serving.generation_policy import reasoning_policy
from ds41f_mlx.serving.capacity import resolve_capacity

cfg = load_runtime_config()
tokenizer = load_v41_tokenizer(cfg.recipe_path)
def prepare(controls, stream=False, checkpoint=None, default='none'):
    body = dict(model=cfg.model_id, messages=[dict(role='user', content='Hello')], stream=stream) | controls
    return prepare_request('chat_completions', json.dumps(body).encode(), tokenizer=tokenizer,
                           checkpoint=checkpoint or cfg.checkpoint_path, ordinary=True,
                           default_reasoning_effort=default)

rows = []
for stream in (False, True):
    for effort, maximum in [('none',8192), ('low',65536), ('high',65536), ('xhigh',65536), ('max',131072)]:
        request = prepare(dict(reasoning_effort=effort), stream)
        assert request.resolved_max_tokens == maximum
        if effort == 'xhigh':
            assert reasoning_policy(request)['reasoning_effort'] == 'high'
        for explicit in (1, 128, 8192, 393216, 'auto'):
            r = prepare(dict(reasoning_effort=effort, max_tokens=explicit), stream)
            assert r.resolved_max_tokens == (393216 if explicit == 'auto' else explicit)
        rows.append(dict(stream=stream, effort=effort, omitted=maximum))
assert prepare({}).resolved_max_tokens == 8192
assert prepare({}, default='max').resolved_max_tokens == 131072
assert prepare({'reasoning_effort':'none'}, default='max').resolved_max_tokens == 8192
assert prepare({'reasoning_effort':'xhigh'}).token_ids == prepare({'reasoning_effort':'high'}).token_ids
try:
    prepare({'max_tokens':393217})
except ValueError:
    pass
else:
    raise AssertionError('ceiling did not refuse')
with tempfile.TemporaryDirectory() as directory:
    checkpoint = Path(directory)
    prompt = len(prepare({}).token_ids)
    (checkpoint/'config.json').write_text(json.dumps(dict(max_position_embeddings=prompt+8192)))
    r = prepare({}, checkpoint=checkpoint)
    r.token_ids.append(1)
    try:
        resolve_capacity(r, checkpoint=checkpoint, ordinary=True)
    except ValueError:
        pass
    else:
        raise AssertionError('expanded prompt overflow did not refuse')
    (checkpoint/'config.json').write_text(json.dumps(dict(max_position_embeddings=prompt+8191)))
    assert prepare({'max_tokens':'auto'}, checkpoint=checkpoint).resolved_max_tokens == 8191
root = Path(ds41f_mlx.__file__).parent
manifest = root/'execution_manifest.json'
print(json.dumps(dict(status='PASS', python=sys.executable, installed_root=str(root),
    manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), rows=rows,
    explicit_budgets='PASS', auto_capacity='PASS', expanded_prompt_overflow='PASS',
    headless_none_and_explicit_default_precedence='PASS', ceiling=393216), indent=2))
