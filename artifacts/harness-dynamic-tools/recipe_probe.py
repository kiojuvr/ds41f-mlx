"""Direct native Recipe probe; explicit wire fixtures, not Harness serialization/E2E."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import deepseek_recipe as d
import deepseek_recipe._native as native

projection = json.loads(Path(sys.argv[1]).read_text())
options = d.ConversionOptions(default_thinking_mode=False).with_model_versions_by_name(
    {'deepseek-flash': d.ModelVersion.DeepseekV41Flash})
rows = []
for mode, projected in projection.items():
    # Deliberately explicit equivalent fixtures for this one projection; no general adapter.
    tools = [dict(name=t['name'], description=t['description'], input_schema=t['parameters'],
                  **({'defer_loading': True} if t.get('deferLoading') else {}))
             for t in projected['tools']]
    updates = [dict(type='tool_addition', tool=dict(type='tool_reference', name='fetch'))]
    if mode == 'in-history':
        updates.append(dict(type='tool_removal', tool=dict(type='tool_reference', name='search')))
    body = dict(model='deepseek-flash', max_tokens=64, thinking={'type': 'disabled'}, tools=tools,
                messages=[dict(role='user', content='before'), dict(role='system', content=updates),
                          dict(role='user', content='after')])
    row = dict(mode=mode, fixture=body)
    try:
        request = d.MessagesRequest(json.dumps(body).encode())
        row['deserialized'] = True
        converted = request.convert(options)
        row['prompt'] = d.DeepseekV41Encoding().render_conversation(converted.conversation).prompt
    except Exception as exc:
        row['error_type'] = type(exc).__name__
        row['error'] = str(exc)
    if mode == 'addition-only' and '--expect-all-deferred' in sys.argv[2:]:
        assert 'error' not in row, row
        initial, activated = row['prompt'].split('### Updated Tool Schemas')
        assert '### Available Tool Schemas' in initial
        assert '"name": "fetch"' not in initial
        assert '"name": "fetch"' in activated and '"name": "search"' not in row['prompt']
    elif mode == 'addition-only':
        assert row.get('deserialized') and 'All tools cannot be deferred' in row['error']
    else:
        assert not row.get('deserialized') and 'tool_removal' in row['error']
    rows.append(row)
print(json.dumps(dict(python=sys.version, executable=sys.executable,
    recipe_version=importlib.metadata.version('deepseek-recipe'),
    native_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(), rows=rows), indent=2))
