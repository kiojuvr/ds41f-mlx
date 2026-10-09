"""Controlled real official-recipe/native/tokenizer boundaries, NOT sampled model."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import deepseek_recipe as d
from ds41f_mlx.serving.server import prepare_request, load_v41_tokenizer
from ds41f_mlx.serving.recovery_certificate import reconstruction_certificate
from ds41f_mlx.runtime.recipe_semantic_guard import RecipeSemanticGuard
from tools.run_m11_tool_boundary_qualification import initial_body
ROOT = Path(__file__).resolve().parents[1]
RECIPE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/tmp/ds41f-m32-recipe')
t = load_v41_tokenizer(RECIPE)
plain = dict(model='deepseek-v4.1-flash', messages=[dict(role='user', content='Say hello.')],
             reasoning_effort='none', max_tokens=32, stream=True)
tool = initial_body('Berlin'); tool['stream'] = True; tool['stop'] = None
think = dict(plain, reasoning_effort='high')
# Forced required tool choice begins *inside* tool_calls in the official parser.
call = '<｜DSML｜ invoke name="lookup_weather">\n<｜DSML｜ parameter name="city" string="true">Berlin</｜DSML｜ parameter>\n</｜DSML｜ invoke>'
rows = []
def run(name, body, source=None, ids=None, finish=None):
    body = dict(body)
    request = prepare_request('chat_completions', json.dumps(body).encode(), tokenizer=t, recipe_path=RECIPE)
    p = d.StreamProcessor(d.ChatCompletionRequest.chunk_generator(request.conversation_request, name, body['model']), request.conversation_request.parsing_options, t)
    response = d.ChatCompletionResponse(name, body['model'], 0, 0, 0)
    g = RecipeSemanticGuard(p, control_token_ids=request.stop_token_ids, response=response, frontier=len(request.token_ids))
    g._push(d.InferenceChunk.ready(prompt_usage=d.PromptUsage(prompt_tokens=len(request.token_ids), prompt_cache_hit_tokens=0)))
    candidates = t.encode(source) if ids is None else ids
    tokens = []
    for token in candidates:
        if g.finished: break
        pred = g.preview([token])
        g.observe_canonical_emit(token, None if pred is None else pred.identity)
        tokens.append(token)
    snapshot = list(p.semantic_snapshot())
    if finish and not g.finished: g.finish_backend(finish)
    response = json.loads(response.to_json())
    canonical = request.token_ids + list(tokens)
    cert = reconstruction_certificate(body, response, canonical, tokenizer=t, recipe_path=RECIPE,
        completed_tool_block=g.tool_complete)
    rows.append(dict(name=name, fixture='controlled native official recipe; no checkpoint model',
                     source=source, tokens=tokens, finish=finish, snapshot= snapshot,
                     response=response, canonical=canonical, certificate=cert))
    p.close()
for name, source, finish in [('text','Hello',None),('unicode_full','café 🙂',None),
        ('length','Hello','length'),('leading_newline','\nHello',None),
        ('marker_prefix','<｜DSML｜',None),('think_marker_prefix','</thi',None)]:
    run(name,plain,source,finish=finish)
u=t.encode('🙂');run('unicode_pending',plain,ids=u[:-1]);run('unicode_eos_pending',plain,ids=u[:-1]+list(t.encode(d.EOS_TOKEN)),finish='stop')
run('eos',plain,ids=t.encode('Hello')+list(t.encode(d.EOS_TOKEN)),finish='stop')
for name, source in [('reasoning_only','Let me think.'),('reasoning_transition','Let me think.</think>\nHello'),('reasoning_transition_exact','Let me think.</think>Hello')]:run(name,think,source)
for name, source, finish in [('tool_block_prefix','invoke ',None),('partial_name','invoke name="lookup_wea',None),
    ('partial_arguments',call[:call.index('Berlin')+2],None),
    ('valid_json_unfinished',call,None),('valid_json_length',call,'length'),
    ('complete_tool',call+'\n</｜DSML｜ calls>',None),
    ('complete_tool_alternate_spelling',call+'\n</｜DSML｜tool_calls>',None)]:run(name,tool,source,finish=finish)
full = t.encode(call+'\n</｜DSML｜ calls>')
for cut in range(len(full)-4, len(full)):
    run('valid_json_canonical_prefix_'+str(cut), tool, ids=full[:cut])
run('complete_two_tools', tool, call.replace('Berlin','Paris')+'\n'+call+'\n</｜DSML｜ calls>')
auto = dict(tool, tool_choice='auto')
run('complete_auto_tool', auto, '\n\n<｜DSML｜ calls>\n'+call+'\n</｜DSML｜ calls>')
run('enabled_marker_prefix', auto, '<｜DSML｜')
run('enabled_block_prefix', auto, '\n\n<｜DSML｜ calls>\n')
j=dict(plain,response_format=dict(type='json_object'), messages=[dict(role='user',content='Output json.')])
run('json_fence_lossy',j,'```json\n{"x":1}\n```',finish='length')
path=Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT/'artifacts/m36r/recipe-matrix.json';path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(dict(schema='ds41f.m36r.recipe-matrix.v1',rows=rows),indent=2)+'\n')
print([(r['name'],r['certificate']['representable'],r['certificate'].get('first_mismatch'),r['response']['choices']) for r in rows])
