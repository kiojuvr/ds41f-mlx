"""Recompute actual post-run receipts; no generation requests or model imports."""
from pathlib import Path
import gzip,hashlib,json,re,socket,subprocess
from html.parser import HTMLParser
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
def load(n):return json.loads((OUT/n).read_text())
def lines(n):return [json.loads(l) for l in (OUT/n).read_text().splitlines()]
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,default=str)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
wire=lines('wire.jsonl'); requests=[r for r in wire if r['kind']=='request' and r['path']=='/v1/messages'];prepared=lines('prepared-images.jsonl')
traces=[json.loads(gzip.decompress(p.read_bytes())) for p in sorted(OUT.glob('trace-*.raw.json.gz'))]
assert len(requests)==len(prepared)==len(traces)+1==28, '27 completions plus one admission rejection'
assert 'paired-byte qualification fallback' in (OUT/'runtime.log').read_text()
assert load('trace-retention.json')['status']=='FAIL'
assert load('trace-retention.json')['archived']==27
# The pre-generation rejection has no archived native trace. Do not invent one.
assert requests[-1]['body_sha256']==prepared[-1]['body_sha256']
assert len(prepared[-1]['image_identities'])==8
save('terminal.json',dict(request=28,wire=requests[-1]['request'],layer='paired-byte conditional admission fallback',error='aggregate image patch budget exceeded (paired-byte qualification fallback)',prompt_positions=prepared[-1]['prompt_tokens'],output_reservation=requests[-1]['max_tokens'],images=8,total_patches=8*4784,fresh_patches=4784,historical_flags=prepared[-1]['image_is_historical'],last_successful_frontier=traces[-1]['canonical_frontier'],native_generation=False,publication=False,exact_projected_and_retained_bytes='not retained; which byte ceiling failed cannot be separated',trace_retention=load('trace-retention.json'),retry=False))
rows=[];last=0;registry=None
for i,(w,p,t) in enumerate(zip(requests,prepared,traces)):
 assert w['body_sha256']==p['body_sha256']
 assert w['max_tokens']==2600
 if registry is None:registry=w['tools']
 assert registry==w['tools']
 assert p['prompt_tokens']==t['cached_tokens']+t['remaining_suffix_tokens']
 assert len(t['target_offsets'])==40 and len(t['dspark_offsets'])==3
 assert set(t['target_offsets']+t['dspark_offsets'])=={t['canonical_frontier']}
 assert t['authority_connection']['disposition']=='settled' and t['prediction_retired'] and t['queue_empty']
 assert t['prompt_replay']==t['full_cache_repack']==0
 assert t['quiescence']['counters']['history_replay']==t['quiescence']['counters']['full_cache_repack']==0
 fresh=t.get('vision_encoded_count',0);hist=sum(p['image_is_historical'] or [])
 assert fresh==len(p['image_identities'])-hist
 rows.append(dict(generation=i+1,wire=w['request'],body_sha256=w['body_sha256'],body_bytes=w['body_bytes'],output_reservation=w['max_tokens'],prompt=p['prompt_tokens'],restored=t['cached_tokens'],previous_frontier=last,exact_previous_frontier_restore=t['cached_tokens']==last,suffix=t['remaining_suffix_tokens'],settled=t['canonical_frontier'],images=len(p['image_identities']),fresh_encodes=fresh,historical_reuses=hist,image_identities=p['image_identities'],historical_flags=p['image_is_historical'],patch_discount=t.get('historical_patch_discount',False),retained_pair_bytes=t.get('retained_pair_bytes'),projected_pair_bytes=t.get('projected_pair_bytes'),cache_published=t['cache_published'],stop=t['response']['stop_reason'],recovery_error=t.get('recovery_error'),output_tokens=t['response']['usage']['output_tokens']))
 last=t['canonical_frontier']
# Join PNG -> upload bytes -> Files receipt -> actual nested Messages references.
uploads=[r for r in wire if r.get('upload')];receipts=[r for r in wire if r['kind']=='file_receipt'];screens=[]
for png in sorted((OUT/'renders').glob('render-*.png')):
 digest=sha(png);up=[u for u in uploads if u['upload']['sha256']==digest]
 ids=[r['file']['id'] for r in receipts if any(u['request']==r['request'] for u in up)]
 occurrences=[(i+1,j) for i,p in enumerate(prepared) for j,x in enumerate(p['image_identities']) if x['sha256']==digest]
 wire_refs=sum(im.get('file_id') in ids for w in requests for im in w['images'])
 screens.append(dict(file=str(png.relative_to(OUT)),sha256=digest,bytes=png.stat().st_size,file_ids=ids,prepared_generation_and_index=occurrences,wire_file_references=wire_refs))
 for u in up:assert u['upload']['width']==1280 and u['upload']['height']==720
 if occurrences:assert ids and wire_refs==len(occurrences)
# Frozen content compared independently, not fed back to the agent.
class Text(HTMLParser):
 def __init__(self):super().__init__();self.body=False;self.text=[]
 def handle_starttag(self,t,a):
  if t=='body':self.body=True
 def handle_endtag(self,t):
  if t=='body':self.body=False
 def handle_data(self,s):
  if self.body and s.strip():self.text.append(' '.join(s.split()))
def bodytext(s):
 p=Text();p.feed(s);return p.text
baseline=bodytext(load('task.json')['starter'])
content=[dict(file=str(p.relative_to(OUT)),body_text_preserved=bodytext(p.read_text())==baseline) for p in sorted((OUT/'renders').glob('artifact-*.html'))]
save('content-preservation.json',content)
samples=lines('samples.jsonl');hosts=lines('host.jsonl');reserve=[];pressure=[];swap=[]
for h in hosts:
 page=int(re.search(r'page size of (\d+)',h['vm'])[1]);pages=lambda n:int(re.search(n+r':\s+(\d+)',h['vm'])[1])*page
 reserve.append(sum(pages(n) for n in ['Pages free','Pages inactive','Pages speculative']))
 assert pages('Swapins')==pages('Swapouts')==0
 swap.append(float(re.search(r'used = ([\d.]+)M',h['swap'])[1]));pressure.append(int(re.search(r'free percentage: (\d+)%',h['pressure'])[1]))
summary=dict(real_generations=len(traces),native_completions=sum(r['kind']=='native_message_stop' for r in wire),distinct_rendered_screenshots=len({s['sha256'] for s in screens}),distinct_transported_screenshots=len({s['sha256'] for s in screens if s['prepared_generation_and_index']}),distinct_vision_encoded_screenshots=len({x['sha256'] for r in rows for x,h in zip(r['image_identities'],r['historical_flags'] or []) if not h}),max_successful_history_image_occurrences=max((r['images'] for r in rows),default=0),max_attempted_history_image_occurrences=8,fresh_encodes=sum(r['fresh_encodes'] for r in rows),historical_reuses=sum(r['historical_reuses'] for r in rows),discounted_generations=[r['generation'] for r in rows if r['patch_discount']],paired_integrity=True,all_exact_previous_frontier_restores=all(r['exact_previous_frontier_restore'] for r in rows),all_published=all(r['cache_published'] for r in rows),all_body_text_preserved=all(r['body_text_preserved'] for r in content),tool_registry=registry,total_output_tokens=sum(r['output_tokens'] for r in rows),max_successful_prompt=max((r['prompt'] for r in rows),default=0),rejected_prompt=prepared[-1]['prompt_tokens'],max_settled=last,max_body_bytes=max((r['body_bytes'] for r in rows),default=0),physical_lifetime_max_bytes=max(s['lifetime_max_footprint'] for s in samples),sampled_footprint_max_bytes=max(s['footprint'] for s in samples),mlx_peak_active_bytes=max(s['mlx_peak'] for s in samples),mlx_active_plus_cache_max_bytes=max(s['mlx_active']+s['mlx_cache'] for s in samples),host_reserve_min_bytes=min(reserve),pressure_free_min_percent=min(pressure),swap_max_MiB=max(swap),safety_stop=load('SAFETY_STOP.json') if (OUT/'SAFETY_STOP.json').exists() else None,post_terminal_observer_aborts=sum(r['kind']=='observer_read_error' and r.get('message_stop_already_captured',False) for r in wire),rejections=[r for r in wire if r['kind'] in ['http_error','budget_stop','transport_error'] or (r['kind']=='observer_read_error' and not r.get('message_stop_already_captured',False))],screenshots=screens,generations=rows)
save('summary.json',summary)
# Owned cleanup, without touching unrelated processes or installations.
execution=load('execution.json');final=load('final-state.json')
assert execution['runtime_exit']==1 and execution['harness_exit']==1 and final['renderer_exit']==0 and final['removed']
assert load('runtime-exit.json')['server_returned']
for k in ['runtime_pid','harness_pid','renderer_supervisor_pid']:
 assert subprocess.run(['kill','-0',str(execution[k])],capture_output=True).returncode!=0
with socket.socket() as s:assert s.connect_ex(('127.0.0.1',18083))!=0
for p,x in load('host-before.json')['deployed_manifests'].items():assert sha(Path(p))==x['sha256']
assert not Path(final['temporary_root']).exists()
host={k:subprocess.check_output(v,text=True) for k,v in dict(vm=['vm_stat'],swap=['sysctl','vm.swapusage'],pressure=['memory_pressure','-Q']).items()}
host.update(deployed_manifests_unchanged=True,owned_processes_exited=True,port_free=True,temporary_workspace_removed=True)
if not (OUT/'host-after.json').exists():save('host-after.json',host)
print(json.dumps({k:v for k,v in summary.items() if k not in ['generations','screenshots','tool_registry']},indent=2))
