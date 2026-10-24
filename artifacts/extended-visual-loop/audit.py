"""Recompute observed evidence only; never loads weights or makes generation calls."""
from pathlib import Path
import gzip,hashlib,json,re,socket,subprocess
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
def load(n):return json.loads((OUT/n).read_text())
def lines(n):return [json.loads(l) for l in (OUT/n).read_text().splitlines()]
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,default=str)+'\n')
wire=lines('wire.jsonl');requests=[r for r in wire if r['kind']=='request' and r['path']=='/v1/messages'];prepared=lines('prepared-images.jsonl')
traces=[json.loads(gzip.decompress(p.read_bytes())) for p in sorted(OUT.glob('trace-*.raw.json.gz'))]
assert len(requests)==len(prepared)==len(traces)==6
rows=[];last=0;image_identity=None
for i,(w,p,t) in enumerate(zip(requests,prepared,traces)):
 assert w['body_sha256']==p['body_sha256']
 assert t['cached_tokens']==last
 assert p['prompt_tokens']==t['cached_tokens']+t['remaining_suffix_tokens']
 assert len(t['target_offsets'])==40 and len(t['dspark_offsets'])==3
 assert set(t['target_offsets']+t['dspark_offsets'])=={t['canonical_frontier']}
 assert t['authority_connection']['disposition']=='settled' and t['prediction_retired'] and t['queue_empty']
 assert t['prompt_replay']==t['full_cache_repack']==0
 assert t['quiescence']['counters']['history_replay']==t['quiescence']['counters']['full_cache_repack']==0
 if p['image_identities']:
  if image_identity is None:image_identity=p['image_identities']
  assert p['image_identities']==image_identity
 fresh=t.get('vision_encoded_count',0)
 hist=sum(p['image_is_historical'] or [])
 assert fresh==len(p['image_identities'])-hist
 if i<5:
  assert t['cache_published'] and not t.get('recovery_error')
  assert any(r['kind']=='native_message_stop' and r['request']==w['request'] for r in wire)
 else:
  assert not t['cache_published'] and t['response']['stop_reason']=='max_tokens'
  assert 'unfinished canonical tool protocol' in t['recovery_error']
 rows.append(dict(generation=i+1,wire=w['request'],body_sha256=w['body_sha256'],body_bytes=w['body_bytes'],prompt=p['prompt_tokens'],restored=t['cached_tokens'],suffix=t['remaining_suffix_tokens'],settled=t['canonical_frontier'],images=len(p['image_identities']),fresh_encodes=fresh,historical_reuses=hist,image_identity=p['image_identities'],historical_flags=p['image_is_historical'],patch_discount=t.get('historical_patch_discount',False),byte_guards='not invoked: total patches 0 or 4784 <=36864',cache_published=t['cache_published'],stop=t['response']['stop_reason'],recovery_error=t.get('recovery_error'),output_tokens=t['response']['usage']['output_tokens']))
 last=t['canonical_frontier']
# Join exact screenshot to official upload, File-ID, wire block, prepared digest.
png=OUT/'renders/render-00.png';digest=hashlib.sha256(png.read_bytes()).hexdigest()
u=next(r for r in wire if r.get('upload'));f=next(r for r in wire if r['kind']=='file_receipt')
assert u['upload']['sha256']==digest==image_identity[0]['sha256']
assert all(r['images'][0]['file_id']==f['file']['id'] for r in requests[3:])
assert (OUT/'workspace-final/index.html').read_text()==load('task.json')['starter']
events=lines('harness-events.jsonl');assert not any(r.get('tool') in ['write','edit'] for r in events if r['type']=='tool_call')
assert events[-2]['reason']['kind']=='error'
samples=lines('samples.jsonl');hosts=lines('host.jsonl')
reserve=[];swap=[];compressor=[];wired=[];pressure=[]
for h in hosts:
 page=int(re.search(r'page size of (\d+)',h['vm'])[1]);pages=lambda n:int(re.search(n+r':\s+(\d+)',h['vm'])[1])*page
 reserve.append(sum(pages(n) for n in ['Pages free','Pages inactive','Pages speculative']))
 compressor.append(pages('Pages occupied by compressor'));wired.append(pages('Pages wired down'))
 assert pages('Swapins')==pages('Swapouts')==0
 swap.append(float(re.search(r'used = ([\d.]+)M',h['swap'])[1]));pressure.append(int(re.search(r'free percentage: (\d+)%',h['pressure'])[1]))
assert max(swap)==0 and not (OUT/'SAFETY_STOP.json').exists()
summary=dict(classification='PARTIAL',successful_native_generations=5,successful_tool_effects=3,denied_tool_effects=2,terminal_failed_generation=6,distinct_task_screenshots=1,screenshot_occurrences_in_history=1,transmitted_occurrences_across_generations=3,image_driven_corrections=0,fresh_encodes=sum(r['fresh_encodes'] for r in rows),historical_reuses=sum(r['historical_reuses'] for r in rows),max_prompt=max(r['prompt'] for r in rows),max_settled=last,max_body=max(r['body_bytes'] for r in rows),physical_lifetime_max_bytes=max(s['lifetime_max_footprint'] for s in samples),sampled_footprint_max_bytes=max(s['footprint'] for s in samples),mlx_active_max_bytes=max(s['mlx_active'] for s in samples),mlx_peak_bytes=max(s['mlx_peak'] for s in samples),mlx_cache_max_bytes=max(s['mlx_cache'] for s in samples),mlx_active_plus_cache_max_bytes=max(s['mlx_active']+s['mlx_cache'] for s in samples),host_reserve_min_bytes=min(reserve),system_wired_max_bytes=max(wired),compressor_min_max_bytes=[min(compressor),max(compressor)],pressure_free_min_percent=min(pressure),swap_max_MiB=max(swap),cleanup_sample=samples[-1],generations=rows)
save('summary.json',summary)
partial=''.join(e.get('delta',{}).get('partial_json','') for e in traces[-1]['protocol_events'])
try:
 json.loads(partial)
 raise AssertionError('expected truncated edit arguments')
except json.JSONDecodeError as e:
 save('terminal.json',dict(layer='model output budget / fail-closed canonical tool settlement',reservation=1000,stop_reason=traces[-1]['response']['stop_reason'],recovery_error=traces[-1]['recovery_error'],json_error=str(e),partial_arguments=partial,tool_dispatched=False,canonical_settled=True,cache_published=False,cancelled=traces[-1]['cancelled'],native_message_stop=False,harness_error=events[-2]['reason']))
# Read-only final host/process/deployed inventory and owned cleanup verification.
commands={'ps':['ps','-axo','pid,ppid,rss,command'],'vm':['vm_stat'],'swap':['sysctl','vm.swapusage'],'pressure':['memory_pressure','-Q']}
host={k:subprocess.check_output(v,text=True) for k,v in commands.items()}
with socket.socket() as s:assert s.connect_ex(('127.0.0.1',18083))!=0
execution=load('execution.json');final=load('final-state.json')
assert execution['runtime_exit']==0 and execution['harness_exit']==1 and final['renderer_exit']==0 and final['removed']
for pid in [execution['runtime_pid'],execution['harness_pid'],execution['renderer_supervisor_pid']]:
 assert subprocess.run(['kill','-0',str(pid)],capture_output=True).returncode!=0
before=load('host-before.json')['deployed_manifests']
assert len(before)==2
for p,x in before.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==x['sha256']
host.update(deployed_manifests_unchanged=True,owned_processes_exited=True,port_free=True,temporary_workspace_removed=not Path(final['temporary_root']).exists())
if not (OUT/'host-after.json').exists():save('host-after.json',host)
print(json.dumps({k:v for k,v in summary.items() if k not in ['generations','cleanup_sample']},indent=2))
