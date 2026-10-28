"""Focused replay against the ACTUAL durable multimodal Session; no native work."""
import gzip,hashlib,importlib.util,json,subprocess
from pathlib import Path
OUT=Path(__file__).resolve().parent
TEMP=Path(json.loads((OUT/'location.json').read_text())['temp'])
SID=json.loads((OUT/'resources.json').read_text())['session_id']
def inventory():return {str(p.relative_to(TEMP/'home/sessions')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (TEMP/'home/sessions').rglob('*') if p.is_file() and p.suffix!='.lock'}
before=inventory()
old=subprocess.run(['node',str(OUT/'persisted-executed.mjs'),SID,'must-not-write'],capture_output=True,text=True,timeout=30)
assert old.returncode!=0 and 'requires a message projection' in old.stderr
for _ in range(2):
 result=subprocess.run(['node',str(OUT/'persisted.mjs'),SID,'01-postrun'],capture_output=True,text=True,timeout=30)
 assert result.returncode==0,result.stderr
assert before==inventory(),'read-only inspection changed durable Session'
persisted=json.loads((OUT/'persisted-01-postrun.json').read_text());events=persisted['events']
assert len(events)==65 and sum(e['type']=='image/offload' for e in events)==3
assert sum(e['type']=='tool/result' and not e['data']['message'].get('isError',False) for e in events)==6
spec=importlib.util.spec_from_file_location('safe',OUT.parent/'phase4-step4ar1-evidence/safety.py');safe=importlib.util.module_from_spec(spec);spec.loader.exec_module(safe)
preps=[json.loads(l) for l in (OUT/'prepared.jsonl').read_text().splitlines()]
manifest=json.loads((OUT.parent/'phase4-step4a-evidence/images.json').read_text());by_hash={r['sha256']:r['label'] for r in manifest}
joins=[]
for i,p in enumerate(preps):
 t=json.loads(gzip.decompress((OUT/f'trace-{i:02}.json.gz').read_bytes()))
 assert safe.settled_trace_stop(p,t) is None
 if i>=5:
  assert safe.settled_trace_stop(p,t,transition='offload',previous_tokens=preps[i-1]['token_ids']) is None
  joins.append(dict(request_index=i,C=t['cached_tokens'],N=len(p['token_ids']),historical=p['image_is_historical'],fresh=t['vision_encoded_count'],current_sources=[by_hash[im['sha256']] for im in p['image_identities']],all_image_roots_incompatible=all(not r['exact_prefix'] for r in p['available_paired_roots'] if r['images']),compatible_text_roots=[r['frontier'] for r in p['available_paired_roots'] if r['exact_prefix'] and not r['images']]))
assert [r['current_sources'] for r in joins]==[['B','C','D'],['C','D','E'],['D','E','F']]
assert all(r['all_image_roots_incompatible'] for r in joins)
attachments=[]
for e in events:
 if e['type']!='tool/result':continue
 for b in e['data']['message']['content']:
  if b['type']!='image':continue
  a=b['attachment'];digest=a['attachmentId'].removeprefix('sha256:')
  data=(TEMP/'home/attachments/v1/objects'/digest[:2]/digest).read_bytes()
  assert hashlib.sha256(data).hexdigest()==digest and len(data)==a['bytes']
  attachments.append(dict(label=by_hash[digest],sha256=digest,tool_id=e['data']['message']['toolCallId'],result_seq=e['seq']))
assert [a['label'] for a in attachments]==list('ABCDEF')
report=dict(attachments=attachments,pass_=True,scope='READ_ONLY_POST_SHUTDOWN_NO_NATIVE_CONTINUATION',original_inspector_reproduced_failure=True,corrected_inspector_two_successes=True,durable_bytes_unchanged=True,durable_files=before,events=65,offload_joins=joins,production_changes=False)
(OUT/'postrun-checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
