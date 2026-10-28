"""Read-only independent joins against immutable evidence and installed checkpoint cache.
No Runtime/MLX imports, checkpoint tensors read, admission or Session writes.
"""
import gzip, hashlib, json, subprocess
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
OLD=OUT.parent/'phase4-step4b-evidence'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text())
wire=[json.loads(x) for x in (OLD/'wire.jsonl').read_text().splitlines()]
events=[x['event'] for x in wire if x['kind']=='sse_event']
t= json.loads(gzip.decompress((OLD/'trace-00.json.gz').read_bytes()))
r=load(OUT/'replay-00-1.json')
assert events==t['protocol_events']
# Match JS compact UTF-8 encoding of actual retained ASCII protocol content.
assert hashlib.sha256(json.dumps(events,separators=(',',':')).encode()).hexdigest()==r['event_sha256']
assert r['classification']=='POST_TERMINAL_CLONE_ABORT' and r['gate']['stopped'] is None
assert r['terminal']['terminal_validated'] and r['terminal']['stop_reason']=='tool_use'
assert r['terminal']['blocks'][1]['input']=={'file_path':'image-A.png'}
assert r['terminal']['usage']==next(x['usage'] for x in wire if x['kind']=='native_message_stop')
assert t['authority_connection']['disposition']=='settled'
assert t['target_offsets']==[623]*40 and t['dspark_offsets']==[623]*3
assert t['cache_published'] and t['prediction_retired'] and t['queue_empty']
saved=load(OLD/'persisted-00.json')
failed=[x for x in saved['events'] if x['type']=='tool/result']
assert len(failed)==1 and failed[0]['data']['message']['isError'] is True
assert 'PHASE4_STOP' in failed[0]['data']['message']['content'][0]['text']
for name,digest in load(OLD/'frozen-inputs.json').items(): assert sha(ROOT/name)==digest
# Baseline binds the cache receipt as one of the 817 operational records (also
# independently verified by integrity.py). Reuse its admitted file identities.
operational=Path('/Volumes/SDXC-512/ds41f-1017')
cache_path=operational/'share/ds41f-mtp/checkpoint-bytes.json'
baseline=load(ROOT/'artifacts/multimodal-runtime/phase2-step2br2-original-before.json')['files']
assert sha(cache_path)==baseline['share/ds41f-mtp/checkpoint-bytes.json']['sha256']
cache=load(cache_path)
checkpoint=Path('/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017')
admitted=load(OLD/'admission.json')['runtime']['checkpoint']
expected={s['name']:s['lfs_sha256'] for s in admitted['shards']}
assert cache['sha256']==expected and len(expected)==48
for name,identity in cache['files'].items():
 p=checkpoint/name;s=p.stat()
 assert [str(p.resolve()),s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]==identity,name
for name,digest in admitted['metadata'].items():assert sha(checkpoint/name)==digest
paths=['artifacts/multimodal-runtime/phase4-step4ar1-evidence','artifacts/multimodal-runtime/phase4-step4b-evidence','artifacts/multimodal-runtime/phase4-step4a-evidence','artifacts/multimodal-runtime/phase4-step4a-acceptance-contract.md','artifacts/multimodal-runtime/phase4-step4ar1-readiness.md','artifacts/multimodal-runtime/phase4-step4b-native-integration.md']
subprocess.run(['git','diff','--exit-code','c9ed15daf3e98a4652a4807b9906ce979be549f5','--',*paths],cwd=ROOT,check=True,capture_output=True)
result=dict(pass_=True,original_event_sequence_unchanged=True,event_count=len(events),counterfactual_classification=r['classification'],native_settlement_frontier=623,target_layers=40,dspark_rings=3,complete_pair_published=True,original_failed_tool_still_failed=True,historical_step4b='BLOCKED; no C8 PASS',checkpoint_shards_identity_unchanged=48,checkpoint_metadata_hashes_unchanged=True,checkpoint_verification='retained admitted cache (itself baseline hash verified), exact dev/inode/size/mtime/ctime; no full re-admission',checkpoint_cache_sha256=sha(cache_path),historical_tracked_files_unchanged=True,source_sha256={n:sha(OUT/n) for n in ['observer.mjs','terminal.mjs']},immutable_receipt_sha256={n:sha(OLD/n) for n in ['wire.jsonl','lifecycle.jsonl','gate.jsonl','trace-00.json.gz','persisted-00.json','observer-finding.json']})
(OUT/'offline.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS: immutable 00:1 joins, failed tool preserved, frozen inputs/history unchanged; original 48 checkpoint shard identities and metadata unchanged')
