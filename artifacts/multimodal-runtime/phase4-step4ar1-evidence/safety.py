"""Task-owned pure Supervisor/preparation predicates derived from Step3A/R3.
No model loader, process launcher, Scheduler or shutdown owner. The existing
external runner supplies real sampled/progress inputs and owns process retirement.
"""
import json,time
from pathlib import Path
B=json.loads((Path(__file__).resolve().parent.parent/'phase4-step4a-evidence/budgets.json').read_text());G=1024**3

def resource_stop(sample,baseline,*,loaded,overall_s,session_s,last_progress_s,generation_age_s=None,session_bytes=0,attachment_bytes=0):
 if sample['competitors']:return 'competing model worker'
 if sample['pressure']!=1:return 'memory pressure'
 if sample['swap_used']>baseline['swap_used'] or sample['swapouts']>baseline['swapouts']:return 'swap growth'
 if sample['available']<=64*G:return 'host reserve'
 peak=max(sample['footprint'],sample['lifetime_footprint'])
 if peak>=430*G:return 'absolute physical ceiling'
 if peak>=425*G:return 'physical-footprint stop'
 if not loaded and sample['available']<455.226*G:return 'preload available memory'
 if overall_s>=5400:return 'overall duration'
 if not loaded and overall_s>=1200:return 'startup duration'
 if session_s is not None and session_s>=4200:return 'session duration'
 # Existing retire() grants up to 90s to each of two process groups. Reserve
 # those 180s INSIDE the frozen 600s overhead, not beyond either wall ceiling.
 if overall_s>=5400-180 or (session_s is not None and session_s>=4200-180):return 'shutdown reserve inside frozen wall budget'
 if last_progress_s>=600:return 'no semantic/native progress'
 if generation_age_s is not None and generation_age_s>=180:return 'generation duration'
 if session_bytes>67108864:return 'session storage'
 if attachment_bytes>8388608:return 'attachment storage'
 return None

class ProgressClock:
 """Semantic/native receipt clock, not log-size or heartbeat activity."""
 KINDS=frozenset(['installed_proof','model_loaded','prepared','native_frontier','canonical_output','request_retired','persisted'])
 def __init__(self,now):self.last=now;self.key=None
 def note(self,kind,key,now):
  if kind not in self.KINDS or (kind,key)==self.key:return False
  self.key=(kind,key);self.last=now;return True
 def age(self,now):return now-self.last

class LaunchBudget:
 """Finite task counters only; no Session history or process/shutdown owner."""
 def __init__(self):self.loads=0;self.invocations=0;self.active=False;self.session_id=None;self.stopped=False
 def authorize_load(self):
  if self.loads or self.stopped:raise RuntimeError('one load; no reload/restart')
  self.loads=1
 def begin_invocation(self,session_id):
  if self.stopped or self.active or self.invocations>=9 or (self.invocations and session_id!=self.session_id) or (not self.invocations and session_id is not None):raise RuntimeError('exclusive one-session/nine-invocation budget')
  self.invocations+=1;self.active=True
 def end_invocation(self,session_id,*,success):
  if not self.active or not session_id or (self.session_id and session_id!=self.session_id):raise RuntimeError('invalid official session adoption receipt')
  self.session_id=session_id;self.active=False
  if not success:self.stopped=True

def admit_prepared(p,output):
 """Invoke in existing worker prepare observer before enqueue/native work."""
 count=0 if p.multimodal is None else len(p.multimodal.spans)
 if len(p.token_ids)+output>32768 or count>3:
  if p.multimodal is not None:p.multimodal.release_sources()
  raise ValueError('frozen Phase4 prepared-plus-output/current-image envelope')
 return p

def checked_prepare(original,*args,output,**kwargs):
 return admit_prepared(original(*args,**kwargs),output)

def settled_trace_stop(prepared,trace,*,transition=None,previous_tokens=None):
 """Existing Step3A audit predicates at an already-retired safe boundary.
 Consumes passive receipts only; never touches cache/native execution or effects.
 """
 try:
  f=trace['canonical_frontier'];C=trace['cached_tokens']
  if trace['protocol_events'][-1]['type']!='message_stop':return 'incomplete native protocol'
  if not trace['cache_published'] or not trace['prediction_retired'] or not trace['queue_empty'] or trace['cancelled']:return 'unsettled/cancelled/unpublished outcome'
  if len(trace['target_offsets'])!=40 or len(trace['dspark_offsets'])!=3 or set(trace['target_offsets']+trace['dspark_offsets'])!={f}:return 'incomplete paired frontier'
  if trace['authority_connection']['disposition']!='settled':return 'canonical authority not settled'
  if trace['prompt_replay'] or trace['full_cache_repack']:return 'historical replay/repack'
  images=prepared['image_identities'];fresh=[im for im in images if im['start']>=C]
  if trace.get('vision_encoded_count',0)!=len(fresh):return 'fresh/covered Vision disagreement'
  if transition in ('offload','compaction'):
   divergence=next((i for i,(a,b) in enumerate(zip(previous_tokens,prepared['token_ids'])) if a!=b),min(len(previous_tokens),len(prepared['token_ids'])))
   if previous_tokens==prepared['token_ids'] or C>divergence:return 'unchanged projection/unexpected cache match'
   if transition=='offload' and not any(prepared['image_is_historical'][j] and im in fresh for j,im in enumerate(images)):return 'offload lacks fresh surviving historical Vision'
 except (KeyError,IndexError,TypeError):return 'missing required native trace evidence'
 return None

# The existing runner must atomically publish this after applying resource_stop
# at each <=2s sample. A stale/missing receipt makes the Node task guard fail shut.
def publish_receipt(path,*,stop,session_bytes,attachment_bytes,persistence_ok=False):
 p=Path(path);tmp=p.with_suffix('.tmp')
 tmp.write_text(json.dumps(dict(at_ms=int(time.time()*1000),stop=stop,session_bytes=session_bytes,attachment_bytes=attachment_bytes,persistence_ok=persistence_ok)))
 tmp.replace(p)
