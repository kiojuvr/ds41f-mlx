import json,sys
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from safety import resource_stop,admit_prepared,publish_receipt,LaunchBudget,ProgressClock,settled_trace_stop,G
OUT=Path(__file__).resolve().parent

def test_supervisor_stops():
 base=dict(available=480*G,pressure=1,swap_used=0,swapouts=0,footprint=300*G,lifetime_footprint=301*G,competitors=[])
 kwargs=dict(loaded=True,overall_s=50,session_s=20,last_progress_s=0)
 assert resource_stop(base,base,**kwargs) is None
 cases=[('preload available memory',dict(available=455.225*G),dict(loaded=False)),('host reserve',dict(available=64*G),{}),('memory pressure',dict(pressure=2),{}),('swap growth',dict(swap_used=1),{}),('swap growth',dict(swapouts=1),{}),('physical-footprint stop',dict(footprint=425*G),{}),('absolute physical ceiling',dict(lifetime_footprint=430*G),{}),('competing model worker',dict(competitors=[123]),{}),('overall duration',{},dict(overall_s=5400)),('startup duration',{},dict(loaded=False,overall_s=1200)),('session duration',{},dict(session_s=4200)),('shutdown reserve inside frozen wall budget',{},dict(session_s=4020)),('shutdown reserve inside frozen wall budget',{},dict(overall_s=5220)),('no semantic/native progress',{},dict(last_progress_s=600)),('generation duration',{},dict(generation_age_s=180)),('session storage',{},dict(session_bytes=67108865)),('attachment storage',{},dict(attachment_bytes=8388609))]
 for reason,changes,k in cases:assert resource_stop(dict(base,**changes),base,**dict(kwargs,**k))==reason
 (OUT/'supervisor-tests.json').write_text(json.dumps(dict(pass_=True,cases=[c[0] for c in cases],shutdown='return stop to existing runner: stop new work, existing cancellation/lifespan then bounded SIGTERM/SIGKILL; no reload or new shutdown owner',weights_loaded=0),indent=2)+'\n')

def test_preparation_retires_sources_before_refusal():
 released=[];mm=NS(spans=[1,2,3],release_sources=lambda:released.append(True))
 admit_prepared(NS(token_ids=range(31568),multimodal=mm),1200)
 with pytest.raises(ValueError):admit_prepared(NS(token_ids=range(31569),multimodal=mm),1200)
 assert released==[True]
 with pytest.raises(ValueError):admit_prepared(NS(token_ids=range(30721),multimodal=None),2048)

def test_launch_budget_no_replacement_or_parallelism():
 b=LaunchBudget();b.authorize_load()
 with pytest.raises(RuntimeError):b.authorize_load()
 b.begin_invocation(None)
 with pytest.raises(RuntimeError):b.begin_invocation(None)
 b.end_invocation('official-id',success=True)
 with pytest.raises(RuntimeError):b.begin_invocation('replacement-id')
 for i in range(8):b.begin_invocation('official-id');b.end_invocation('official-id',success=True)
 with pytest.raises(RuntimeError):b.begin_invocation('official-id')
 b=LaunchBudget();b.begin_invocation(None);b.end_invocation('failed-official-id',success=False)
 with pytest.raises(RuntimeError):b.begin_invocation('failed-official-id')

def test_existing_process_retirement_after_stop():
 import importlib.util,subprocess,sys
 old=OUT.parent/'phase3-step3a-evidence/run.py'
 spec=importlib.util.spec_from_file_location('retained_retire_only',old);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)
 try:
  module.retire(p);assert p.poll() is not None
 finally:
  if p.poll() is None:p.kill();p.wait()

def test_passive_transition_stop_predicates_from_retained_native_trace():
 import gzip,copy
 old=OUT.parent/'phase3-step3a-evidence'
 prepared=[json.loads(x) for x in (old/'prepared.jsonl').read_text().splitlines()]
 t=json.loads(gzip.decompress((old/'trace-04.json.gz').read_bytes()))
 assert settled_trace_stop(prepared[4],t,transition='offload',previous_tokens=prepared[3]['token_ids']) is None
 for field,value in [('cache_published',False),('cancelled',True),('target_offsets',[]),('protocol_events',[]),('vision_encoded_count',0),('cached_tokens',len(prepared[4]['token_ids']))]:
  bad=copy.deepcopy(t);bad[field]=value
  assert settled_trace_stop(prepared[4],bad,transition='offload',previous_tokens=prepared[3]['token_ids']) is not None
 assert settled_trace_stop(prepared[4],{},transition='offload',previous_tokens=prepared[3]['token_ids']) is not None

def test_control_text_lengths_fit_frozen_ordinary_output():
 import re
 from deepseek_recipe import Tokenizer
 tokenizer=Tokenizer.from_file(str(Path(sys.prefix)/'share/ds41f-mtp/recipe/static/tokenizers/v41/tokenizer.json'))
 phrase=re.search(r"const phrase='([^']+)';",(OUT/'control.mjs').read_text())[1]
 cases=[]
 for size in [188,200,380,552,584,704,716,800,1000,1372,2080,4200]:
  text=(phrase*((size+len(phrase)-1)//len(phrase)))[:size];n=len(tokenizer.encode(text))
  assert n+128<=1200 # conservative control-tool syntax allowance, not a native usage claim
  cases.append(dict(chars=size,text_tokens=n,tool_syntax_allowance=128))
 (OUT/'control-output-tokens.json').write_text(json.dumps(dict(kind='MODEL_FREE_ACTUAL_RECIPE_TOKENIZER_TEXT_ONLY',cases=cases,pass_=True),indent=2)+'\n')

def test_progress_clock_does_not_accept_log_noise():
 clock=ProgressClock(0)
 assert not clock.note('log','heartbeat',599)
 assert clock.age(600)==600
 assert clock.note('native_frontier',('request-1',100),601)
 assert not clock.note('native_frontier',('request-1',100),1199)
 assert clock.age(1201)==600
 assert clock.note('canonical_output',('request-1',1),1202)
 assert clock.age(1203)==1

def test_atomic_receipt(tmp_path):
 p=tmp_path/'supervision.json';publish_receipt(p,stop=None,session_bytes=123,attachment_bytes=456)
 j=json.loads(p.read_text());assert j['session_bytes']==123 and j['stop'] is None and j['persistence_ok'] is False
 publish_receipt(p,stop=None,session_bytes=123,attachment_bytes=456,persistence_ok=True)
 assert json.loads(p.read_text())['persistence_ok'] is True
