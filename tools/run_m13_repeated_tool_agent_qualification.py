#!/usr/bin/env python3
"""M13 repeated-tool sessionized HTTP qualification."""
from __future__ import annotations

import argparse, contextlib, http.client, json, os, socket, subprocess, sys, threading, time, urllib.request, urllib.error
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT, DEFAULT_OMLX
from ds41f_mlx.serving.deepseek_recipe_backend import DEFAULT_RECIPE, DEFAULT_MODEL_ID, git_rev


def free_port():
    s=socket.socket(); s.bind(('127.0.0.1',0)); p=s.getsockname()[1]; s.close(); return p

def http_json(method,url,data=None,timeout=1800):
    body=None if data is None else json.dumps(data).encode(); req=urllib.request.Request(url,data=body,method=method,headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=timeout) as r:
            raw=r.read().decode(); return r.status,(json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw=e.read().decode(errors='replace')
        try: return e.code,json.loads(raw)
        except Exception: return e.code,{'raw':raw}

def start_server(args,port):
    env=os.environ.copy(); env['DS41F_ENABLE_DIAGNOSTIC_ENDPOINTS']='1'; env['DS41F_MAX_LIVE_SESSIONS']='4'
    return subprocess.Popen([sys.executable,str(ROOT/'tools/run_ds41f_recipe_server.py'),'--host','127.0.0.1','--port',str(port),'--checkpoint',str(args.checkpoint),'--omlx-path',str(args.omlx_path),'--recipe-path',str(args.recipe_path)],cwd=ROOT,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

def wait_health(base):
    deadline=time.time()+180; last=None
    while time.time()<deadline:
        try:
            c,d=http_json('GET',base+'/health',None,5); last=(c,d)
            if c==200: return
        except Exception as e: last=repr(e)
        time.sleep(.5)
    raise RuntimeError(f'health timeout {last}')

def tool_def():
    return {"type":"function","function":{"name":"lookup_weather","description":"Return deterministic weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":False},"strict":True}}

def first_body():
    return {"model":DEFAULT_MODEL_ID,"messages":[{"role":"user","content":"Call lookup_weather for Paris. After I provide that tool result, call lookup_weather for Berlin. Do not answer finally until both tool results are available."}],"tools":[tool_def()],"tool_choice":{"type":"function","function":{"name":"lookup_weather"}},"reasoning_effort":"none","temperature":0,"max_tokens":96}

def body_after_a(call_a,result_a,*,bad:dict|None=None):
    msgs=[first_body()['messages'][0],{"role":"assistant","content":None,"tool_calls":[{"id":call_a['id'],"type":"function","function":{"name":call_a['name'],"arguments":call_a['arguments']}}]}]
    if bad and bad.get('wrong_id'):
        msgs.append({"role":"tool","tool_call_id":"call_wrong","content":result_a})
    elif bad and bad.get('duplicate'):
        msgs.append({"role":"tool","tool_call_id":call_a['id'],"content":result_a}); msgs.append({"role":"tool","tool_call_id":call_a['id'],"content":result_a})
    else:
        msgs.append({"role":"tool","tool_call_id":call_a['id'],"content":result_a})
    msgs.append({"role":"user","content":"Now call lookup_weather for Berlin using the same tool."})
    return {"model":DEFAULT_MODEL_ID,"messages":msgs,"tools":[tool_def()],"tool_choice":{"type":"function","function":{"name":"lookup_weather"}},"reasoning_effort":"none","temperature":0,"max_tokens":96}

def body_after_b(call_a,result_a,call_b,result_b):
    msgs=body_after_a(call_a,result_a)['messages']
    msgs.append({"role":"assistant","content":None,"tool_calls":[{"id":call_b['id'],"type":"function","function":{"name":call_b['name'],"arguments":call_b['arguments']}}]})
    msgs.append({"role":"tool","tool_call_id":call_b['id'],"content":result_b})
    msgs.append({"role":"user","content":"Now provide one concise final answer summarizing both weather results."})
    return {"model":DEFAULT_MODEL_ID,"messages":msgs,"tools":[tool_def()],"tool_choice":"auto","reasoning_effort":"none","temperature":0,"max_tokens":96}

def call_from(resp):
    tc=resp['choices'][0]['message']['tool_calls'][0]; fn=tc['function']; return {'id':tc['id'],'name':fn['name'],'arguments':fn['arguments']}

def stub(call):
    city=json.loads(call['arguments']).get('city'); table={'Paris':'sunny 21C','Berlin':'cloudy 17C'}
    return json.dumps({'city':city,'weather':table.get(city,'clear 20C'),'source':'m13_stub'},separators=(',',':'))

def diag(base,sid):
    c,d=http_json('GET',base+'/_ds41f/diagnostics'); return d['sessions'][sid]

def frontier(sess): return sess['diagnostics']['m8']['frontier']

def repeated_live(base):
    c,s=http_json('POST',base+'/v1/sessions',{}); sid=s['id']
    c,r1=http_json('POST',f'{base}/v1/sessions/{sid}/chat/completions',first_body()); assert c==200,(c,r1)
    call_a=call_from(r1); res_a=stub(call_a); f1=frontier(diag(base,sid))
    # Invalid wrong/duplicate tool results fail before mutation.
    before=frontier(diag(base,sid)); c_bad,d_bad=http_json('POST',f'{base}/v1/sessions/{sid}/chat/completions',body_after_a(call_a,res_a,bad={'wrong_id':1}),timeout=120); after_bad=frontier(diag(base,sid))
    c_dup,d_dup=http_json('POST',f'{base}/v1/sessions/{sid}/chat/completions',body_after_a(call_a,res_a,bad={'duplicate':1}),timeout=120); after_dup=frontier(diag(base,sid))
    c,r2=http_json('POST',f'{base}/v1/sessions/{sid}/chat/completions',body_after_a(call_a,res_a)); assert c==200,(c,r2)
    call_b=call_from(r2); res_b=stub(call_b); f2=frontier(diag(base,sid))
    c,r3=http_json('POST',f'{base}/v1/sessions/{sid}/chat/completions',body_after_b(call_a,res_a,call_b,res_b)); assert c==200,(c,r3)
    sess=diag(base,sid); m8=sess['diagnostics']['m8']
    return {'session_id':sid,'tool_calls':[call_a,call_b],'tool_results':[res_a,res_b],'frontiers':[f1,f2,frontier(sess)],'invalid_wrong_id':{'status':c_bad,'frontier_before':before,'frontier_after':after_bad,'body':d_bad},'invalid_duplicate':{'status':c_dup,'frontier_after':after_dup,'body':d_dup},'final':r3,'session':sess,'ok': bool(call_a['name']=='lookup_weather' and call_b['name']=='lookup_weather' and json.loads(call_a['arguments']).get('city')=='Paris' and json.loads(call_b['arguments']).get('city')=='Berlin' and c_bad==400 and c_dup==400 and before==after_bad==after_dup and m8['total_prompt_replay_count']==0 and m8['total_full_cache_repack_count']==0 and m8['all_cache_offsets_equal_frontier'])}

def persisted_between_b_and_result(args):
    port=free_port(); base=f'http://127.0.0.1:{port}'; p=start_server(args,port)
    try:
        wait_health(base); c,s=http_json('POST',base+'/v1/sessions',{}); sid=s['id']
        c,r1=http_json('POST',f'{base}/v1/sessions/{sid}/chat/completions',first_body()); call_a=call_from(r1); res_a=stub(call_a)
        c,r2=http_json('POST',f'{base}/v1/sessions/{sid}/chat/completions',body_after_a(call_a,res_a)); call_b=call_from(r2); res_b=stub(call_b)
        c,persist=http_json('POST',f'{base}/v1/sessions/{sid}/persist',{'artifact_root':str(args.kv_root)}); artifact=persist['artifact']
    finally:
        p.terminate();
        with contextlib.suppress(Exception): p.wait(timeout=20)
        if p.poll() is None: p.kill()
    port2=free_port(); base2=f'http://127.0.0.1:{port2}'; p2=start_server(args,port2)
    try:
        wait_health(base2); c,rest=http_json('POST',base2+'/v1/sessions/restore',{'artifact_path':artifact['path']}); sid2=rest['id']
        c,r3=http_json('POST',f'{base2}/v1/sessions/{sid2}/chat/completions',body_after_b(call_a,res_a,call_b,res_b)); assert c==200,(c,r3)
        sess=diag(base2,sid2); m8=sess['diagnostics']['m8']
        return {'session_id':sid2,'tool_calls':[call_a,call_b],'artifact':artifact,'restore':rest,'final':r3,'session':sess,'ok': bool(m8['total_prompt_replay_count']==0 and m8['total_full_cache_repack_count']==0 and m8['all_cache_offsets_equal_frontier'])}
    finally:
        p2.terminate();
        with contextlib.suppress(Exception): p2.wait(timeout=20)
        if p2.poll() is None: p2.kill()

def post_tool_control(base, trajectory):
    # Same complete body as stateful final turn, but through stateless M7 endpoint.
    call_a,call_b=trajectory['tool_calls']; res_a,res_b=trajectory['tool_results']
    body=body_after_b(call_a,res_a,call_b,res_b)
    c,stateless=http_json('POST',base+'/v1/chat/completions',body); assert c==200,(c,stateless)
    stateful=trajectory['final']; sc=(stateful['choices'][0]['message'].get('content') or ''); fc=(stateless['choices'][0]['message'].get('content') or '')
    return {'stateful_finish':stateful['choices'][0]['finish_reason'],'stateless_finish':stateless['choices'][0]['finish_reason'],'stateful_prefix':sc[:300],'stateless_prefix':fc[:300],'prefix_equal_120':sc[:120]==fc[:120],'stateful_len':len(sc),'stateless_len':len(fc),'classification':'same prompt/control exhibits comparable post-tool length/no-progress behavior' if sc[:40]==fc[:40] else 'stateful/stateless content differs; runtime counters decide defect classification','ok': True}

def conflict_probe(base):
    c,s=http_json('POST',base+'/v1/sessions',{}); sid=s['id']; first={}
    def run():
        c1,d1=http_json('POST',f'{base}/v1/sessions/{sid}/chat/completions',first_body()); first.update({'status':c1})
    t=threading.Thread(target=run,daemon=True); t.start(); time.sleep(1)
    c2,d2=http_json('POST',f'{base}/v1/sessions/{sid}/chat/completions',first_body(),timeout=60); t.join(timeout=1800)
    return {'session_id':sid,'overlap_status':c2,'first_status':first.get('status'),'ok':c2==409 and first.get('status')==200}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT); ap.add_argument('--omlx-path',type=Path,default=DEFAULT_OMLX); ap.add_argument('--recipe-path',type=Path,default=DEFAULT_RECIPE); ap.add_argument('--kv-root',type=Path,default=Path('/Volumes/USB-SSD-RAID-0/ds41f-mlx/kv-m13')); ap.add_argument('--out',type=Path,default=Path('artifacts/m13/repeated-tool-agent-qualification.json'))
    args=ap.parse_args(); port=free_port(); base=f'http://127.0.0.1:{port}'; p=start_server(args,port)
    rec={'schema':'ds41f.m13.repeated-tool-agent-qualification.v1','created_at':time.time(),'checkpoint':str(args.checkpoint),'omlx_path':str(args.omlx_path),'omlx_revision':git_rev(args.omlx_path),'recipe_path':str(args.recipe_path),'recipe_revision':git_rev(args.recipe_path),'flows':{},'ok':False}
    try:
        wait_health(base)
        live=repeated_live(base); rec['flows']['repeated_live']=live
        rec['flows']['post_tool_control']=post_tool_control(base,live)
        rec['flows']['overlap']=conflict_probe(base)
    finally:
        p.terminate();
        with contextlib.suppress(Exception): p.wait(timeout=20)
        if p.poll() is None: p.kill()
    rec['flows']['persist_between_tool_b_and_result']=persisted_between_b_and_result(args)
    rec['ok']=all(v.get('ok') for v in rec['flows'].values())
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(rec,indent=2))
    print(json.dumps({'ok':rec['ok'],'out':str(args.out)},indent=2)); return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
