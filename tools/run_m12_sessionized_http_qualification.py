#!/usr/bin/env python3
"""M12 sessionized HTTP qualification for Chat Completions tool loops."""
from __future__ import annotations

import argparse, contextlib, http.client, json, os, socket, subprocess, sys, threading, time, urllib.request, urllib.error
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT, DEFAULT_OMLX
from ds41f_mlx.serving.deepseek_recipe_backend import DEFAULT_RECIPE, DEFAULT_MODEL_ID, git_rev


def free_port() -> int:
    s=socket.socket(); s.bind(('127.0.0.1',0)); p=s.getsockname()[1]; s.close(); return p


def http_json(method: str, url: str, data: dict[str, Any] | None = None, timeout: float = 1800) -> tuple[int, dict[str, Any]]:
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, method=method, headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw=r.read().decode(); return r.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw=e.read().decode(errors='replace')
        try: parsed=json.loads(raw)
        except Exception: parsed={'raw':raw}
        return e.code, parsed


def start_server(args: argparse.Namespace, port: int) -> subprocess.Popen:
    env=os.environ.copy(); env['DS41F_ENABLE_DIAGNOSTIC_ENDPOINTS']='1'; env['DS41F_MAX_LIVE_SESSIONS']='4'
    return subprocess.Popen([sys.executable, str(ROOT/'tools/run_ds41f_recipe_server.py'), '--host','127.0.0.1','--port',str(port),'--checkpoint',str(args.checkpoint),'--omlx-path',str(args.omlx_path),'--recipe-path',str(args.recipe_path)], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_health(base: str) -> None:
    deadline=time.time()+180
    last=None
    while time.time()<deadline:
        try:
            c,d=http_json('GET',base+'/health',None,timeout=5); last=(c,d)
            if c==200: return
        except Exception as e: last=repr(e)
        time.sleep(.5)
    raise RuntimeError(f'health timeout: {last}')


def tool_def() -> dict[str, Any]:
    return {"type":"function","function":{"name":"lookup_weather","description":"Return deterministic weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":False},"strict":True}}


def initial_body(*, stream: bool = False) -> dict[str, Any]:
    return {"model":DEFAULT_MODEL_ID,"messages":[{"role":"user","content":"Use the tool to look up the weather in Paris, then answer concisely."}],"tools":[tool_def()],"tool_choice":{"type":"function","function":{"name":"lookup_weather"}},"reasoning_effort":"none","temperature":0,"max_tokens":96,"stream":stream}


def tool_stub(name: str, arguments: str) -> str:
    args=json.loads(arguments or '{}'); city=args.get('city','unknown')
    return json.dumps({'city':city,'weather':'sunny 21C','source':'m12_http_stub'}, separators=(',',':'))


def continuation_body(call: dict[str, Any], result: str, *, stream: bool = False) -> dict[str, Any]:
    return {"model":DEFAULT_MODEL_ID,"messages":[
        initial_body()["messages"][0],
        {"role":"assistant","content":None,"tool_calls":[{"id":call['id'],"type":"function","function":{"name":call['name'],"arguments":call['arguments']}}]},
        {"role":"tool","tool_call_id":call['id'],"content":result},
    ],"tools":[tool_def()],"tool_choice":"auto","reasoning_effort":"none","temperature":0,"max_tokens":96,"stream":stream}


def call_from_chat_response(resp: dict[str, Any]) -> dict[str, Any]:
    tc=resp['choices'][0]['message']['tool_calls'][0]; fn=tc['function']
    return {'id':tc['id'],'name':fn['name'],'arguments':fn['arguments']}


def stream_chat(base: str, sid: str, body: dict[str, Any], *, cancel_after_first: bool = False) -> dict[str, Any]:
    conn=http.client.HTTPConnection('127.0.0.1', int(base.rsplit(':',1)[1]), timeout=1800)
    conn.request('POST', f'/v1/sessions/{sid}/chat/completions', body=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
    resp=conn.getresponse(); events=[]; buf=b''; cancelled=False
    try:
        while True:
            b=resp.read(1)
            if not b: break
            buf += b
            if b'\n\n' in buf:
                frame, buf = buf.split(b'\n\n',1)
                text=frame.decode(errors='replace')
                if text.startswith('data: '):
                    data=text[6:]
                    events.append(data)
                    if cancel_after_first and data != '[DONE]':
                        cancelled=True; conn.close(); break
                    if data == '[DONE]': break
        return {'status':resp.status,'events':events,'cancelled_client':cancelled}
    finally:
        with contextlib.suppress(Exception): conn.close()


def call_from_stream_events(events: list[str]) -> dict[str, Any]:
    call={'id':'','name':'','arguments':''}
    for data in events:
        if data == '[DONE]': continue
        obj=json.loads(data); delta=obj.get('choices',[{}])[0].get('delta',{})
        for tc in delta.get('tool_calls') or []:
            if tc.get('id'): call['id']=tc['id']
            fn=tc.get('function') or {}
            if fn.get('name'): call['name']=fn['name']
            if fn.get('arguments') is not None: call['arguments'] += fn.get('arguments')
    return call


def live_http_flow(base: str, *, stream: bool = False) -> dict[str, Any]:
    c,s=http_json('POST',base+'/v1/sessions',{}) ; assert c==200, (c,s)
    sid=s['id']; body0=initial_body(stream=stream)
    if stream:
        st=stream_chat(base,sid,body0); call=call_from_stream_events(st['events']); first={'stream':st}
    else:
        c,first=http_json('POST',f'{base}/v1/sessions/{sid}/chat/completions',body0); assert c==200,(c,first); call=call_from_chat_response(first)
    result=tool_stub(call['name'], call['arguments'])
    c,second=http_json('POST',f'{base}/v1/sessions/{sid}/chat/completions',continuation_body(call,result)); assert c==200,(c,second)
    c,diag=http_json('GET',base+'/_ds41f/diagnostics'); assert c==200
    sess=diag['sessions'][sid]
    return {'session_id':sid,'stream':stream,'tool_call':call,'tool_result':result,'first':first,'second':second,'session':sess,'session_traces':diag['session_traces'],'ok': bool(call['id'] and call['name']=='lookup_weather' and sess['diagnostics']['m8']['total_prompt_replay_count']==0 and sess['diagnostics']['m8']['total_full_cache_repack_count']==0 and sess['diagnostics']['m8']['all_cache_offsets_equal_frontier'])}


def persisted_http_flow(args: argparse.Namespace) -> dict[str, Any]:
    port=free_port(); base=f'http://127.0.0.1:{port}'; p=start_server(args,port)
    try:
        wait_health(base); c,s=http_json('POST',base+'/v1/sessions',{}) ; sid=s['id']
        c,first=http_json('POST',f'{base}/v1/sessions/{sid}/chat/completions',initial_body()); call=call_from_chat_response(first)
        result=tool_stub(call['name'], call['arguments'])
        c,persist=http_json('POST',f'{base}/v1/sessions/{sid}/persist',{'artifact_root':str(args.kv_root)}); assert c==200,(c,persist)
        artifact=persist['artifact']
    finally:
        p.terminate();
        with contextlib.suppress(Exception): p.wait(timeout=20)
        if p.poll() is None: p.kill()
    port2=free_port(); base2=f'http://127.0.0.1:{port2}'; p2=start_server(args,port2)
    try:
        wait_health(base2); c,rest=http_json('POST',base2+'/v1/sessions/restore',{'artifact_path':artifact['path']}); assert c==200,(c,rest)
        sid2=rest['id']; c,second=http_json('POST',f'{base2}/v1/sessions/{sid2}/chat/completions',continuation_body(call,result)); assert c==200,(c,second)
        c,diag=http_json('GET',base2+'/_ds41f/diagnostics'); sess=diag['sessions'][sid2]
        return {'session_id':sid2,'tool_call':call,'tool_result':result,'artifact':artifact,'restore':rest,'second':second,'session':sess,'session_traces':diag['session_traces'],'ok': bool(sess['diagnostics']['m8']['total_prompt_replay_count']==0 and sess['diagnostics']['m8']['total_full_cache_repack_count']==0 and sess['diagnostics']['m8']['all_cache_offsets_equal_frontier'])}
    finally:
        p2.terminate();
        with contextlib.suppress(Exception): p2.wait(timeout=20)
        if p2.poll() is None: p2.kill()


def conflict_probe(base: str) -> dict[str, Any]:
    c,s=http_json('POST',base+'/v1/sessions',{}) ; sid=s['id']
    first: dict[str, Any] = {}
    def run_first():
        c1,d1=http_json('POST',f'{base}/v1/sessions/{sid}/chat/completions',initial_body(),timeout=1800)
        first.update({'status':c1,'body':d1})
    t=threading.Thread(target=run_first, daemon=True); t.start(); time.sleep(1.0)
    c2,d2=http_json('POST',f'{base}/v1/sessions/{sid}/chat/completions',initial_body(),timeout=60)
    t.join(timeout=1800)
    c3,d3=http_json('POST',f'{base}/v1/sessions/not-a-session/chat/completions',initial_body(),timeout=60)
    return {'session_id':sid,'overlap_status':c2,'overlap_body':d2,'first_status':first.get('status'),'unknown_status':c3,'unknown_body':d3,'ok': c2==409 and first.get('status')==200 and c3==404}


def cancel_probe(base: str) -> dict[str, Any]:
    c,s=http_json('POST',base+'/v1/sessions',{}) ; sid=s['id']
    st=stream_chat(base,sid,initial_body(stream=True),cancel_after_first=True)
    c,diag=http_json('GET',base+'/_ds41f/diagnostics')
    return {'session_id':sid,'stream_cancel':st,'session':diag['sessions'][sid],'ok': st['cancelled_client'] and diag['sessions'][sid]['state']=='idle'}


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT); ap.add_argument('--omlx-path',type=Path,default=DEFAULT_OMLX); ap.add_argument('--recipe-path',type=Path,default=DEFAULT_RECIPE); ap.add_argument('--kv-root',type=Path,default=Path('/Volumes/USB-SSD-RAID-0/ds41f-mlx/kv-m12')); ap.add_argument('--out',type=Path,default=Path('artifacts/m12/sessionized-http-qualification.json'))
    args=ap.parse_args(); port=free_port(); base=f'http://127.0.0.1:{port}'; proc=start_server(args,port)
    rec={'schema':'ds41f.m12.sessionized-http-qualification.v1','created_at':time.time(),'checkpoint':str(args.checkpoint),'omlx_path':str(args.omlx_path),'omlx_revision':git_rev(args.omlx_path),'recipe_path':str(args.recipe_path),'recipe_revision':git_rev(args.recipe_path),'base':base,'flows':{},'ok':False}
    try:
        wait_health(base)
        rec['flows']['nonstream_live']=live_http_flow(base,stream=False)
        rec['flows']['stream_live']=live_http_flow(base,stream=True)
        rec['flows']['cancel_stream_after_first_event']=cancel_probe(base)
        rec['flows']['overlap_and_unknown_session']=conflict_probe(base)
    finally:
        proc.terminate();
        with contextlib.suppress(Exception): proc.wait(timeout=20)
        if proc.poll() is None: proc.kill()
    rec['flows']['persist_restore']=persisted_http_flow(args)
    rec['ok']=all(flow.get('ok') for flow in rec['flows'].values())
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(rec,indent=2))
    print(json.dumps({'ok':rec['ok'],'out':str(args.out)},indent=2))
    return 0 if rec['ok'] else 1

if __name__=='__main__': raise SystemExit(main())
