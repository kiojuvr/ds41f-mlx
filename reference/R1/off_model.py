"""Fresh omitted-profile OFF regression: protocols, tools, multi-session and restore."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from ds41f_mlx.web_client import RuntimeClient
from ds41f_mlx.mtp_profile import WEATHER

OUT=Path(sys.argv[1]);OUT.parent.mkdir(parents=True,exist_ok=True)
result=dict(schema='ds41f.m41.off-protocols.v1',status='RUNNING',cases=[])
with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
log=OUT.with_suffix('.server.log').open('w')
process=subprocess.Popen([sys.executable,'-m','ds41f_mlx.ops','start','--port',str(port)],stdout=log,stderr=subprocess.STDOUT)
rt=RuntimeClient(f'http://127.0.0.1:{port}')
try:
    for _ in range(1000):
        try:h=rt.health();break
        except Exception:
            if process.poll() is not None:raise AssertionError('OFF startup failed')
            time.sleep(.05)
    else:raise AssertionError('OFF health deadline')
    assert h.get('profile','standard-off')=='standard-off' and not h['model_ready']
    bodies=[('/v1/chat/completions',dict(model='deepseek-v4.1-flash',messages=[dict(role='user',content='Say hello briefly.')],
                                      reasoning_effort='none',temperature=0,max_tokens=16)),
            ('/v1/responses',dict(model='deepseek-v4.1-flash',input='Say hello briefly.',temperature=0,max_output_tokens=16)),
            ('/v1/messages',dict(model='deepseek-v4.1-flash',messages=[dict(role='user',content='Say hello briefly.')],temperature=0,max_tokens=16))]
    for route,body in bodies:
        start=time.monotonic();value=rt.request('POST',route,body)
        result['cases'].append(dict(route=route,response=value,wall_s=time.monotonic()-start))
    sessions=[rt.create_session('m41_off_'+str(i)) for i in range(4)]
    assert [s['id'] for s in sessions]==['m41_off_'+str(i) for i in range(4)]
    from ds41f_mlx.web_client import RuntimeHTTPError
    try:rt.create_session('m41_off_5')
    except RuntimeHTTPError as exc:assert exc.status==409
    else:raise AssertionError('OFF live capacity drift')
    sid=sessions[0]['id']
    body=dict(model='deepseek-v4.1-flash',messages=[dict(role='user',content='Use the tool to look up the weather in Paris, then answer concisely.')],
              tools=[WEATHER],tool_choice={'type':'function','function':{'name':'lookup_weather'}},temperature=0,
              max_tokens=96,reasoning_effort='none')
    call=rt.chat(sid,body);msg=call['choices'][0]['message'];calls=msg['tool_calls']
    assert call['choices'][0]['finish_reason']=='tool_calls' and len(calls)==1
    body['messages'].append(msg)
    body['messages'].append(dict(role='tool',tool_call_id=calls[0]['id'],content='{"city":"Paris","weather":"sunny 21C"}'))
    body['tool_choice']='auto';body['stream']=True
    import http.client
    conn=http.client.HTTPConnection('127.0.0.1',port,timeout=1800)
    conn.request('POST',f'/v1/sessions/{sid}/chat/completions',json.dumps(body),{'Content-Type':'application/json'})
    response=conn.getresponse();assert response.status==200
    raw=response.read().decode();response.close();conn.close()
    assert 'data: [DONE]' in raw
    events=[json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: ') and line!='data: [DONE]']
    text=''.join(e['choices'][0].get('delta',{}).get('content','') or '' for e in events if e.get('choices'))
    body['stream']=False;body['messages'].append(dict(role='assistant',content=text))
    persisted=rt.persist(sid,str(OUT.parent.resolve()/'off-kv'))
    artifact=persisted['artifact']
    rt.close_session(sid)
    # Corrupt dormant artifacts must never become a live cache authority.
    manifest_path=Path(artifact['path'])/'manifest.json'
    original=manifest_path.read_text(); manifest=json.loads(original)
    result['persistence_rejections']=[]
    try:
        for field,bad in [('schema','wrong'),('tensor_sha256','0'*64),('frontier',-1)]:
            broken=dict(manifest);broken[field]=bad
            manifest_path.write_text(json.dumps(broken))
            try:rt.restore(artifact['path'],'r1_corrupt_'+field)
            except RuntimeHTTPError as exc:
                assert 400 <= exc.status < 500
                result['persistence_rejections'].append(dict(field=field,status=exc.status))
            else:raise AssertionError('corrupt persistence admitted')
    finally:manifest_path.write_text(original)
    restored=rt.restore(artifact['path'],'m41_off_restored')
    body['messages'].append(dict(role='user',content='Say goodbye briefly.'))
    final=rt.chat(restored['id'],body)
    assert final['choices'][0]['message']['role']=='assistant'
    result.update(status='PASS',requested_ids=True,live_sessions=4,stateful_tool=call,
                  streamed_result_events=len(events),persist=persisted,restore=restored,continued=final,
                  final_health=rt.health())
    for sid in [s['id'] for s in sessions[1:]]+[restored['id']]:rt.close_session(sid)
except BaseException as exc:result.update(status='FAIL',error=repr(exc));raise
finally:
    OUT.write_text(json.dumps(result,indent=2)+'\n');process.terminate()
    try:process.wait(timeout=120)
    except subprocess.TimeoutExpired:process.kill();process.wait()
    log.close()
print(json.dumps(dict(status=result['status'],cases=len(result['cases']))))
