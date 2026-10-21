"""Bounded installed Messages acceptance, no external tool effects."""
import json, time, threading, traceback
from pathlib import Path
OUT=Path('/private/tmp/ds41f-messages-http-real'); OUT.mkdir(exist_ok=True)
from ds41f_mlx.mtp_identity import config, inspect
cfg=config(host='127.0.0.1', port=18037, profile='mtp-serving-v1')
identity=inspect(cfg)
(OUT/'inspect.json').write_text(json.dumps(identity,indent=2,default=str))
cfg.apply_environment()
from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
from ds41f_mlx.serving.server import create_app
from ds41f_mlx.serving.local_h11 import LocalH11Protocol
import httpx, uvicorn
backend=ProductionMTPBackend(runtime_config=cfg)
backend.dependency_identity=identity['identity_sha256']; backend.glm_profile=identity['glm_profile']
app=create_app(profile='mtp-serving-v1',runtime_config=cfg,backend=backend)
async def load(): await backend._call(backend.load)
app.router.add_event_handler('startup',load)
server=uvicorn.Server(uvicorn.Config(app,host=cfg.host,port=cfg.port,workers=1,proxy_headers=False,ws='none',loop='asyncio',limit_concurrency=8,backlog=8,timeout_keep_alive=5,h11_max_incomplete_event_size=16384,http=LocalH11Protocol))
thread=threading.Thread(target=server.run)
receipt={'cases': [], 'status':'RUNNING'}
def save(): (OUT/'results.json').write_text(json.dumps(receipt,indent=2,default=str)+'\n')
thread.start(); save()
try:
    while not server.started and thread.is_alive(): time.sleep(.2)
    assert server.started
    with httpx.Client(base_url='http://127.0.0.1:18037',timeout=1800) as c:
        receipt['ready']=c.get('/health').json(); assert receipt['ready']['model_ready']
        def post(label,body):
            r=c.post('/v1/messages',json=body)
            row={'label':label,'request':body,'http_status':r.status_code,'wire':r.text}
            receipt['cases'].append(row); save(); r.raise_for_status()
            for _ in range(18000):
                health=c.get('/health').json()
                if not health['active_requests'] and not health['queued_requests']: break
                time.sleep(.1)
            assert not health['fatal_error'],health
            t=dict(backend.serving_traces[-1]);row['trace']=t;save()
            assert t['cache_published'] and not t['cancelled'] and not t.get('recovery_error'),t
            assert set(t['target_offsets']+t['dspark_offsets'])=={t['canonical_frontier']}
            assert t['prompt_replay']==t['full_cache_repack']==0
            assert t['prediction_retired'] and t['queue_empty']
            if body.get('stream'):
                from ds41f_mlx.serving.server import sse_frame
                assert r.text==''.join(sse_frame(e['type'],json.dumps(e,separators=(',',':'))) for e in t['protocol_events'])
            else: assert r.json()==t['response']
            print(label,t['response'],flush=True)
            return t['response'],t
        base=dict(model='deepseek-flash',temperature=0,thinking={'type':'disabled'},max_tokens=32)
        post('text',base|dict(messages=[dict(role='user',content='Say Hello briefly.')]))
        tools=[dict(name='read_file',description='Read a file',input_schema=dict(type='object',properties=dict(path=dict(type='string')),required=['path']))]
        history=[dict(role='user',content='Call read_file with path /tmp/example.txt. Do not answer until you receive the file content.')]
        call,t=post('tool',base|dict(max_tokens=128,stream=True,tools=tools,messages=history))
        calls=[b for b in call['content'] if b['type']=='tool_use']; assert calls and call['stop_reason']=='tool_use'
        assert calls==[c['call'] for c in t['call_certificates']]
        history += [dict(role='assistant',content=call['content']),dict(role='user',content=[dict(type='tool_result',tool_use_id=b['id'],content='Synthetic tool result: the file contains MESSAGES_OK. Report this exact marker.') for b in calls])]
        post('continuation',base|dict(max_tokens=128,tools=tools,messages=history))
        receipt['status']='PASS';save()
except BaseException:
    receipt['status']='FAIL';receipt['error']=traceback.format_exc();save();raise
finally:
    server.should_exit=True;thread.join(timeout=1800);backend.close()
    receipt['server_retired']=not thread.is_alive();save()
