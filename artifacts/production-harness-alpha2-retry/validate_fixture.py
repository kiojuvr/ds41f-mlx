"""Independent installed candidate HTTP integration, actual ingress/parser/Recipe.
Only numerical response is controlled, as in tests/test_image_files.py; no model.
"""
import copy, hashlib, json, pathlib, sys
from types import SimpleNamespace
from ds41f_mlx.execution_identity import inspect_payload
admission = inspect_payload()  # Authenticate before lazy Transformers imports.
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from fixture import image_bytes, upload_request, initial_body, append_result
from ds41f_mlx.mtp_identity import config
from ds41f_mlx.serving.image_files import parse_upload, FileError
from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
from ds41f_mlx.serving.server import create_app

O=pathlib.Path(__file__).resolve().parent
report={'scope':'Genuine HTTP ASGI integration; controlled numerical response only; no model load',
    'python':sys.executable, 'admission':admission}
raw=image_bytes(); req=upload_request('http://127.0.0.1:8000',raw); wire=req.read()
values=parse_upload(wire,req.headers['content-type'],16*1024**2)
assert set(values)=={b'file',b'expires_after[anchor]',b'expires_after[seconds]'}
assert values[b'file'][1]==raw and values[b'expires_after[anchor]'][1]==b'created_at' and values[b'expires_after[seconds]'][1]==b'604800'
report['parser_fields']={k.decode(): {'bytes':len(v[1]),'sha256':hashlib.sha256(v[1]).hexdigest()} for k,v in values.items()}
backend=ProductionMTPBackend(runtime_config=config(profile='mtp-serving-v1'))
backend.dependency_identity='controlled-model-free'
backend.scheduler=SimpleNamespace(requests={},close_serving=lambda:None)
prepared=[]
async def respond(request,**kwargs):
    sources=[] if request.multimodal is None else list(request.multimodal._sources.values())
    prepared.append(dict(token_ids=list(request.token_ids),tokens=len(request.token_ids),images=[] if request.multimodal is None else request.multimodal.identities(),source_hashes=[hashlib.sha256(s).hexdigest() for s in sources]))
    return JSONResponse({'prepared':True})
backend.ordinary_response=respond
try:
    app=create_app(backend=backend,runtime_config=backend.runtime_config)
    with TestClient(app,base_url='http://127.0.0.1:8000') as client:
        # Preserve the old failure as an explicit negative control.
        bad=client.post('/v1/files',files={'file':('synthetic-circle.png',raw,'image/png')})
        assert bad.status_code==400 and not app.state.image_files._files,bad.text
        report['missing_expiry_negative']={'status':bad.status_code,'response':bad.json()}
        r=client.post('/v1/files',content=wire,headers={'Content-Type':req.headers['content-type']})
        assert r.status_code==200,r.text
        report['upload']=r.json();key=r.json()['id']
        assert app.state.image_files.snapshot(key).data==raw
        listing=client.get('/v1/files');assert listing.status_code==200 and any(f['id']==key for f in listing.json()['data'])
        report['listing']=listing.json()
        body=initial_body();r=client.post('/v1/messages',json=body);assert r.status_code==200,r.text
        # Validate exact prior native response grammar through current continuation ingress.
        response=json.loads((O.parent/'production-harness-alpha2-promotion/all-deferred-tool.json').read_text())
        exact_id=append_result(body['messages'],response,key)
        body.update(tool_choice={'type':'none'},stream=True)
        r=client.post('/v1/messages',json=body);assert r.status_code==200,r.text
        assert prepared[-1]['source_hashes']==[hashlib.sha256(raw).hexdigest()]
        body['messages'].extend([dict(role='assistant',content=[dict(type='text',text='A red circle.')]),dict(role='user',content='Reply only OK.')])
        body.update(stream=False,max_tokens=16)
        r=client.post('/v1/messages',json=body);assert r.status_code==200,r.text
        r=client.post('/v1/messages',json=dict(body,temperature=-1));assert r.status_code==400,r.text
        report['exact_id_schema_check']=exact_id
        report['prepared']=prepared
        assert len(prepared)==3
        first, result_ids = prepared[0]['token_ids'], prepared[1]['token_ids']
        common = next((i for i, (a,b) in enumerate(zip(first,result_ids)) if a!=b), min(len(first),len(result_ids)))
        assert common < min(len(first),len(result_ids)), 'Expected named/none canonical reprojection'
        report['canonical_reprojection'] = dict(shared_prefix=common,initial_prompt=len(first),result_prompt=len(result_ids),initial_prompt_not_prefix=True,
            explanation='Changing named tool choice to none changes canonical projection; the prior full-prompt/settled pair is not reusable. C=0 reconstruction is supported, not failure.')
        deletion=client.delete('/v1/files/'+key);assert deletion.status_code==200
        assert not client.get('/v1/files').json()['data']
        report['deletion']=deletion.json()
        try:app.state.image_files.snapshot(key)
        except FileError as e:assert e.status_code==404
        else:raise AssertionError('deleted ID still resolves')
        report.update(status='PASS',production_untouched=True)
finally:
    backend.scheduler=None;backend.close()
(O/'fixture-validation.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
print('PASS: actual parser, HTTP upload/list, File-ID native Messages preparation, all three request shapes, cleanup; no model')
