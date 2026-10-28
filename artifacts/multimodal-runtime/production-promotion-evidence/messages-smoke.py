"""Three bounded public native Messages requests; not an alternative Harness."""
import base64, hashlib, json
from pathlib import Path
import httpx
from PIL import Image, ImageDraw
out = Path(__file__).resolve().parent
fixture = out/'smoke.png'
im = Image.new('RGB', (384,384), 'white')
ImageDraw.Draw(im).ellipse((80,80,304,304), fill='red')
im.save(fixture)
base = dict(model='deepseek-flash', max_tokens=160, temperature=0, thinking=dict(type='disabled'),
    tools=[dict(name='read_image', description='Read a local image.', input_schema=dict(type='object', properties=dict(path=dict(type='string')), required=['path']))])
messages=[dict(role='user', content='Call read_image with path smoke.png. Then describe the visible shape and color in one sentence.')]
result=dict(status='RUNNING', cases={}, fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest())
with httpx.Client(base_url='http://127.0.0.1:8000', timeout=180) as client:
    def post(label):
        r=client.post('/v1/messages',json=base|dict(messages=messages)); r.raise_for_status()
        value=r.json();result['cases'][label]=value
        (out/'messages-smoke.json').write_text(json.dumps(result,indent=2)+'\n')
        return value
    first=post('tool_call')
    assert first['stop_reason']=='tool_use', first
    calls=[c for c in first['content'] if c['type']=='tool_use']
    assert len(calls)==1 and calls[0]['name']=='read_image' and calls[0]['input']==dict(path='smoke.png'), calls
    # Execute the actual requested read once, preserving the model's tool identity.
    raw=(out/calls[0]['input']['path']).read_bytes()
    messages += [dict(role='assistant',content=first['content']),dict(role='user',content=[dict(type='tool_result',tool_use_id=calls[0]['id'],content=[dict(type='image',source=dict(type='base64',media_type='image/png',data=base64.b64encode(raw).decode()))])])]
    second=post('image_result')
    assert second['stop_reason']=='end_turn', second
    text=' '.join(c.get('text','') for c in second['content']).lower()
    assert 'red' in text and ('circle' in text or 'circular' in text), text
    messages += [dict(role='assistant',content=second['content']),dict(role='user',content='Name that same color again in one word. Do not call tools.')]
    third=post('paired_continuation')
    assert third['stop_reason']=='end_turn', third
    assert 'red' in ' '.join(c.get('text','') for c in third['content']).lower()
    result['health']=client.get('/health').json()
    assert result['health']['active_requests']==result['health']['queued_requests']==0
result['status']='PASS'
(out/'messages-smoke.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS native Messages tool/image consumption and continuation')
