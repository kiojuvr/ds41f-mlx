"""Bounded native Messages regression, not a Harness dispatcher; no retries."""
import base64
import hashlib
import json
import sys
from pathlib import Path
import httpx
from PIL import Image, ImageDraw
out = Path(sys.argv[1]); out.mkdir(exist_ok=True, parents=True)
url = sys.argv[2]
wide = Image.new('RGB', (1600,200), 'white')
draw = ImageDraw.Draw(wide)
draw.rectangle((30,25,450,175), fill='red')
draw.ellipse((650,20,830,180), fill='blue')
draw.polygon([(1200,175),(1350,20),(1500,175)], fill='green')
wide.save(out/'wide.png')
ordinary = Image.new('RGB', (384,384), 'white')
ImageDraw.Draw(ordinary).ellipse((80,80,304,304), fill='red')
ordinary.save(out/'ordinary.png')
base = dict(model='deepseek-flash', max_tokens=180, temperature=0,
    thinking=dict(type='disabled'), tools=[dict(name='read_image', description='Read a local image.',
    input_schema=dict(type='object', properties=dict(path=dict(type='string')), required=['path']))])
result = dict(status='RUNNING', url=url, cases={}, fixtures={})
for name in ('wide.png','ordinary.png'):
    result['fixtures'][name] = hashlib.sha256((out/name).read_bytes()).hexdigest()
def save(): (out/'native.json').write_text(json.dumps(result,indent=2)+'\n')
with httpx.Client(base_url=url, timeout=240) as client:
    raw=(out/'wide.png').read_bytes()
    upload=client.post('/v1/files', files={'file':('wide.png',raw,'image/png')},
        data={'expires_after[anchor]':'created_at','expires_after[seconds]':'3600'})
    upload.raise_for_status(); result['upload']=upload.json()
    messages=[dict(role='user', content=[dict(type='image', source=dict(type='file',file_id=upload.json()['id'])),
        dict(type='text',text='Describe the three colored shapes from left to right in one sentence. Do not call a tool yet.')])]
    def post(label):
        response=client.post('/v1/messages',json=base|dict(messages=messages))
        result['cases'][label]=dict(status=response.status_code, body=response.json());save()
        response.raise_for_status()
        return response.json()
    def text(value): return ' '.join(c.get('text','') for c in value['content']).lower()
    first=post('wide_file')
    assert first['stop_reason']=='end_turn', first
    t=text(first)
    assert all(word in t for word in ('red','blue','green','triangle')), t
    assert t.index('red') < t.index('blue') < t.index('green'), t
    messages += [dict(role='assistant',content=first['content']),dict(role='user',content='Now call read_image with path ordinary.png, then describe its visible shape and color.')]
    second=post('tool_call')
    calls=[c for c in second['content'] if c['type']=='tool_use']
    assert second['stop_reason']=='tool_use' and len(calls)==1, second
    call=calls[0]
    assert call['name']=='read_image' and call['input']==dict(path='ordinary.png'),call
    # Perform the one requested read, retaining the exact original tool ID.
    data=(out/call['input']['path']).read_bytes()
    result['tool_id']=call['id']
    messages += [dict(role='assistant',content=second['content']),dict(role='user',content=[dict(type='tool_result',tool_use_id=call['id'],content=[dict(type='image',source=dict(type='base64',media_type='image/png',data=base64.b64encode(data).decode()))])])]
    third=post('ordinary_tool_image')
    assert third['stop_reason']=='end_turn' and 'red' in text(third) and 'circle' in text(third),third
    messages += [dict(role='assistant',content=third['content']),dict(role='user',content='Name the color of that last image in one word, without tools.')]
    fourth=post('paired_continuation')
    assert fourth['stop_reason']=='end_turn' and 'red' in text(fourth),fourth
    result['health']=client.get('/health').json()
    assert result['health']['active_requests']==result['health']['queued_requests']==0
result['status']='PASS';save();print('PASS wide Files image, ordinary tool result, paired continuation')
