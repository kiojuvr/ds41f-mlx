"""Retry fixture revision 1: supported Files expiry, unchanged synthetic workload."""
import io
import httpx
from PIL import Image, ImageDraw

EXPIRY = {'expires_after[anchor]': 'created_at', 'expires_after[seconds]': '604800'}

def image_bytes():
    image = Image.new('RGB', (384, 384), 'white')
    ImageDraw.Draw(image).ellipse((64, 64, 320, 320), fill='red')
    output = io.BytesIO()
    image.save(output, format='PNG')
    return output.getvalue()

def upload_request(base, raw):
    return httpx.Request('POST', base + '/v1/files',
        files={'file': ('synthetic-circle.png', raw, 'image/png')}, data=EXPIRY)

def initial_body():
    tools = [dict(name='read_image', description='Read the synthetic red circle test image.',
        input_schema={'type': 'object', 'properties': {}, 'additionalProperties': False}, defer_loading=True)]
    history = [dict(role='user', content='Prepare the synthetic fixture.'),
        dict(role='system', content=[dict(type='tool_addition', tool=dict(type='tool_reference', name='read_image'))]),
        dict(role='user', content='Call read_image exactly once with no arguments.')]
    return dict(model='deepseek-flash', max_tokens=96, temperature=0, thinking={'type': 'disabled'},
        tools=tools, messages=history, tool_choice={'type': 'tool', 'name': 'read_image'})

def append_result(history, response, key):
    calls = [b for b in response['content'] if b['type'] == 'tool_use']
    assert len(calls) == 1 and calls[0]['name'] == 'read_image' and calls[0]['input'] == {}, response
    history.extend([dict(role='assistant', content=response['content']),
        dict(role='user', content=[dict(type='tool_result', tool_use_id=calls[0]['id'], content=[
            dict(type='image', source=dict(type='file', file_id=key)),
            dict(type='text', text='Describe the color and shape briefly. Do not call tools.')])])])
    return calls[0]['id']
