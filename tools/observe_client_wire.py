"""Acceptance-only byte-preserving OpenCode transport observer (loopback).

No request defaults, normalization, retries, tool execution or response rewriting.
The production runtime remains the sole execution authority.
"""
import json
from pathlib import Path
import sys

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

output = Path(sys.argv[1]).resolve()
output.parent.mkdir(parents=True, exist_ok=True)
app = FastAPI()


@app.post('/v1/chat/completions')
async def chat(request: Request):
    body = await request.body()
    with output.open('a') as stream:
        stream.write(json.dumps(dict(body=body.decode())) + '\n')
    client = httpx.AsyncClient(timeout=1800, trust_env=False)
    response = await client.send(client.build_request('POST',
        'http://127.0.0.1:8000/v1/chat/completions', content=body,
        headers={'content-type': 'application/json'}), stream=True)
    with output.with_suffix('.statuses.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(status=response.status_code)) + '\n')

    async def chunks():
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()
    return StreamingResponse(chunks(), status_code=response.status_code,
                             media_type=response.headers.get('content-type'))


uvicorn.run(app, host='127.0.0.1', port=8001, log_level='warning')
