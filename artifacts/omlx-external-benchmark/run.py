"""Bounded invocation of unmodified upstream external benchmark functions.

No server/MLX dependencies are installed. The admin package is loaded as a
namespace to avoid its __init__ importing the unrelated web-server routes.
Transport capture only buffers wire bytes; files are written between trials.
"""
import asyncio
import importlib
import json
import logging
import os
from pathlib import Path
import sys
import types

import httpx

UPSTREAM = Path(os.environ['OMLX_UPSTREAM']).resolve()
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(UPSTREAM))
import omlx
admin = types.ModuleType('omlx.admin')
admin.__path__ = [str(UPSTREAM / 'omlx/admin')]
sys.modules['omlx.admin'] = admin
b = importlib.import_module('omlx.admin.benchmark')
e = importlib.import_module('omlx.admin.external_api')


class CaptureStream(httpx.AsyncByteStream):
    def __init__(self, inner, record):
        self.inner, self.record = inner, record

    async def __aiter__(self):
        async for chunk in self.inner:
            self.record['chunks'].append(chunk)
            yield chunk

    async def aclose(self):
        await self.inner.aclose()


class CaptureTransport(httpx.AsyncHTTPTransport):
    async def handle_async_request(self, request):
        record = {'request': json.loads(request.content), 'chunks': []}
        records.append(record)
        response = await super().handle_async_request(request)
        record['status'] = response.status_code
        response.stream = CaptureStream(response.stream, record)
        return response


records = []


def save_wire(label):
    record = records[-1]
    (OUT / f'{label}.request.json').write_text(json.dumps(record['request'], indent=2) + '\n')
    (OUT / f'{label}.sse').write_bytes(b''.join(record['chunks']))
    (OUT / f'{label}.http.json').write_text(json.dumps({'status': record.get('status')}) + '\n')
    record['chunks'].clear()


async def health(label):
    async with httpx.AsyncClient() as client:
        response = await client.get('http://127.0.0.1:8000/health')
        response.raise_for_status()
        data = response.json()
    (OUT / f'{label}.health.json').write_text(json.dumps(data, indent=2) + '\n')
    if data['active_requests'] or data['queued_requests'] or data['status'] != 'ready':
        raise RuntimeError('Production not idle/ready; stop before sending next request')
    return data


async def main():
    request = b.BenchmarkRequest(
        model_id='deepseek-v4.1-flash', prompt_lengths=b.VALID_PROMPT_LENGTHS,
        generation_length=128, batch_sizes=[], context_profile='code_python',
        warmup_mode='quick', external=e.ExternalEndpointConfig(
            base_url='http://127.0.0.1:8000/v1', model='deepseek-v4.1-flash',
            extra_body={'reasoning_effort': 'none'}))
    (OUT / 'configuration.json').write_text(request.model_dump_json(indent=2) + '\n')
    client = e.ExternalAPIClient(request.external, transport=CaptureTransport())
    results = []
    try:
        await health('initial')
        # Exact external quick warmup from upstream _run_external_benchmark.
        print('warmup: approximate pp32, max_tokens=8, temperature=0', flush=True)
        stats = await client.stream_chat_completion(
            messages=[{'role': 'user', 'content': b._generate_external_prompt(
                b._warmup_prompt_tokens(request.warmup_mode), request.context_profile)}],
            max_tokens=8, temperature=0.0)
        save_wire('warmup')
        (OUT / 'warmup.stats.json').write_text(json.dumps(vars(stats), indent=2) + '\n')
        for pp in b._single_prompt_lengths(request):
            label = f'pp{pp}'
            await health(label + '.before')
            print(f'{label}: begin', flush=True)
            try:
                metrics = await b._run_external_single_test(
                    client=client, prompt=b._generate_external_prompt(pp, request.context_profile),
                    max_tokens=request.generation_length)
            finally:
                save_wire(label)
            result = dict(test_type='single', pp=metrics['prompt_tokens'], requested_pp=pp,
                          tg=request.generation_length, **metrics)
            results.append(result)
            (OUT / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
            print(json.dumps(result), flush=True)
        await health('final')
    finally:
        await client.aclose()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
