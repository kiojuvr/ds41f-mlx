"""Finite model-free lifetime observation. Run from repository root."""
import base64
import gc
import json
from pathlib import Path
import resource
import subprocess
import sys
import tracemalloc
import weakref

sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / 'tests')]
from test_http_encoded_soft_caps import sized_png
from test_image_files import body, ref, form
from ds41f_mlx.serving.image_files import ImageFiles, resolve_messages
from ds41f_mlx.serving.server import prepare_request, load_v41_tokenizer
from ds41f_mlx.mtp_identity import config, checkpoint_variant
from ds41f_mlx import execution_identity

manifest = Path(sys.prefix) / 'lib/python3.13/site-packages/ds41f_mlx/execution_manifest.json'
execution_identity._admitted_variant = checkpoint_variant(json.loads(manifest.read_text()))
runtime = config(profile='mtp-serving-v1')
tokenizer = load_v41_tokenizer(runtime.recipe_path)
store = ImageFiles()
data = sized_png(1024 * 1024)
# Use the real bounded multipart parser/validator, not synthetic StoredImage publication.
from httpx import Request
upload = Request('POST', 'http://audit.invalid/v1/files', **form(data=data))
receipt = store.upload(upload.read(), upload.headers['content-type'])
key = receipt['id']
wire = json.dumps(body([ref(key) for _ in range(4)])).encode()
rows = []
tracemalloc.start()

def observe(stage, **extra):
    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    rss = int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(__import__('os').getpid())])) * 1024
    rows.append(dict(stage=stage, python_current=current, python_peak=peak,
                     rss_bytes=rss, process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     **extra))

observe('baseline', wire_bytes=len(wire), stored_bytes=len(data))
payload = resolve_messages(json.loads(wire), store)
values = [b['source']['data'] for b in payload['messages'][0]['content'] if b['type']=='image']
observe('resolved', logical_encoded_bytes=4*len(data), distinct_base64_objects=len({id(v) for v in values}),
        represented_base64_bytes=sum(map(len, values)))
del values, payload
observe('resolved_released')
prepared = prepare_request('messages', wire, tokenizer=tokenizer, ordinary=True,
                           checkpoint=runtime.checkpoint_path, image_files=store)
weak = weakref.ref(prepared)
mm_weak = weakref.ref(prepared.multimodal)
observe('prepared', queued_source_bytes=sum(map(len, prepared.multimodal._sources.values())),
        distinct_encoded_objects=len({id(v) for v in prepared.multimodal._sources.values()}),
        source_count=len(prepared.image_sources))
prepared.multimodal.release_sources()
observe('sources_released_but_prepared_retained', queued_source_bytes=len(prepared.multimodal._sources),
        retained_recipe_source_count=len(prepared.image_sources))
del prepared
observe('prepared_released', prepared_collectible=weak() is None, multimodal_collectible=mm_weak() is None)
# Error path: full production-size guard, not a lowered threshold.
large = sized_png(2*1024*1024)
request = Request('POST', 'http://audit.invalid/v1/files', **form(data=large))
large_key = store.upload(request.read(), request.headers['content-type'])['id']
del request, large
observe('before_rejection')
def reject():
    try:
        resolve_messages(body([ref(large_key) for _ in range(33)]), store)
    except ValueError as exc:
        return str(exc)  # Do not retain exception traceback/request-local tree.
    raise AssertionError('guard did not reject')
error = reject()
assert 'aggregate encoded' in error
observe('rejection_unwound', error=error)
store.clear()
del upload, data, wire
observe('request_and_store_released')
print(json.dumps({'observations':rows,'notes':[
    'tracemalloc excludes Rust Vec/String, Pillow/native, MLX and allocator arenas.',
    'RSS is a sampled process resident set, not macOS physical footprint or a native-allocation counter.',
    'ru_maxrss is a process high-water mark in bytes on macOS; not reset between stages.',
    'No model loaded, no native inference, no global memory-budget or leak-free claim.'
]}, indent=2))
