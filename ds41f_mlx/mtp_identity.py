"""MTP admission: installed execution identity, host compatibility, exact assets.

Build provenance and historical qualification are reported, never startup equality.
There is no local executable seal or checkout source authority.
"""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from .config import load_runtime_config
from .mtp_setup import sha
from .mtp_profile import PROFILE, LIMITS, WEATHER
from .execution_identity import MANIFEST, inventory

CHECKPOINT_CACHE = Path(sys.prefix)/'share/ds41f-mtp/checkpoint-bytes.json'
CHECKPOINT_REVISION = 'dba1be0a40aa45a94ad051997016db3960a90277'
CHECKPOINT = {'config.json':'8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879',
    'model.safetensors.index.json':'74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8',
    'tokenizer.json':'c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b'}


def config(host=None, port=None, *, profile=PROFILE):
    if profile not in (PROFILE, 'mtp-serving-v1'):
        raise ValueError('unknown MTP profile')
    allowed = {'DS41F_CHECKPOINT','DS41F_KV_ROOT','DS41F_HOST','DS41F_PORT',
               'DS41F_MAX_LIVE_SESSIONS','DS41F_TRACE_HISTORY_LIMIT',
               'DS41F_ENABLE_DIAGNOSTIC_ENDPOINTS','DS41F_MODEL_ID'}
    unknown = sorted(k for k in os.environ if
        (k.startswith('DS41F_') and k not in allowed) or k.startswith(('OMLX_','UVICORN_','DYLD_')))
    if unknown:
        raise ValueError(f'unqualified profile environment settings: {unknown}')
    if os.environ.get('WEB_CONCURRENCY', '1') != '1':
        raise ValueError('MTP requires one worker')
    if os.environ.get('DS41F_TRACE_HISTORY_LIMIT', '32') != '32':
        raise ValueError('MTP trace budget is fixed at 32')
    if os.environ.get('PYTHONPATH'):
        raise ValueError('PYTHONPATH unavailable in MTP')
    from .serving.production_sessions import ProductionSessions
    live_limit = 1 if profile == PROFILE else ProductionSessions.MAX_LIVE
    if os.environ.get('DS41F_MAX_LIVE_SESSIONS', str(live_limit)) != str(live_limit):
        raise ValueError(f'{profile} logical conversation retention is fixed at {live_limit}; execution remains single-active')
    cfg = load_runtime_config()
    host = host if host is not None else cfg.host
    port = port if port is not None else cfg.port
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('MTP requires a fixed port')
    if profile == PROFILE and host != '127.0.0.1':
        raise ValueError('MTP requires literal 127.0.0.1 and a fixed port')
    if cfg.enable_diagnostics or cfg.model_id != 'deepseek-v4.1-flash':
        raise ValueError('diagnostics/custom model configuration unavailable in MTP')
    spec = importlib.util.find_spec('omlx')
    if spec is None or not spec.origin:
        raise ValueError('install MTP execution dependencies first')
    parent = Path(spec.origin).resolve().parent.parent
    if not parent.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError('oMLX must be installed inside the selected environment')
    return replace(cfg, omlx_path=parent,
        recipe_path=Path(sys.prefix)/'share/ds41f-mtp/recipe', host=host, port=port,
        max_live_sessions=live_limit, trace_history_limit=32, mtp='ON', dspark='ON', speculative_decode='ON')


def qualified():
    return json.loads(MANIFEST.read_text())


def snapshot():
    from .execution_identity import inspect_payload
    authority, _ = inspect_payload()
    return dict(schema='ds41f.execution-identity.v2', profile=PROFILE, limits=LIMITS,
                tool=WEATHER, files=authority['files'], payloads=authority['payloads'],
                semantic_native=authority['semantic_native'], tokenizer_sha256=authority['tokenizer_sha256'])


def inspect(cfg, *, checkpoint=True):
    from .execution_identity import inspect_payload
    if checkpoint and not os.environ.get('DS41F_CHECKPOINT'):
        raise ValueError('DS41F_CHECKPOINT must explicitly identify the official external asset')
    authority, compatibility = inspect_payload()
    if authority['profile'] != 'mtp':
        raise ValueError('installed payload is not the MTP execution profile')
    hardware = subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
    memory = int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True))
    if hardware != 'Apple M3 Ultra' or memory < 500_000_000_000:
        raise ValueError('MTP qualification requires M3 Ultra 512 GB class')
    assets = {}
    if checkpoint:
        metadata_files = {**CHECKPOINT, **authority['checkpoint_metadata']}
        if {p.name for p in cfg.checkpoint_path.glob('*.json')} != set(metadata_files) or any(
                (cfg.checkpoint_path/name).exists() for name in ('chat_template.jinja', 'chat_templates')):
            raise ValueError('unqualified checkpoint/tokenizer configuration files')
        for file, digest in metadata_files.items():
            if sha(cfg.checkpoint_path/file) != digest:
                raise ValueError(f'unqualified checkpoint metadata: {file}')
        assets = checkpoint_inventory(cfg.checkpoint_path)
        if assets != authority['checkpoint']:
            raise ValueError('unqualified checkpoint shard identity')
        verify_checkpoint_bytes(cfg.checkpoint_path, assets)
        assets = dict(assets, verification='full-byte SHA256; installation cache only for unchanged file identity')
    actual = dict(schema='ds41f.execution-identity.v2', profile=PROFILE, limits=LIMITS,
                  tool=WEATHER, files=authority['files'], payloads=authority['payloads'], semantic_native=authority['semantic_native'],
                  tokenizer_sha256=authority['tokenizer_sha256'])
    identity_digest = hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest()
    return dict(status='PASS', profile=PROFILE, environment_valid=True,
                release_qualified=False, identity_sha256=identity_digest,
                config=cfg.to_json(), identity=actual, checkpoint=assets,
                build=authority['provenance'], compatibility=compatibility,
                hardware=dict(cpu=hardware, memory_bytes=memory),
                actual_origins={name:importlib.util.find_spec(name).origin for name in
                                ('ds41f_mlx','omlx','deepseek_recipe','deepseek_recipe._native','mlx_lm')})


def checkpoint_inventory(root):
    shards = []
    for p in sorted(root.glob('model-*.safetensors')):
        meta = root/'.cache/huggingface/download'/(p.name+'.metadata')
        lines = meta.read_text().splitlines() if meta.exists() else []
        if (len(lines) < 2 or lines[0] != CHECKPOINT_REVISION or len(lines[1]) != 64 or
                any(c not in '0123456789abcdef' for c in lines[1])):
            raise ValueError('qualified checkpoint source revision/LFS hash metadata required')
        shards.append(dict(name=p.name, bytes=p.stat().st_size, source_revision=lines[0], lfs_sha256=lines[1]))
    if len(shards) != 48 or len({s['source_revision'] for s in shards}) != 1:
        raise ValueError('official 48-shard source inventory required')
    return dict(metadata=CHECKPOINT, shards=shards,
                verification='metadata/source inventory only; weights not freshly rehashed')


def verify_checkpoint_bytes(root, assets):
    """Rehash once per installation/file identity; never trust arbitrary LFS metadata."""
    cache = CHECKPOINT_CACHE
    def file_identity(path):
        s = path.stat()
        return [str(path.resolve()), s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]
    files = {s['name']:file_identity(root/s['name']) for s in assets['shards']}
    expected = {s['name']:s['lfs_sha256'] for s in assets['shards']}
    saved = json.loads(cache.read_text()) if cache.exists() else {}
    if saved == dict(files=files, sha256=expected):
        return
    for name, digest in expected.items():
        h = hashlib.sha256()
        with (root/name).open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024), b''):
                h.update(chunk)
        if h.hexdigest() != digest or file_identity(root/name) != files[name]:
            raise ValueError(f'unqualified checkpoint bytes: {name}')
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(dict(files=files, sha256=expected),sort_keys=True)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['inspect'])
    parser.parse_args(argv)
    print(json.dumps(inspect(config()),indent=2))


if __name__ == '__main__':
    main()
