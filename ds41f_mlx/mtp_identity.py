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
CHECKPOINT_REVISION = '1bae0e1088993b9b82de08f59de5be5a34e42170'
CHECKPOINT = {'config.json':'8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879',
    'model.safetensors.index.json':'74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8',
    'tokenizer.json':'c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b'}


# Transferred from DeepSeek-V4.1-Flash-1017 at 166a211fa98f86f223aa346b67a4c5a16f775696.
# Shard names/sizes and model metadata are identical to the inherited authority.
FLASH_1017_HASHES = tuple('''
13bd121075f0b68cb7e8e8021a1011d42a81a83f557a2602fd9c07d52fddb0ec
43616eebd88dd4f1f1b642772356fc160478d1d9d04edd240431e7d065d2eda7
41611994fc9b27829b757d72257570d65a976fe65b8c6df8641adcd139d65ded
e8e901b3cb1ab46f32729c7eaab37bd774a09094f0a02c51a60e883f779f2c39
c5df71c02d6a22bcb20a0dbbcda9c6b65aa104f44415102a9a6a03bee92bb9de
a5d42afde2aee33619fb7a674b4c737c38d1f1d47a636a26f3edfb3b54164167
24856942add8f73f5e1fbaea1e9c9b6ff597fd1b4aa4651d789833dd3a5a2ab9
7bf5f7afb5d2bacad49feffb049989fb365d6dbf96986e76265bd7a7a3469564
e6ae47e9daca138fda8c04f833c8328eb21671c37d1c8e2c3f7a82aac26649f6
835e9a8b0a28cc571cb1cf9e6014b3d5eb84ac41990a8aa4519e1ac435d7e23c
5df968edefe59033e451ff8512bef57da2cf97b57b4e7a1380118489dd02915b
4f97bf292f58178ecdf593b55252a401d6e9102f18625a4fccc52b8ee57e7be0
1b8567854887396840eb5392ca763101308064a142cb36156159096e581659eb
e5ba6d7e4380728ad70b8208676075a898f80a64c3618060f1b81e4218a159d1
f4b29d936024e68f44fb14d0ffffb15b42b27be04b4a0296acdae695afa94c30
3aa67228351c80e708dab4ac7f6e56b19000649cce2cd655def3168a3de4115c
b704eec7b7aa1bad6c9d57fa7f61f2db28862e8e5fc2468fd9d872689c778c7d
1405cf03376739f46e72e6cdd4d404969f1b94c717bb01782a300e2fbbd17e29
a84beaded1913e8d6119ac111f25040c8ec42e59f200dbe04c3a991cac512669
038914f79cf5499da90f0bfb8e1b7761cc37d5f4dd132e2431a6d02c9fb9fdfe
2ba49f59c8cec144e41370f9f820f268899d0b5cd0501e881908ca65bccd2dc2
52f12cceca5e2fdd0191dfbd33b808b6c7ce844cf9351c119413c5ba3bcd28c8
d6e47e8b7147c64eb6619732694dda811e1a9f8af1894977919af9522724ca31
9d611d65374d37f5e02fdae14e680cb69bf55af8a8478802ac465b97c65be22b
8db532a7cbb23cd4b5aa775636ca13af95b758501290fb935074348f660a29cd
2d6d7261937d2a98c7398a55ed48dbf129d9b4f188aa9148e46ea9f720255115
8e76de6a5cdc7ac0923526d1c1a2e6459c55bbfa692793b29cdfe07b4f9fbc52
3c0e5ad6a4e8e811c1a55fd8f5b59c9fa329b241c671997c28fc7a01944d5a77
392a950bcb062b89f683abc52d49a17e92c66f80cd1cb6fb92b71091da75319b
0b44d1364e578fc26dc0cc86564e49a92871a51063fb7f2f3b51cb020de44ab0
d3230ffc32e11c5a34505c5acaeaff642aed5c770b297695e1f7a94071d361ac
96d284016ac885415859346d971c0de43bd9e7947337edfb53c34462866ef1d7
313953b3b6aaa6d4c60bbcfc177dda5d8d1380c9fe72d82560ba078f55278a8d
98eb27655774796bb3feadb35ec64a9849104b88c4c6ae7cabf6fc1ee317a056
4a321a28c048389039e4595641fe0b1f130428c61fc8f655545afe0083d37cd9
97c7910a8a14002dd941f45f6ddb2bd335eaf6fa05f4a76b2ff9285d6557b65d
d37ee013b805344619536e299c73adf92df03744aa0a242f2c5465590015e5be
7713bf7a78319eb1c6214b312e0a2da8cac88d77dce851d50f1113d78f8975a4
658a77d47d217995b9b78612e69ee0d0e33f6c2240f4ec629fecdca3ecb03f4c
6a9dbcc8d5894ba614b5e962764f45f6d0df7f14f1287b17f173d1f2cd762186
458308f08ed29b34850bd6b9d10bf769620729d3725c77f7bc0c1d14e01b47cc
3d283a4ed6eb2d5013bacc97c3500361926a3261957943434513304df52adfd7
4ae12b941c3facf8616ce62dd45bc6868076c2aeb82106f98123ea6c5bb64182
3e8ac315c2bcca3e83c5e6e25c5758a75c03732063a6ff0c1b0e24b9ff49f6e6
9c73b27c4343236790008979d5e0b565f687aa64440031b9e096fed1ab8039c5
1dccc19a3741972c661de734572dc7d4b899e6e99bfbdf3ae4e057c707e5b339
76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5
9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd
'''.split())
FLASH_1017_AUXILIARY = {'configuration.json':
    '563571360b3c5a4cf9b14b09732f686a8ed9da2d0dfcd1728d72140b4aeb1467'}


def flash_1017_inventory(original):
    if len(original['shards']) != 48 or len(FLASH_1017_HASHES) != 48:
        raise ValueError('official 48-shard authority required')
    return dict(original, shards=[dict(s, source_revision=CHECKPOINT_REVISION, lfs_sha256=h)
                                  for s, h in zip(original['shards'], FLASH_1017_HASHES)])


def admit_checkpoint(root, authority):
    """Admit only Flash-1017, including ModelScope's pinned auxiliary file.

    Optional HF receipts never substitute for full-byte verification. The installed
    package must explicitly admit the fixed 1017 inventory; no legacy fallback.
    """
    assets = authority['checkpoint']
    if assets != flash_1017_inventory(assets):
        raise ValueError('Flash-1017 checkpoint is not admitted by this installed package')
    metadata = {**CHECKPOINT, **authority['checkpoint_metadata']}
    names = {p.name for p in root.glob('*.json')}
    if names not in (set(metadata), set(metadata) | set(FLASH_1017_AUXILIARY)) or any(
            (root/name).exists() for name in ('chat_template.jinja', 'chat_templates')):
        raise ValueError('unqualified checkpoint/tokenizer configuration files')
    for name, digest in {**metadata, **{k:v for k,v in FLASH_1017_AUXILIARY.items() if k in names}}.items():
        if sha(root/name) != digest:
            raise ValueError(f'unqualified checkpoint metadata: {name}')
    for shard in assets['shards']:
        receipt = root/'.cache/huggingface/download'/(shard['name']+'.metadata')
        if receipt.exists() and receipt.read_text().splitlines()[:2] != [CHECKPOINT_REVISION, shard['lfs_sha256']]:
            raise ValueError('unqualified checkpoint source revision/LFS metadata')
    if {p.name for p in root.glob('*.safetensors')} != {s['name'] for s in assets['shards']} or any(
            (root/s['name']).stat().st_size != s['bytes'] for s in assets['shards']):
        raise ValueError('unqualified checkpoint shard inventory/size')
    verify_checkpoint_bytes(root, assets)
    return assets


def config(host=None, port=None, *, profile=PROFILE):
    if profile not in (PROFILE, 'mtp-serving-v1'):
        raise ValueError('unknown MTP profile')
    allowed = {'DS41F_CHECKPOINT','DS41F_KV_ROOT','DS41F_HOST','DS41F_PORT',
               'DS41F_MAX_LIVE_SESSIONS','DS41F_TRACE_HISTORY_LIMIT',
               'DS41F_ENABLE_DIAGNOSTIC_ENDPOINTS','DS41F_MODEL_ID',
               'DS41F_DEFAULT_REASONING_EFFORT'}
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
    authority, compatibility = inspect_payload()
    return dict(schema='ds41f.execution-identity.v2', profile=PROFILE,
                glm_profile=compatibility['glm_profile'], limits=LIMITS,
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
        assets = admit_checkpoint(cfg.checkpoint_path, authority)
        assets = dict(assets, verification='full-byte SHA256; installation cache only for unchanged file identity')
    actual = dict(schema='ds41f.execution-identity.v2', profile=PROFILE,
                  glm_profile=compatibility['glm_profile'], limits=LIMITS,
                  tool=WEATHER, files=authority['files'], payloads=authority['payloads'], semantic_native=authority['semantic_native'],
                  tokenizer_sha256=authority['tokenizer_sha256'])
    if checkpoint:
        # Paired state must never cross checkpoint or numerical-profile identities.
        actual['checkpoint'] = assets
        actual['numerics'] = 'flash-1017-pre-woa-fp8-v1'
    identity_digest = hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest()
    return dict(status='PASS', profile=PROFILE, glm_profile=compatibility['glm_profile'], environment_valid=True,
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
