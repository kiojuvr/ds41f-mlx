"""Read-only identity/state receipt; run in the isolated client environment."""
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import psutil

root = Path('/Volumes/SDXC-512/ds41f-1017')
site = root / 'lib/python3.13/site-packages'
checkpoint = Path('/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017')
manifest_path = site / 'ds41f_mlx/execution_manifest.json'
manifest = json.loads(manifest_path.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
process = psutil.Process(14750)
settings = Path.home() / '.config/ds41f/operator.json'
identity = dict(
    captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    platform=platform.platform(), hardware=dict(model='Mac Studio', chip='Apple M3 Ultra',
        cores=32, memory_bytes=psutil.virtual_memory().total),
    os=subprocess.check_output(['sw_vers'], text=True),
    process=dict(pid=process.pid, ppid=process.ppid(), create_time=process.create_time(),
        command=process.cmdline(), executable=process.exe(),
        environment={k:v for k,v in process.environ().items()
                     if k.startswith('DS41F_') or k == 'VIRTUAL_ENV'}),
    runtime=str(root), installed_versions={d.metadata['Name']:d.version
        for d in importlib.metadata.distributions(path=[str(site)])},
    runtime_source_commit=manifest['provenance']['source_commit'],
    numerics=manifest['numerics'], recipe_provenance=manifest['provenance']['recipe_source'],
    native_build=manifest['provenance']['native_build'],
    manifest_sha256=sha(manifest_path),
    direct_urls={name:json.loads((next(site.glob(name+'-*.dist-info')) / 'direct_url.json').read_text())
                 for name in ['ds41f_mlx','deepseek_recipe']},
    checkpoint=str(checkpoint),
    checkpoint_metadata_sha256={p.name:sha(p) for p in [checkpoint/'config.json',
        checkpoint/'model.safetensors.index.json',checkpoint/'tokenizer.json']},
    checkpoint_files={p.name:dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
        for p in sorted(checkpoint.iterdir()) if p.is_file()},
    operator_settings=json.loads(settings.read_text()), operator_settings_sha256=sha(settings),
    # Metadata inventory avoids re-reading multi-GB libraries during timed trials.
    installation_files={str(p.relative_to(root)):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
        for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts},
)
Path(sys.argv[1]).write_text(json.dumps(identity, indent=2)+'\n')
