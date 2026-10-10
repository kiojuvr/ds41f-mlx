"""Installed boundary negatives and the real 10.48 -> 10.49 counterexample.

Run using the deployment Python. Mutates only owned installed Python/tokenizer
files, restores them before exit; never changes Homebrew or model weights.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    command = [sys.executable, '-I', '-c',
               'from ds41f_mlx.mtp_identity import snapshot; import json; print(json.dumps(snapshot(),sort_keys=True))']
    def probe(expect=None, code=None):
        cmd = command if code is None else [sys.executable, '-I', '-c', code]
        proc = subprocess.run(cmd,cwd=tempfile.gettempdir(),capture_output=True,text=True,timeout=180)
        if expect is None:
            assert proc.returncode == 0, proc.stderr
            return hashlib.sha256(proc.stdout.encode()).hexdigest()
        assert proc.returncode != 0 and expect in proc.stderr, proc.stdout+proc.stderr
        return dict(status='REJECT', reason=expect)
    baseline = probe()
    site = Path(sysconfig.get_paths()['purelib'])
    cases = {}
    def mutate(path, reason):
        original = path.read_bytes()
        try:
            path.write_bytes(original+b'\n# qualification drift\n')
            return probe(reason)
        finally:
            path.write_bytes(original)
    cases['installed_python'] = mutate(site/'ds41f_mlx/config.py', 'executable payload drift')
    cases['recipe_code'] = mutate(site/'deepseek_recipe/__init__.py', 'executable payload drift')
    cases['tokenizer'] = mutate(Path(sys.prefix)/'share/ds41f-mtp/recipe/static/tokenizers/v41/tokenizer.json', 'recipe tokenizer drift')
    # Inject a changed numerical library digest at the byte-verification boundary;
    # do not mutate shared Homebrew libraries. This is explicitly a simulation.
    cases['semantic_native_digest_simulation'] = probe('semantic native drift',
        "from ds41f_mlx import mtp_setup; original=mtp_setup.sha; "
        "mtp_setup.sha=lambda p: '0'*64 if 'libopencv_imgproc' in str(p) else original(p); "
        "from ds41f_mlx.execution_identity import inspect_payload; inspect_payload()")
    assert probe() == baseline
    # Build/source evidence can disappear entirely at deployment. Point the old
    # build ROOT/SOURCES at an empty directory; no startup consumer may read it.
    assert probe(code="from ds41f_mlx import mtp_setup; from pathlib import Path; "
                      "mtp_setup.ROOT=Path('/nonexistent/checkout'); mtp_setup.SOURCES=Path('/nonexistent/source-exports'); "
                      "from ds41f_mlx.mtp_identity import snapshot; import json; print(json.dumps(snapshot(),sort_keys=True))") == baseline
    cases['source_exports_absent'] = dict(status='PASS', execution_digest_unchanged=True)
    # Adding unrelated site content is not an executable inventory of the world.
    unrelated = site/'qualification_unrelated_package.txt'
    assert not unrelated.exists()
    try:
        unrelated.write_text('unrelated installation\n')
        assert probe() == baseline
    finally:
        unrelated.unlink()
    cases['unrelated_installation'] = dict(status='PASS', execution_digest_unchanged=True)
    old = json.loads((ROOT/'third_party/mtp/normal-local.json').read_text())
    pcre = Path('/opt/homebrew/opt/pcre2/lib/libpcre2-8.0.dylib').resolve()
    digest = hashlib.sha256(pcre.read_bytes()).hexdigest()
    old_pcre = next((path, h) for path,h in old['native_links'].items() if 'pcre2' in path)
    assert '/10.49/' in str(pcre) and digest != old_pcre[1]
    native = site/'deepseek_recipe/_native.abi3.so'
    assert hashlib.sha256(native.read_bytes()).hexdigest() == old['native_sha256']
    cases['actual_homebrew_drift'] = dict(status='PASS', current=str(pcre), historical=old_pcre[0],
        historical_sha256=old_pcre[1], current_sha256=digest,
        same_admitted_native_sha256=old['native_sha256'], not_resealed=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(status='PASS', cases=cases,
        snapshot_sha256=baseline),indent=2)+'\n')
    print('PASS: actual Homebrew drift, exact payload negatives, source/evidence separation')


if __name__ == '__main__':
    main()
