"""Installed native startup negatives; never mutate model/shared dependencies."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import sysconfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    inspect = 'from ds41f_mlx.execution_identity import inspect_payload; inspect_payload()'
    def probe(prefix='', expected=None):
        result = subprocess.run([sys.executable, '-I', '-c', prefix+inspect],
                                capture_output=True, text=True, timeout=240, cwd='/tmp')
        if expected is None:
            assert result.returncode == 0, result.stderr
            return dict(status='PASS')
        assert result.returncode != 0 and expected in result.stderr, result.stdout+result.stderr
        return dict(status='REJECT', reason=expected, portable_fallback=False)
    cases = {'native_load': probe()}
    cases['extension_import_failure_injection'] = probe(
        'import sys\nfrom importlib.abc import MetaPathFinder\n'
        'class RefuseGLM(MetaPathFinder):\n'
        ' def find_spec(self, fullname, path, target=None):\n'
        '  if fullname == "omlx.custom_kernels.glm_moe_dsa._ext": raise ImportError("injected native loader failure")\n'
        'sys.meta_path.insert(0, RefuseGLM())\n', 'fallback refused')
    cases['mlx_loaded_image_redirect_simulation'] = probe(
        'from ds41f_mlx.runtime import resource_admission as a; from pathlib import Path; '
        'original=a.loaded_images; '
        'a.loaded_images=lambda: [Path("/wrong/libmlx.dylib") if p.name=="libmlx.dylib" else p for p in original()]; ',
        'redirected GLM MLX library')
    imported = 'from omlx.custom_kernels.glm_moe_dsa import fast; '
    cases['load_disabled_simulation'] = probe(imported+'fast._ext=None; ', 'fallback refused')
    cases['symbol_missing_simulation'] = probe(imported+
        'del fast._ext.deepseek_v41_packed_attention; ', 'symbols missing')
    cases['abi_failure_simulation'] = probe(imported+
        'fast._ext.abi_probe=lambda a: (_ for _ in ()).throw(TypeError("ABI mismatch")); ',
        'array ABI incompatible')
    cases['metallib_load_failure_simulation'] = probe(imported+
        'fast._ext.deepseek_v41_packed_attention=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("metallib load failed")); ',
        'GPU resource unavailable')
    cases['uuid_failure_simulation'] = probe(
        'from ds41f_mlx.runtime import resource_admission as a; '
        'original=a._macho_uuid; calls=[]; '
        'a._macho_uuid=lambda data: (calls.append(1) or (b"wrong-build" if len(calls)==1 else original(data))); ',
        'native image UUID')
    site = Path(sysconfig.get_paths()['purelib'])
    root = site/'omlx/custom_kernels/glm_moe_dsa'
    resource = root/'omlx_glm_kernels.metallib'
    data = resource.read_bytes()
    try:
        resource.write_bytes(bytes([data[0]^1])+data[1:])
        cases['metallib_byte_drift'] = probe(expected='executable payload drift')
        resource.unlink()
        cases['missing_metallib'] = probe(expected='executable payload drift')
    finally:
        resource.write_bytes(data)
    cases['restored_native'] = probe()
    args.output.write_text(json.dumps(dict(status='PASS', cases=cases), indent=2)+'\n')


if __name__ == '__main__':
    main()
