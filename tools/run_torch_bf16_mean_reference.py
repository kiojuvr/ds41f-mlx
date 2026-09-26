#!/usr/bin/env python3
"""Reference-only PyTorch BF16 mean/cat subprocess helper for Boundary12c.

Protocol: JSON on stdin, JSON on stdout. BF16 tensors are raw little-endian
uint16 payloads encoded as base64. This helper receives only bounded fixture
bytes, never checkpoint/model authority.
"""
from __future__ import annotations
import base64, hashlib, json, platform, struct, sys


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def words_to_bytes(words) -> bytes:
    return struct.pack('<' + 'H' * len(words), *[int(w) & 0xFFFF for w in words])


def tensor_bf16_bytes(torch, t) -> bytes:
    if str(t.dtype) != 'torch.bfloat16':
        raise TypeError(f'expected torch.bfloat16 output, got {t.dtype}')
    words = t.detach().cpu().contiguous().view(torch.uint16).reshape(-1).tolist()
    return words_to_bytes(words)


def tensor_from_b64(torch, payload: str, shape):
    raw = base64.b64decode(payload)
    # frombuffer preserves exact BF16 bit patterns; clone detaches from read-only bytes.
    x = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).clone().reshape(tuple(shape))
    return raw, x


def mean_one(torch, req, device: str):
    raw, x = tensor_from_b64(torch, req['bf16_base64'], req['shape'])
    if device != 'cpu':
        x = x.to(device)
    in_bytes = tensor_bf16_bytes(torch, x)
    y = x.mean(dim=2)
    out_bytes = tensor_bf16_bytes(torch, y)
    return {
        'executed': True,
        'device': device,
        'input_dtype': str(x.dtype),
        'input_shape': list(x.shape),
        'input_digest': sha(in_bytes),
        'input_roundtrip_exact': in_bytes == raw and sha(raw) == req.get('input_digest'),
        'operation': 'Tensor.mean(dim=2)',
        'explicit_dtype_argument': None,
        'output_dtype': str(y.dtype),
        'output_shape': list(y.shape),
        'output_digest': sha(out_bytes),
        'output_bf16_base64': base64.b64encode(out_bytes).decode('ascii'),
    }


def cat_one(torch, req):
    tensors = []
    input_digests = []
    for item in req['inputs']:
        raw, x = tensor_from_b64(torch, item['bf16_base64'], item['shape'])
        in_bytes = tensor_bf16_bytes(torch, x)
        input_digests.append(sha(in_bytes))
        if in_bytes != raw or sha(raw) != item.get('input_digest'):
            raise ValueError('cat input roundtrip failed')
        tensors.append(x)
    y = torch.cat(tensors, dim=req.get('dim', -1))
    out_bytes = tensor_bf16_bytes(torch, y)
    return {
        'executed': True,
        'device': 'cpu',
        'input_digests': input_digests,
        'operation': 'torch.cat(..., dim=-1)',
        'output_dtype': str(y.dtype),
        'output_shape': list(y.shape),
        'output_digest': sha(out_bytes),
        'output_bf16_base64': base64.b64encode(out_bytes).decode('ascii'),
    }


def main():
    import torch
    req = json.load(sys.stdin)
    meta = {
        'python_executable': sys.executable,
        'python_version': sys.version,
        'platform_machine': platform.machine(),
        'sys_platform': sys.platform,
        'platform_platform': platform.platform(),
        'torch_version': torch.__version__,
        'torch_git_version': getattr(torch.version, 'git_version', None),
        'pytorch_role': 'reference-only bounded framework semantic oracle',
        'pytorch_is_production_dependency': False,
    }
    out = {'ok': True, 'metadata': meta, 'results': {}}
    for r in req.get('requests', []):
        try:
            if r['op'] == 'mean':
                res = {'cpu': mean_one(torch, r, 'cpu')}
                mps_available = bool(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
                res['mps_reference_available'] = mps_available
                if req.get('try_mps') and mps_available:
                    try:
                        res['mps'] = mean_one(torch, r, 'mps')
                        res['mps_reference_executed'] = True
                        res['cpu_mps_exact_if_both_executed'] = res['mps']['output_digest'] == res['cpu']['output_digest']
                    except Exception as e:
                        res['mps_reference_executed'] = False
                        res['mps_error'] = repr(e)
                else:
                    res['mps_reference_executed'] = False
                out['results'][r['name']] = res
            elif r['op'] == 'cat':
                out['results'][r['name']] = {'cpu': cat_one(torch, r)}
            else:
                raise ValueError(f"unknown op {r['op']}")
        except Exception as e:
            out['ok'] = False
            out['results'][r.get('name', '<unnamed>')] = {'executed': False, 'error': repr(e)}
    print(json.dumps(out, sort_keys=True))
    return 0 if out['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
