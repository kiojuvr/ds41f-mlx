"""Bounded acceptance observer around canonical installed serve.main.

No runtime implementation changes. Forward calls unchanged to the admitted
nanobind entries; count only model callers, excluding admission/synthetic smoke.
Receipt is written on normal canonical shutdown. Run with deployment Python -I.
"""
import argparse
from collections import Counter
import inspect
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--profile', choices=('mtp-serving-v1', 'standard-off'), default='mtp-serving-v1')
    args = p.parse_args()
    from ds41f_mlx.execution_identity import inspect_payload
    from ds41f_mlx.mtp_identity import snapshot
    authority, compatibility = inspect_payload()
    assert compatibility['glm_profile'] == 'native'
    from omlx.custom_kernels.glm_moe_dsa import fast
    ext = fast._ext
    counts = Counter()
    examples = {}
    originals = {}
    def observer(name, original):
        def call(*args, **kwargs):
            frames = inspect.stack(context=0)
            callers = [f for f in frames[1:] if '/omlx/patches/deepseek_v41/' in f.filename
                       or '/ds41f_mlx/prefill_fp8_mlx/' in f.filename
                       or '/ds41f_mlx/model_execution/' in f.filename]
            result = original(*args, **kwargs)
            if callers:
                counts[name] += 1
                shapes = [list(a.shape) for a in args if hasattr(a, 'shape')]
                multirow = name == 'deepseek_v41_packed_attention' and shapes and shapes[0][2] > 1
                key = name+(':multirow' if multirow else '')
                if key not in examples:
                    examples[key] = dict(shapes=shapes,
                        callers=[dict(file=f.filename, function=f.function, line=f.lineno) for f in callers],
                        layer_indices=[getattr(f.frame.f_locals.get('self'), 'layer_idx', None) for f in callers])
                if multirow:
                    counts['packed_attention_multirow'] += 1
            del frames, callers
            return result
        return call
    identity = snapshot()
    for name in fast.NATIVE_SYMBOLS:
        original = getattr(ext, name)
        originals[name] = original
        setattr(ext, name, observer(name, original))
    result = None
    try:
        from ds41f_mlx.serve import main as serve
        result = serve(['--profile', args.profile])
    finally:
        for name, original in originals.items():
            setattr(ext, name, original)
        receipt = dict(schema='ds41f.native-selected-path.v1', launcher_result=result,
            identity=identity, bundle=authority['glm'], compatibility=compatibility,
            successful_model_native_calls=dict(counts), examples=examples,
            observer='forwarding counters on original admitted extension; model callers only; no admission smoke counts')
        args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')


if __name__ == '__main__':
    main()
