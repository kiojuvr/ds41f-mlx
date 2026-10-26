"""Zero-weight characterization of the installed diagnostic's import ordering.

No spec mutation, module reload or admission override. Run in isolated Python.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import sysconfig


def transformers_origin():
    spec = importlib.util.find_spec('transformers')
    module = sys.modules.get('transformers')
    locations = list(spec.submodule_search_locations or ())
    return dict(origin=spec.origin,
        locations=[dict(value=str(p), type=type(p).__name__) for p in locations],
        resolved_locations=[str(Path(p).resolve()) for p in locations],
        actual_directory=str(Path(spec.origin).resolve().parent),
        installed_directory=str(Path(sysconfig.get_paths()['purelib']).resolve()/'transformers'),
        module_file=None if module is None else str(module.__file__),
        module_type=None if module is None else type(module).__name__,
        init_sha256=hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest())


def characterize(*, early):
    from tools.qualify_installed_admission import prove_installed_admission
    result = dict(early=early, before=transformers_origin(), weights_loaded=False)
    if early:
        proof = prove_installed_admission()
        result['admission'] = proof
        from ds41f_mlx import execution_identity as identity
        from ds41f_mlx.model_execution import activation
        authenticated_identity, authenticated_function = identity, activation.pre_woa
        profile = identity.numerical_profile()
    result['before_preparation'] = transformers_origin()
    # Reproduce the original worker's real Recipe/tokenizer preparation, not
    # just import transformers (which alone need not initialize every lazy path).
    import base64
    from ds41f_mlx.serving.server import load_v41_tokenizer, prepare_request
    from ds41f_mlx.mtp_identity import config
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    from ds41f_mlx.serving.paired_checkpoint import validate_pair
    result['after_serving_imports'] = transformers_origin()
    checkpoint = Path('/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017')
    tokenizer = load_v41_tokenizer(Path(sys.prefix)/'share/ds41f-mtp/recipe')
    raw = dict(model='deepseek-v4.1-flash', max_tokens=1, temperature=0,
        messages=[dict(role='user', content=[dict(type='image', source=dict(type='base64',
            media_type='image/jpeg', data=base64.b64encode(
                (checkpoint/'inference/examples/images/corn.jpeg').read_bytes()).decode())),
            dict(type='text', text='Identify this food in one short sentence.')])])
    prepared = prepare_request('messages', json.dumps(raw).encode(), tokenizer=tokenizer,
                               checkpoint=checkpoint, ordinary=True)
    assert len(prepared.token_ids) == 202
    prepared.multimodal.release_sources()
    result['after'] = transformers_origin()
    if early:
        import mlx.core as mx
        assert sys.modules['ds41f_mlx.execution_identity'] is authenticated_identity
        assert activation.pre_woa is authenticated_function
        assert identity.numerical_profile() == profile
        value = activation.pre_woa(mx.ones((1, 2, 1, 128), mx.bfloat16))
        mx.eval(value)
        assert bool(mx.all(mx.isfinite(value)).item())
        result['preserved'] = dict(identity_object=True, function_object=True,
            profile=profile, actual_post_import_pre_woa=True)
    else:
        try:
            prove_installed_admission()
        except ValueError as exc:
            result['late_error'] = str(exc)
        else:
            result['late_error'] = None
    assert result['before']['actual_directory'] == result['after']['actual_directory']
    assert result['after']['actual_directory'] == result['after']['installed_directory']
    assert result['before']['init_sha256'] == result['after']['init_sha256']
    assert result['before']['resolved_locations'] == result['after']['resolved_locations']
    return result


if __name__ == '__main__':
    print(json.dumps(characterize(early=sys.argv[1] == 'early'), indent=2, default=str))
