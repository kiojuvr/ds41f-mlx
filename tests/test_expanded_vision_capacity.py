"""Expanded ordinary admission only; no full-checkpoint execution claim."""
import json
from types import SimpleNamespace as NS

import pytest

from ds41f_mlx.runtime import multimodal as mm
from ds41f_mlx.serving.capacity import resolve_capacity, context_envelope, vision_capacity
from test_metadata_first_vision import prepare, source, forbidden
from ds41f_mlx.model_execution import processing


def test_expansion_and_off_contract(monkeypatch):
    monkeypatch.setattr(processing, 'image_patch_array', forbidden)
    sources = [source()] * 6
    inp = prepare(sources, ordinary=True, prefix=8200)
    assert len(inp.spans) == 6 and len(inp.token_ids) > 8192
    with pytest.raises(ValueError, match='count'):
        prepare(sources)
    with pytest.raises(ValueError, match='context'):
        prepare([source()], prefix=8200)
    with pytest.raises(ValueError, match='intersects'):
        inp.validate_prefix(inp.spans[4].start + 1, inp.identities()[:4])
    changed = inp.identities()[:4]
    changed[2]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='identity'):
        inp.validate_prefix(inp.spans[4].start, changed)


@pytest.mark.parametrize('budget,limit,reason', [
    ('MAX_REQUEST_IMAGE_BYTES', 1, 'byte budget'),
    ('MAX_REQUEST_PIXELS', 400*300, 'pixel budget'),
    ('MAX_REQUEST_PATCHES', 1, 'patch budget'),
])
def test_aggregate_preparation_rejection(monkeypatch, budget, limit, reason):
    monkeypatch.setattr(mm, budget, limit)
    monkeypatch.setattr(processing, 'image_patch_array', forbidden)
    with pytest.raises(ValueError, match=reason):
        prepare([source(), source()], ordinary=True)


def test_actual_aggregate_pixel_and_patch_budgets(monkeypatch):
    monkeypatch.setattr(processing, 'image_patch_array', forbidden)
    # Five maximum-sized sources exceed the retained four-image pixel budget,
    # while count alone is no longer an admission rule.
    with pytest.raises(ValueError, match='aggregate decoded pixel budget'):
        prepare([source(size=(2048, 2048))] * 5, ordinary=True)
    # Small originals still incur the checkpoint minimum resize/patch work.
    with pytest.raises(ValueError, match='aggregate image patch budget'):
        prepare([source(size=(1, 1))] * 30, ordinary=True)


def test_preparation_checkpoint_bound_before_patches(monkeypatch):
    from ds41f_mlx.model_execution.config import ModelConfig
    monkeypatch.setattr(processing, 'image_patch_array', forbidden)
    config = ModelConfig(vision_n_layers=1, max_seq_len=100)
    with pytest.raises(ValueError, match='context envelope'):
        mm.prepare_multimodal([7, config.image_token_id, 8, 9], [source()], config, ordinary=True)


def test_actual_expansion_output_checkpoint_and_text_limits(tmp_path):
    inp = prepare([source()] * 6, ordinary=True, prefix=8200)
    prompt = len(inp.token_ids)
    (tmp_path/'config.json').write_text(json.dumps({'max_position_embeddings': prompt+32}))
    request = NS(token_ids=inp.token_ids, multimodal=inp, inference_options=NS(max_tokens=32))
    request.conversation_request = NS(resolve_max_tokens=lambda *_: request.inference_options.max_tokens)
    budget = resolve_capacity(request, checkpoint=tmp_path, ordinary=True)
    assert budget['implemented_total_tokens'] == prompt+32
    assert 'qualified_total_tokens' not in budget
    assert budget['expanded_capacity_model_execution'] == 'unverified'
    request.inference_options.max_tokens = 33
    with pytest.raises(ValueError, match='requested output exceeds'):
        resolve_capacity(request, checkpoint=tmp_path, ordinary=True)
    assert resolve_capacity(request, checkpoint=tmp_path, ordinary=True, automatic=True)['max_tokens'] == 32
    (tmp_path/'config.json').write_text(json.dumps({'max_position_embeddings': prompt-1}))
    with pytest.raises(ValueError):
        resolve_capacity(request, checkpoint=tmp_path, ordinary=True, automatic=True)
    (tmp_path/'config.json').write_text(json.dumps({'max_position_embeddings': 2_097_152}))
    assert context_envelope(tmp_path) == 1_048_576
    assert context_envelope(tmp_path, multimodal=True) == 8192
    assert context_envelope(tmp_path, multimodal=True, ordinary=True) == 2_097_152
    capability = vision_capacity(tmp_path)
    assert capability['max_images'] == 600
    assert capability['pixel_patch_budget_scope'] == 'all_history_pixels; fresh_patches_with_complete_pair'
    assert capability['historical_discount_qualified'] is True
    assert capability['historical_patch_discount']['supported_pair_layout_required'] is True
    assert capability['historical_patch_discount']['complete_historical_coverage_required'] is True
    assert 'max_projected_pair_bytes' not in capability['historical_patch_discount']
    assert capability['expanded_capacity_model_execution'] == 'unverified'
    assert capability['max_request_patches'] == mm.MAX_REQUEST_PATCHES
