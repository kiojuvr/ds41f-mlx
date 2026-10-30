"""Model-free witnesses for the deferred patch-cap resource-ownership gap.

No admission override or capacity estimator: these assert logical shape and
Python ownership only, not physical allocation/reservation or native peaks.
"""
from types import SimpleNamespace as NS
import weakref

import mlx.core as mx
import numpy as np
import pytest

from ds41f_mlx.prefill_fp8_mlx.p6_append import _make_segment_tensors
from ds41f_mlx.model_execution.config import ModelConfig
from ds41f_mlx.runtime.multimodal import ImageEmbeddings, prepare_multimodal
from test_metadata_first_vision import source


@pytest.mark.parametrize('size', [(1, 1), (777, 1023), (2048, 2048), (1024, 2048)])
def test_output_residency_uses_expanded_rows_not_patch_count(size):
    config = ModelConfig(dim=5120, vision_n_layers=1, max_seq_len=1_048_576)
    inp = prepare_multimodal([7, config.image_token_id, 8], [source(size=size)],
                             config, ordinary=True)
    span, = inp.spans
    def encode(pixels, h, w, kinds):
        assert pixels.dtype == mx.bfloat16
        assert pixels.shape[0] == h*w == span.grid[0]*span.grid[1]
        # Shape/dtype of the pinned adapter's LLM output; no ViT weights loaded.
        return mx.zeros((len(kinds), inp._config.dim), dtype=mx.bfloat16)
    embeddings = inp.encode_new(NS(config=inp._config), span_encoder=encode)
    _, rows = embeddings.rows[0]
    assert rows.shape == (span.length, 5120)
    assert rows.nbytes == span.length * 5120 * 2
    assert span.length != span.grid[0]*span.grid[1]
    assert not inp._sources


def test_merge_does_not_consume_completed_rows_and_p6_expands_hidden_state():
    dim, count, hc_mult = 5120, 3, 4
    row = mx.full((count, dim), 7, dtype=mx.bfloat16)
    ref = weakref.ref(row)
    embeddings = ImageEmbeddings(((0, row),), 129264)
    del row
    lm = NS(_config=NS(hc_mult=hc_mult),
            embed=lambda ids: mx.zeros((*ids.shape, dim), dtype=mx.bfloat16))
    h, other, pre, ids, hashes, history = _make_segment_tensors(
        lm, mx, [129264]*count, None, None, embeddings, 0)
    mx.eval(h, other, pre)
    assert h.shape == other.shape == (1, count, hc_mult, dim)
    assert pre.shape == (1, count, hc_mult) and pre.dtype == mx.float32
    assert h.nbytes == other.nbytes == count*hc_mult*dim*2
    assert ref() is not None  # Merge referenced/sliced; it did not transfer ownership.
    np.testing.assert_array_equal(np.asarray(h.astype(mx.float32)), 7)
    # A later text-only segment still does not remove the completed image rows.
    text = embeddings.merge(mx.zeros((1, 1, dim), dtype=mx.bfloat16), count)
    mx.eval(text)
    assert ref() is not None and len(embeddings.rows) == 1
    del embeddings
    assert ref() is None  # Python ownership only; MLX backing may outlive handles.
