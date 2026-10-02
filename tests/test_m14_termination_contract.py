from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig


def test_decode_config_carries_stop_token_ids():
    cfg = OMLXDecodeConfig(stop_token_ids=(1, 2))
    assert cfg.stop_token_ids == (1, 2)
