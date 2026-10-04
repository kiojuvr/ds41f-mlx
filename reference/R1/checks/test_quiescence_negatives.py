"""Forbidden settlement work is a semantic failure, not merely diagnostic data."""
import pytest
from ds41f_mlx.runtime.mtp_lifecycle import QuiescenceCounters, MTPLifecycleError


@pytest.mark.parametrize('field',['new_verify_cycles','new_proposals','history_replay','full_cache_repack'])
def test_forbidden_settlement_work(field):
    with pytest.raises(MTPLifecycleError):
        QuiescenceCounters(**{field:1}).require_m28_zeroes()


def test_permitted_canonical_materialization():
    QuiescenceCounters(target_forwards=1,dspark_appends=1).require_m28_zeroes()
