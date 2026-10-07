"""CPU-only tests of the opt-in investigation's occupancy accounting."""
import pytest
from tools.profile_mtp_verification import occupancy


def test_expert_local_not_global_rows():
    value = occupancy([0, 1, 2, 3, 0, 4, 2, 5], 256)
    assert value['routes'] == 8
    assert value['active_experts'] == 6
    assert value['rows_per_active_expert_histogram'] == {1: 4, 2: 2}
    assert value['singleton_route_fraction'] == .5
    assert value['max_rows'] == 2


def test_empty_and_concentrated():
    assert occupancy([], 256)['max_rows'] == 0
    assert occupancy([7]*6, 256)['rows_per_active_expert_histogram'] == {6: 1}
    assert occupancy([7]*6, 256)['singleton_route_fraction'] == 0


def test_invalid_route_is_not_hidden():
    with pytest.raises(ValueError):
        occupancy([256], 256)
