"""M52 ownership and physical block call topology on reduced MLX state."""
import pytest
from test_m52_generation import session, compare


@pytest.mark.parametrize('draft_count', [1, 3, 5, 7, 8, 15, 31])
@pytest.mark.parametrize('prefix', [0, 1, 3, 7, 15, 31])
def test_physical_calls_amortize_not_one_forward_per_accepted_row(draft_count, prefix):
    accepted = min(prefix, draft_count)
    gen, target, cache = session()
    oracle, _, _ = session()
    original = target.forward
    widths = []
    def counted(token, *args, **kwargs):
        widths.append(token.shape[0])
        return original(token, *args, **kwargs)
    target.forward = counted
    drafts = list(range(12, 12 + draft_count))
    if accepted < draft_count:
        drafts[accepted] = 99
    try:
        receipt = gen.speculative_cycle(drafts)
        assert receipt['proposal_acceptance_count'] == accepted
        assert sum(widths) == draft_count + 1
        assert len(widths) == (draft_count + 8) // 8
        assert widths == [min(8, draft_count + 1 - start)
                          for start in range(0, draft_count + 1, 8)]
        assert gen._cache is cache
        oracle.generate(accepted + 1)
        compare(gen, oracle)
    finally:
        gen.close()
        oracle.close()
