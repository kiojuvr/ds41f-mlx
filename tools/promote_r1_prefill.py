"""One-time promotion of ownership/operation fixtures, excluding historical receipts."""
import ast
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'reference/R1/checks'
remove={'_verify_real_smoke_evidence','test_smoke_cli_is_bounded_and_no_timing','test_real_2048_p5_smoke_evidence','test_real_8192_suffix_p5_smoke_evidence','test_p5_cli_is_bounded','test_generation_p6_fence_precedes_frontier_mismatch_in_source','test_p6_smoke_zero_decoder_uses_runner_records_not_model_calls','test_real_pinned_omlx_donor_exact_match_and_mismatch'}
for n in ['prefill_fp8_mlx_p1_p2','prefill_fp8_mlx_p5','prefill_fp8_mlx_p6','prefill_fp8_mlx_p7','suffix_math_regressions']:
    s=(R/f'tests/test_{n}.py').read_text();lines=s.splitlines(keepends=True)
    tree=ast.parse(s);ranges=[]
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in remove:ranges.append((node.lineno,node.end_lineno))
    s=''.join(line for i,line in enumerate(lines,1) if not any(a<=i<=b for a,b in ranges))
    s=s.replace('self.skipTest(str(exc))','raise').replace("self.skipTest(f'pinned oMLX unavailable: {exc}')",'raise')
    (O/f'test_{n}.py').write_text(s)
