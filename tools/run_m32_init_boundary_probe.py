"""Diagnostic: does the required chain-only hook cover native MTP initialization?

No serving or source patch. Uses a real model-sampled first response token and
sets a diagnostic recipe stop to its exact decoded string, NOT retokenized IDs.
It intentionally stops before another generation call or quiescence drain.
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--omlx', type=Path, default=Path('/tmp/ds41f-m32-omlx'))
p.add_argument('--recipe', type=Path, default=Path('/tmp/ds41f-m32-recipe'))
p.add_argument('--checkpoint', type=Path, default=Path('/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash'))
a = p.parse_args()
assert subprocess.check_output(['git', '-C', str(a.omlx), 'rev-parse', 'HEAD'], text=True).strip() == '4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40'
sys.path.insert(0, str(a.omlx))
import deepseek_recipe as d
import deepseek_recipe._native as native
import mlx.core as mx
import omlx.scheduler
from mlx_lm.generate import BatchGenerator, generation_stream
from mlx_lm.sample_utils import make_sampler
from omlx.patches.deepseek_v41.loading import load
from omlx.patches.mlx_lm_mtp import batch_generator as mtp, cache_rollback, prompt_priming
assert mtp.apply() and cache_rollback.apply()
from tokenizers import Tokenizer
stop_decoder = Tokenizer.from_file(str(a.recipe / 'static/tokenizers/v41/tokenizer.json'))
t = d.Tokenizer.from_file(str(a.recipe / 'static/tokenizers/v41/tokenizer.json'))
request = d.ChatCompletionRequest({'model':'v41', 'messages':[{'role':'user','content':'Say hello in one short sentence.'}], 'thinking':{'type':'disabled'}}).convert(d.ConversionOptions(default_thinking_mode=False))
ids = d.DeepseekV41Encoding().with_tokenizer(t).encode(request.conversation)
result = dict(scope='Actual model initialization diagnostic, NOT guarded-MTP protocol qualification', prompt_ids=ids,
              native_path=native.__file__, native_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
              chain_verify_calls=0, parser=None, status='RUNNING')
original_init = mtp._post_init_mtp
original_chain = mtp._run_verify_cycle_chain
processor = None

def chain(*args, **kwargs):
    result['chain_verify_calls'] += 1
    return original_chain(*args, **kwargs)
mtp._run_verify_cycle_chain = chain

def init(gb, **kwargs):
    global processor
    main = int(gb._next_tokens.tolist()[0])
    stop = stop_decoder.decode([main], skip_special_tokens=False)
    assert stop and '\ufffd' not in stop, 'first token is not a complete decoded string'
    opts = d.ParsingOptions()
    opts.stop_sequences = [stop]
    processor = d.StreamProcessor(d.ChatCompletionChunkGenerator('init-boundary','v41',True,False), opts, t)
    before = processor.semantic_snapshot()
    preview = processor.preview_tokens([main])
    assert processor.semantic_snapshot() == before
    assert preview.terminal_kind == 'STOP_SEQUENCE' and preview.safe_token_count == 0
    result.update(first_model_token=main, stop_string=stop, stop_selection='diagnostic from actual sampled ID; no stop ID tokenization',
                  prediction=dict(kind=preview.terminal_kind, index=preview.completing_token_index, safe=preview.safe_token_count, start=preview.source_start, end=preview.source_end),
                  before_init=dict(history=len(gb.tokens[0]), target=[int(c.size()) for c in gb.prompt_cache], parser=before))
    ret = original_init(gb, **kwargs)
    state = gb._omlx_mtp_state
    result['after_init_before_emission'] = dict(history=len(gb.tokens[0]), target=[int(c.size()) for c in gb.prompt_cache],
                                              dspark=[int(c.offset) for c in state.mtp_cache], queue=[[int(tok), src] for tok, lp, src in state.queue],
                                              parser=processor.semantic_snapshot(), stats=dataclasses.asdict(state.stats))
    assert result['after_init_before_emission']['target'] == [len(gb.tokens[0])+1]*40
    assert result['chain_verify_calls'] == 0
    assert processor.semantic_terminal is None
    return ret
mtp._post_init_mtp = init
model = None
try:
    model, _ = load(a.checkpoint, preserve_mtp=True, engram_ssd_offload=True)
    lm = model.language_model
    lm.configure_mtp(True, 5)
    mx.random.seed(3201)
    with mx.stream(generation_stream):
        cache = lm.make_cache()
        for start in range(0, len(ids)-1, 2048):
            logits = lm(mx.array([ids[start:min(start+2048,len(ids)-1)]]), cache=cache)
            mx.eval(logits)
        bg = BatchGenerator(lm, max_tokens=32, sampler=make_sampler(temp=1.0, top_p=1.0, top_k=0), completion_batch_size=1, prefill_batch_size=1, prefill_step_size=2048, stream=generation_stream)
        uid = bg.insert([[ids[-1]]], caches=[cache], all_tokens=[ids[:-1]])[0]
        pr, gr = bg.next()
        assert not gr and any(r.end_of_prompt for r in pr)
        pr, gr = bg.next()
        mx.synchronize(generation_stream)
        assert len(gr)==1 and gr[0].token == result['first_model_token']
        for out in processor.push(d.InferenceChunk.token(int(gr[0].token))):
            pass
        obs = processor.semantic_terminal
        assert obs.kind == result['prediction']['kind']
        assert (obs.source_start, obs.source_end) == (result['prediction']['start'], result['prediction']['end'])
        result['canonical_observation_matches'] = True
        result['after_first_emission'] = dict(history=len(bg._generation_batch.tokens[0]), parser=processor.semantic_snapshot(), backend_finish=gr[0].finish_reason)
        result['required_unforwarded_terminal_invariant'] = False
        result['status'] = 'CHAIN_ONLY_CLAMP_INSUFFICIENT_AT_INIT'
        bg.remove([uid]); bg.close()
        prompt_priming.drop_ctx(lm)
finally:
    if processor is not None:
        processor.close()
    (ROOT / 'artifacts/m32/init-boundary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
print(json.dumps(result, indent=2, ensure_ascii=False))
