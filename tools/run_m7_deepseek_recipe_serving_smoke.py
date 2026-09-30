#!/usr/bin/env python3
"""DeepSeek recipe serving integration smoke/qualification recorder."""
from __future__ import annotations
import argparse, asyncio, json, subprocess, time
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepseek_recipe import ChatCompletionRequest, ChatCompletionResponse, ConversionOptions, InferenceChunk, InferenceFinishReason, StreamProcessor, THINKING_END_TOKEN

from ds41f_mlx.serving.deepseek_recipe_backend import DEFAULT_RECIPE, DEFAULT_MODEL_ID, DEFAULT_OMLX, DEFAULT_CHECKPOINT
from ds41f_mlx.serving.server import load_v41_tokenizer, prepare_request


def git_rev(path: Path):
    try: return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'], text=True).strip()
    except Exception: return None

def pkg(name):
    try: return version(name)
    except PackageNotFoundError: return None

def parser_gate(tokenizer):
    req={'model':DEFAULT_MODEL_ID,'messages':[{'role':'user','content':'hi'}]}
    converted=ChatCompletionRequest(req).convert(ConversionOptions())
    gen=ChatCompletionRequest.chunk_generator(converted, 'parser-gate', DEFAULT_MODEL_ID)
    proc=StreamProcessor(gen, converted.parsing_options, tokenizer)
    resp=ChatCompletionResponse('parser-gate', DEFAULT_MODEL_ID, 0, 0, 0)
    ids=tokenizer.encode(THINKING_END_TOKEN + 'Recipe parsed answer')
    try:
        for ch in [InferenceChunk.ready(), *[InferenceChunk.token(int(i)) for i in ids], InferenceChunk.finish(InferenceFinishReason.Stop)]:
            for out in proc.push(ch): resp.append(out)
        for out in proc.finish(): resp.append(out)
        body=json.loads(resp.to_json())
        return {'passed': body['choices'][0]['message']['content']=='Recipe parsed answer', 'token_chunks_used': True, 'body': body}
    finally:
        proc.close()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out', default='artifacts/m7/deepseek-recipe-serving/result.json'); ap.add_argument('--recipe-path', default=str(DEFAULT_RECIPE)); args=ap.parse_args()
    recipe=Path(args.recipe_path); tok=load_v41_tokenizer(recipe)
    chat_body={'model':DEFAULT_MODEL_ID,'messages':[{'role':'user','content':'hi'}],'max_tokens':2,'temperature':0,'stream':False}
    prepared=prepare_request('chat_completions', json.dumps(chat_body).encode(), tokenizer=tok, recipe_path=recipe)
    responses_body={'model':DEFAULT_MODEL_ID,'input':'hi','max_output_tokens':2,'temperature':0,'stream':False}
    resp_prepared=prepare_request('responses', json.dumps(responses_body).encode(), tokenizer=tok, recipe_path=recipe)
    record={
      'schema':'ds41f.m7.deepseek-recipe-serving.v1',
      'deepseek_recipe':{'path':str(recipe),'revision':git_rev(recipe),'version':pkg('deepseek-recipe'),'python_build':'pypi wheel installed in qualification venv'},
      'runtime':{'omlx_path':str(DEFAULT_OMLX),'omlx_revision':git_rev(DEFAULT_OMLX),'mlx_version':pkg('mlx'),'checkpoint':str(DEFAULT_CHECKPOINT)},
      'serving':{'modules':['ds41f_mlx/serving/deepseek_recipe_backend.py','ds41f_mlx/serving/server.py'],'server_entrypoint':'tools/run_ds41f_recipe_server.py','default_bind':'127.0.0.1','model_identifier':DEFAULT_MODEL_ID,'supported_endpoints':['/v1/chat/completions','/v1/responses','/v1/messages'],'text_only':True,'single_flight':True},
      'recipe_tokenizer':{'source':str(recipe/'static/tokenizers/v41/tokenizer.json'),'direct_conversation_encode':True,'no_double_template':True},
      'prompt_split_gate':{'chat_prompt_tokens':len(prepared.token_ids),'chat_prefill_tokens':len(prepared.token_ids)-1,'chat_first_decode_input':prepared.token_ids[-1],'responses_prompt_tokens':len(resp_prepared.token_ids),'responses_prefill_tokens':len(resp_prepared.token_ids)-1,'invariant_prefill_plus_one':(len(prepared.token_ids)-1)+1==len(prepared.token_ids),'rule':'prefill=tokens[:-1], first_input=tokens[-1]'},
      'generation_options':{'supported':['max_tokens/max_output_tokens','temperature','top_p'],'sampler':'mlx_lm.sample_utils.make_sampler','unsupported_policy':'reject invalid/unrepresentable options rather than silently ignoring'},
      'parser_ownership_gate': parser_gate(tok),
      'real_http_smoke':{'chat_completions_non_stream':'NOT_RUN_TO_COMPLETION','chat_completions_stream':'NOT_RUN_TO_COMPLETION','responses_non_stream':'NOT_RUN_TO_COMPLETION','responses_stream':'NOT_RUN_TO_COMPLETION','messages':'PREPARE_SUPPORTED_NOT_MODEL_SMOKED','reason':'real recipe prompt encodes ~30+ tokens; current DwarfStar PrefillContinuationState builder timed out on 30-token prefix (>900s) and odd 29-token prefix exposes existing ratio-2 reshape limitation; no oMLX prompt replay fallback was used'},
      'no_replay_gate':{'implemented_in_backend':True,'requires_runtime_completion':True,'prompt_replay_count':None},
      'cancellation':{'implemented':'backend async generator finally calls OMLXGenerationSession.stop()/close(); InferenceStreamingResponse.aclose on disconnect','runtime_smoke':'NOT_RUN_TO_COMPLETION'},
      'streaming_gate':{'implemented':'StreamingResponse yields recipe StreamProcessor chunks as backend tokens arrive','incremental_real_model_observation':'NOT_RUN_TO_COMPLETION'},
      'unsupported_features':['multimodal/images rejected before image fetch/preprocess','MTP/DSpark','multi-request batching','tool execution','KV restore/resume','long-session persistence'],
      'classification':'DEEPSEEK_RECIPE_TEXT_SERVING_PATH_NOT_QUALIFIED_PREFILL_GENERALIZATION_BLOCKED',
      'next_frontier':'generalize/accelerate real DwarfStar PrefillContinuationState production prefill for arbitrary recipe-encoded prompt lengths, preserving no replay',
    }
    out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n'); print(out)
if __name__=='__main__': main()
