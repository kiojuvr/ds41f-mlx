#!/usr/bin/env python3
"""M14 generation termination qualification."""
from __future__ import annotations
import argparse, contextlib, json, os, socket, subprocess, sys, time, urllib.request, urllib.error
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT, DEFAULT_OMLX
from ds41f_mlx.serving.deepseek_recipe_backend import DEFAULT_RECIPE, DEFAULT_MODEL_ID, git_rev
from tools.run_m13_repeated_tool_agent_qualification import tool_def, first_body, body_after_a, body_after_b, call_from, stub

EOS_TEXT='<｜end▁of▁sentence｜>'

def free_port():
    s=socket.socket(); s.bind(('127.0.0.1',0)); p=s.getsockname()[1]; s.close(); return p

def http_json(method,url,data=None,timeout=1800):
    body=None if data is None else json.dumps(data).encode(); req=urllib.request.Request(url,data=body,method=method,headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=timeout) as r:
            raw=r.read().decode(); return r.status,(json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw=e.read().decode(errors='replace')
        try: return e.code,json.loads(raw)
        except Exception: return e.code,{'raw':raw}

def start_server(args,port):
    env=os.environ.copy(); env['DS41F_ENABLE_DIAGNOSTIC_ENDPOINTS']='1'; env['DS41F_MAX_LIVE_SESSIONS']='4'
    return subprocess.Popen([sys.executable,str(ROOT/'tools/run_ds41f_recipe_server.py'),'--host','127.0.0.1','--port',str(port),'--checkpoint',str(args.checkpoint),'--omlx-path',str(args.omlx_path),'--recipe-path',str(args.recipe_path)],cwd=ROOT,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

def wait_health(base):
    deadline=time.time()+180
    while time.time()<deadline:
        try:
            c,_=http_json('GET',base+'/health',None,5)
            if c==200: return
        except Exception: pass
        time.sleep(.5)
    raise RuntimeError('health timeout')

def diag(base,sid):
    return http_json('GET',base+'/_ds41f/diagnostics')[1]['sessions'][sid]

def new_session(base):
    c,s=http_json('POST',base+'/v1/sessions',{}); assert c==200,(c,s); return s['id']

def stop_summary(base,sid,resp,eos_id):
    s=diag(base,sid); last=s['last_turn']; toks=last['generated_tokens']; content=resp['choices'][0]['message'].get('content') or ''
    return {'finish_reason':last['finish_reason'],'recipe_finish_reason':resp['choices'][0]['finish_reason'],'frontier':last['frontier_after_commit'],'generated_count':len(toks),'tail_tokens':toks[-8:],'last_token_is_eos': bool(toks and toks[-1]==eos_id),'eos_leaked_in_content': EOS_TEXT in content,'content_prefix':content[:300],'content_len':len(content),'replay':s['diagnostics']['m8']['total_prompt_replay_count'],'repack':s['diagnostics']['m8']['total_full_cache_repack_count'],'cache_ok':s['diagnostics']['m8']['all_cache_offsets_equal_frontier']}

def plain_case(base,eos_id):
    sid=new_session(base)
    body={'model':DEFAULT_MODEL_ID,'messages':[{'role':'user','content':'Answer in one short sentence: what is 2+2?'}],'reasoning_effort':'none','temperature':0,'max_tokens':96}
    c,r=http_json('POST',f'{base}/v1/sessions/{sid}/chat/completions',body); assert c==200,(c,r)
    summ=stop_summary(base,sid,r,eos_id)
    # exact-prefix next-turn continuation after EOS-retaining history
    body2={'model':DEFAULT_MODEL_ID,'messages':body['messages']+[{'role':'assistant','content':r['choices'][0]['message'].get('content')},{'role':'user','content':'Thanks.'}],'reasoning_effort':'none','temperature':0,'max_tokens':8}
    c2,r2=http_json('POST',f'{base}/v1/sessions/{sid}/chat/completions',body2)
    summ['next_turn_status']=c2; summ['next_turn_ok']=c2==200
    summ['ok']=summ['finish_reason']=='stop' and summ['last_token_is_eos'] and not summ['eos_leaked_in_content'] and summ['replay']==0 and summ['repack']==0 and summ['cache_ok'] and c2==200
    return summ

def single_tool_case(base,eos_id):
    sid=new_session(base)
    b0={'model':DEFAULT_MODEL_ID,'messages':[{'role':'user','content':'Use lookup_weather for Paris, then answer concisely after I provide the result.'}], 'tools':[tool_def()],'tool_choice':{'type':'function','function':{'name':'lookup_weather'}},'reasoning_effort':'none','temperature':0,'max_tokens':96}
    c,r1=http_json('POST',f'{base}/v1/sessions/{sid}/chat/completions',b0); call=call_from(r1); res=stub(call)
    b1={'model':DEFAULT_MODEL_ID,'messages':[b0['messages'][0],{'role':'assistant','content':None,'tool_calls':[{'id':call['id'],'type':'function','function':{'name':call['name'],'arguments':call['arguments']}}]},{'role':'tool','tool_call_id':call['id'],'content':res},{'role':'user','content':'Now answer concisely.'}], 'tools':[tool_def()],'tool_choice':'auto','reasoning_effort':'none','temperature':0,'max_tokens':96}
    c,r2=http_json('POST',f'{base}/v1/sessions/{sid}/chat/completions',b1); assert c==200,(c,r2)
    summ=stop_summary(base,sid,r2,eos_id); summ['tool_call']=call; summ['ok']=summ['finish_reason']=='stop' and summ['last_token_is_eos'] and not summ['eos_leaked_in_content'] and summ['replay']==0 and summ['repack']==0 and summ['cache_ok']
    return summ

def repeated_case(base,eos_id):
    sid=new_session(base)
    c,r1=http_json('POST',f'{base}/v1/sessions/{sid}/chat/completions',first_body()); call_a=call_from(r1); res_a=stub(call_a)
    f_a=diag(base,sid)['last_turn']['frontier_after_commit']
    c,r2=http_json('POST',f'{base}/v1/sessions/{sid}/chat/completions',body_after_a(call_a,res_a)); call_b=call_from(r2); res_b=stub(call_b)
    f_b=diag(base,sid)['last_turn']['frontier_after_commit']
    c,r3=http_json('POST',f'{base}/v1/sessions/{sid}/chat/completions',body_after_b(call_a,res_a,call_b,res_b)); assert c==200,(c,r3)
    summ=stop_summary(base,sid,r3,eos_id); summ.update({'tool_calls':[call_a,call_b],'tool_frontiers':[f_a,f_b]}); summ['ok']=summ['finish_reason']=='stop' and summ['last_token_is_eos'] and not summ['eos_leaked_in_content'] and summ['replay']==0 and summ['repack']==0 and summ['cache_ok']
    return summ

def stateless_repeated_control(base,eos_id):
    # Same complete body as repeated final turn, stateless fresh production path.
    call_a={'id':'call_a','name':'lookup_weather','arguments':'{"city":"Paris"}'}; res_a=stub(call_a)
    call_b={'id':'call_b','name':'lookup_weather','arguments':'{"city":"Berlin"}'}; res_b=stub(call_b)
    body=body_after_b(call_a,res_a,call_b,res_b)
    c,r=http_json('POST',base+'/v1/chat/completions',body); assert c==200,(c,r)
    content=r['choices'][0]['message'].get('content') or ''
    return {'status':c,'finish_reason':r['choices'][0]['finish_reason'],'eos_leaked_in_content':EOS_TEXT in content,'content_prefix':content[:300],'content_len':len(content),'ok':r['choices'][0]['finish_reason']=='stop' and EOS_TEXT not in content}

def eos_identity(args):
    import deepseek_recipe as d
    tok=d.Tokenizer.from_file(str(Path(args.recipe_path)/'static/tokenizers/v41/tokenizer.json'))
    ids=[int(x) for x in tok.encode(EOS_TEXT)]
    return {'eos_text':EOS_TEXT,'token_ids':ids,'single_token':len(ids)==1,'deepseek_recipe_revision':git_rev(args.recipe_path)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT); ap.add_argument('--omlx-path',type=Path,default=DEFAULT_OMLX); ap.add_argument('--recipe-path',type=Path,default=DEFAULT_RECIPE); ap.add_argument('--out',type=Path,default=Path('artifacts/m14/termination-qualification.json'))
    args=ap.parse_args(); ident=eos_identity(args); eos_id=ident['token_ids'][0]
    port=free_port(); base=f'http://127.0.0.1:{port}'; p=start_server(args,port)
    rec={'schema':'ds41f.m14.termination-qualification.v1','created_at':time.time(),'checkpoint':str(args.checkpoint),'omlx_path':str(args.omlx_path),'omlx_revision':git_rev(args.omlx_path),'recipe_path':str(args.recipe_path),'recipe_revision':git_rev(args.recipe_path),'eos_identity':ident,'flows':{},'source_findings':{'direct_batchgenerator_before_m14':'constructed without stop_tokens, so SequenceStateMachine was empty and EOS token id 1 was emitted as ordinary content until max_tokens','m14_bridge':'OMLXDecodeConfig.stop_token_ids -> BatchGenerator(stop_tokens=[[id]]) plus protocol suppression of the terminal EOS token while retaining it in all_tokens/cache'},'ok':False}
    try:
        wait_health(base)
        rec['flows']['plain_answer']=plain_case(base,eos_id)
        rec['flows']['single_tool_final']=single_tool_case(base,eos_id)
        rec['flows']['repeated_tool_final']=repeated_case(base,eos_id)
        rec['flows']['fresh_stateless_repeated_control']=stateless_repeated_control(base,eos_id)
    finally:
        p.terminate();
        with contextlib.suppress(Exception): p.wait(timeout=20)
        if p.poll() is None: p.kill()
    rec['ok']=ident['single_token'] and all(f.get('ok') for f in rec['flows'].values())
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(rec,indent=2))
    print(json.dumps({'ok':rec['ok'],'out':str(args.out)},indent=2)); return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
