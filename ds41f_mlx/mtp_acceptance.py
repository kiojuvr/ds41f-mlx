"""Bounded installed/source-clone local-profile acceptance over supported HTTP.

Default launches the real operator command; --runtime-url accepts an explicitly
owned already-running process. No internal experimental backend injection.
"""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from .local_client import LocalMTPClient, OwnedStream
from .mtp_profile import PROFILE, WEATHER
from .web_client import RuntimeClient, RuntimeHTTPError


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-url')
    p.add_argument('--output', type=Path, default=Path('artifacts/m41/composed.json'))
    args = p.parse_args(argv)
    from .mtp_identity import config, inspect
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def fail(stage, error):
        args.output.write_text(json.dumps(dict(schema='ds41f.m41.composed.v1',status='FAIL',
            stage=stage,error=str(error)),indent=2)+'\n')
        raise RuntimeError(str(error))
    args.output.write_text(json.dumps(dict(schema='ds41f.m41.composed.v1',status='RUNNING',stage='identity'))+'\n')
    try:
        cfg = config()
        provenance = inspect(cfg)
    except Exception as exc:
        fail('identity', exc)
    from .mtp_setup import ROOT
    prefix = args.output.resolve().with_suffix('')
    preview = Path(str(prefix)+'.native-preview.json')
    matrix = Path(str(prefix)+'.recipe-matrix.json')
    parity = Path(str(prefix)+'.native-parity.json')
    native_commands = [
        ('preview',[sys.executable,'-m','tools.run_m32_native_preview',str(cfg.recipe_path/'static/tokenizers/v41/tokenizer.json'),str(preview)]),
        ('parity',[sys.executable,'-m','tools.m32_native_parity_driver',str(ROOT/'artifacts/m31/parity-fixtures.json'),str(cfg.recipe_path/'static/tokenizers/v41/tokenizer.json')]),
        ('representation',[sys.executable,'-m','tools.run_m36r_recipe_matrix',str(cfg.recipe_path),str(matrix)])]
    native_results = []
    for name, command in native_commands:
        process = subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=120)
        Path(str(prefix)+f'.{name}.log').write_text(process.stdout+process.stderr)
        if process.returncode:
            fail('native_'+name, f'native {name} gate failed; see qualification log')
        native_results.append(dict(name=name,status='PASS'))
        if name == 'parity':
            parity.write_text(process.stdout)
            actual = json.loads(process.stdout)
            reference = json.loads((ROOT/'artifacts/m32/canonical-base.json').read_text())
            if actual['records'] != reference['records']:
                fail('native_parity', 'rebuilt native recipe differs from official base corpus')
    actual_matrix = json.loads(matrix.read_text())['rows']
    old_matrix = json.loads((ROOT/'artifacts/m39/recipe-matrix.json').read_text())['rows']
    predicates = ('representable','exact_prefix','semantic_complete','executable_tools','first_mismatch')
    if [(r['name'],r['canonical'],{k:r['certificate'].get(k) for k in predicates}) for r in actual_matrix] != [
            (r['name'],r['canonical'],{k:r['certificate'].get(k) for k in predicates}) for r in old_matrix]:
        fail('native_representation', 'native representation fixture contract changed')
    result = dict(schema='ds41f.m41.composed.v1',status='RUNNING',native_gates=native_results,
                  identity_sha256=provenance['identity_sha256'],cases=[],admission=[])
    process = None
    log = None
    clients = []
    def save():
        args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    if args.runtime_url:
        url = args.runtime_url
    else:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); port = sock.getsockname()[1]
        url = f'http://127.0.0.1:{port}'
        log = args.output.with_suffix('.server.log').open('w')
        process = subprocess.Popen([sys.executable,'-m','ds41f_mlx.ops','start',
            '--profile',PROFILE,'--port',str(port)], stdout=log,stderr=subprocess.STDOUT)
    rt = RuntimeClient(url)
    base = dict(model='deepseek-v4.1-flash',temperature=0,max_tokens=96,
                reasoning_effort='none',stream=True)
    tool_options = dict(base,tools=[WEATHER],tool_choice={'type':'function','function':{'name':'lookup_weather'}})
    def client():
        c = LocalMTPClient(url); clients.append(c); c.create(); return c
    def resolve(c):
        start = time.monotonic()
        while time.monotonic()-start < 1800:
            value = c.reconcile()
            if isinstance(value,OwnedStream):
                list(value)
            elif value.get('outcome_state') != 'active':
                return value
            time.sleep(.01)
        raise AssertionError('settlement deadline')
    def record(name,c,out,started):
        for _ in range(2):
            assert resolve(c) == out
        row = dict(name=name,wall_s=time.monotonic()-started,identity=c.identity.to_json(),outcome=out,
                   client_state=c.state)
        result['cases'].append(row); save()
        if out['outcome_state']=='recoverable':
            m = out['metrics']
            assert m['aligned_idle'] and m['prompt_replay']==m['full_cache_repack']==0
            assert m['settlement']['new_verify_cycles']==m['settlement']['new_proposals']==0
        return row
    def turn(c,text,options=base,name='text',cut=None):
        start = time.monotonic()
        value = c.submit([dict(role='user',content=text)],options=options)
        if isinstance(value,OwnedStream):
            if cut:
                cut(value)
            else:
                list(value)
        out = resolve(c)
        record(name,c,out,start)
        return out
    def reject(method,path,body,headers=None,expected=400):
        from urllib.parse import urlsplit
        u = urlsplit(url); conn=http.client.HTTPConnection(u.hostname,u.port,timeout=40)
        try:
            conn.request(method,path,body,headers or {'Content-Type':'application/json'})
            r=conn.getresponse();payload=r.read()
            assert r.status==expected,(r.status,payload)
            result['admission'].append(dict(method=method,path=path,status=r.status,body_bytes=len(body or b'')))
        finally:conn.close()
    try:
        for _ in range(3000):
            try:
                h=rt.health();break
            except Exception:
                if process and process.poll() is not None:raise AssertionError('operator startup failed')
                time.sleep(.05)
        else:raise AssertionError('startup health deadline')
        assert h['profile']==PROFILE
        if not args.runtime_url:
            assert h['model_ready'] is False
        result['initial_health']=h
        c=client();sid=c.session_id;before=rt.get_session(sid)
        path=f'/v1/sessions/{sid}/chat/completions'
        body=json.dumps(dict(base,messages=[dict(role='user',content='Hello')])).encode()
        for raw in (b'[]',b'{"model":"x","model":"x"}',b'{"x":NaN}',b'{',
                    json.dumps(dict(base,messages=[dict(role='user',content='Hello')],stop=None)).encode(),
                    json.dumps(dict(base,messages=[dict(role='user',content='Hello')],max_tokens=True)).encode(),
                    json.dumps(dict(base,messages=[dict(role='user',content='Hello')],max_tokens=769)).encode(),
                    json.dumps(dict(base,messages=[dict(role='user',content='Hello')],stream_options={'include_usage':True})).encode()):
            reject('POST',path,raw,{'Content-Type':'application/json','X-DS41F-Request-Sequence':'1'})
            assert rt.get_session(sid)==before
        for seq in ('0','01','-1','+1','18446744073709551616'):
            reject('POST',path,body,{'Content-Type':'application/json','X-DS41F-Request-Sequence':seq})
            assert rt.get_session(sid)==before
        reject('POST',path,body) # mandatory fence
        for route in ('/v1/chat/completions','/v1/responses','/v1/messages','/v1/sessions/restore',f'/v1/sessions/{sid}/persist'):
            reject('POST',route,b'not json')
        reject('POST','/v1/sessions',b'{"id":null}')
        reject('POST','/v1/sessions',b'{}',expected=409)
        reject('POST',path,b'x'*1048577,{'Content-Type':'application/json','X-DS41F-Request-Sequence':'1'},expected=413)
        reject('POST',path,body,{'Content-Type':'application/json','X-DS41F-Request-Sequence':'1','Origin':'http://evil.invalid'})
        reject('POST',path,body,{'Content-Type':'application/json','X-DS41F-Request-Sequence':'1','Host':'evil.invalid'})
        assert rt.get_session(sid)==before
        # Official conversion runs but context denial precedes any model authority.
        huge=json.dumps(dict(base,messages=[dict(role='user',content=' word'*9000)])).encode()
        reject('POST',path,huge,{'Content-Type':'application/json','X-DS41F-Request-Sequence':'1'})
        assert rt.get_session(sid)==before
        out=turn(c,'Say hello briefly. Do not use tools.',name='normal_text')
        assert out['outcome_state']=='recoverable'
        out=turn(c,'Say goodbye briefly. Do not use tools.',options=dict(base,stream=False),name='retained_json')
        # Previous settled GET is not an ACK that this next POST cannot arrive.
        # A retained body-loss retry must keep exactly the frozen next identity.
        original=c.runtime.internal_fenced_request
        def lose_body(sid, body, seq):
            with socket.create_connection(('127.0.0.1',port)) as peer:
                headers=(f'POST /v1/sessions/{sid}/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n'
                         f'Content-Type: application/json\r\nX-DS41F-Request-Sequence: {seq}\r\nContent-Length: {len(body)}\r\n\r\n').encode()
                peer.sendall(headers+body[:len(body)//2])
            raise OSError('injected real TCP body loss before admission')
        before=c.runtime.get_session(c.session_id)
        c.runtime.internal_fenced_request=lose_body
        try:
            c.submit([dict(role='user',content='Say hello again briefly. No tools.')],options=base)
            raise AssertionError('body-loss injection did not happen')
        except OSError:
            assert c.state=='ambiguous'
        finally:c.runtime.internal_fenced_request=original
        frozen=c.identity
        time.sleep(.1)
        after=c.runtime.get_session(c.session_id)
        assert after==before
        retry=c.reconcile();assert c.identity is frozen
        if isinstance(retry,OwnedStream):list(retry)
        out=resolve(c);result['cases'].append(dict(name='retained_loss_before_admission',identity=frozen.to_json(),outcome=out))
        assert out['outcome_state']=='recoverable'
        old=c.session_id;c.retire()
        reject('GET',f'/v1/sessions/{old}',None,expected=404)
        # Raw UTF-8 loss inside an SSE frame; no local byte reconstruction.
        c=client()
        def byte_cut(stream):
            while True:
                byte=stream.response.read(1)
                if not byte:raise AssertionError('no multibyte output sampled')
                if byte[0]>=0xc0:
                    stream.close();return
        out=turn(c,'Repeat exactly: café 🙂 café 🙂 café 🙂',name='utf8_byte_loss',cut=byte_cut)
        assert out['outcome_state']=='recoverable'
        out=turn(c,'Say hello briefly.',name='byte_loss_continuation')
        assert out['outcome_state']=='recoverable';c.retire()
        effects=[]
        for multi in (False,True):
            c=client()
            task='Use the tool for Paris and Berlin, then summarize both results concisely.' if multi else 'Use the tool to look up the weather in Paris, then answer concisely.'
            out=turn(c,task,tool_options,name='two_calls' if multi else 'one_call')
            assert out['outcome_state']=='recoverable'
            calls=out['response']['choices'][0]['message']['tool_calls']
            assert len(calls)==(2 if multi else 1)
            def execute(call):
                effects.append(call)
                return json.dumps(dict(city=json.loads(call['function']['arguments'])['city'],weather='sunny 21C'))
            start=time.monotonic();a=c.execute_tools(execute);b=c.execute_tools(execute);assert a==b
            assert len(effects)==(3 if multi else 1)
            value=c.submit_tool_results(options=dict(tool_options,tool_choice='auto'))
            list(value);out=resolve(c);record('stored_result_continuation',c,out,start)
            assert out['outcome_state']=='recoverable';c.retire()
        # Completed call with zero consumed response bytes. Canonical settlement
        # is observed separately; the client still reconciles its frozen identity.
        c=client()
        def settled_loss(stream):
            deadline=time.monotonic()+120
            while time.monotonic()<deadline:
                if rt.get_session(c.session_id)['outcome_state']!='active':
                    stream.close();return
                time.sleep(.01)
            raise AssertionError('terminal delivery window timeout')
        out=turn(c,'Use the tool to look up the weather in Berlin, then answer concisely.',tool_options,
                 name='completed_tool_delivery_loss',cut=settled_loss)
        assert out['outcome_state']=='recoverable' and out['certificate']['executable_tools']
        c.execute_tools(execute);c.execute_tools(execute)
        value=c.submit_tool_results(options=dict(tool_options,tool_choice='auto'));list(value)
        out=resolve(c);record('lost_call_stored_result',c,out,time.monotonic());c.retire()
        # Cut real canonical argument source, not merely a transport byte prefix.
        c=client()
        def argument_cut(stream):
            for event in stream:
                choices=event.get('choices') or []
                for call in choices[0].get('delta',{}).get('tool_calls',[]) if choices else []:
                    if call.get('function',{}).get('arguments'):
                        stream.close();return
            raise AssertionError('no argument delta')
        out=turn(c,'Use the tool to look up the weather in Berlin, then answer concisely.',tool_options,
                 name='partial_tool_negative',cut=argument_cut)
        assert out['outcome_state']=='unrecoverable' and not out['certificate']['representable']
        count=len(effects)
        try:c.execute_tools(execute)
        except Exception:pass
        else:raise AssertionError('negative call executed')
        assert len(effects)==count
        c.retire();c.create() # explicit fresh application prefill, no native restore
        out=turn(c,'Reply with one short greeting. Do not use tools.',name='explicit_fresh_after_negative')
        assert out['outcome_state']=='recoverable';c.retire()
        # All fixed aliases and the output admission edges are exercised on the
        # composed model. An admitted length-1 turn may legitimately be negative.
        for alias,n in (('deepseek-v4.1-flash',1),('deepseek-v41-flash',768),('deepseek-flash',96)):
            c=client()
            out=turn(c,'Say hello briefly.',options=dict(base,model=alias,max_tokens=n,stream=False),
                     name=f'alias_budget_{alias}_{n}')
            assert out['outcome_state'] in ('recoverable','unrecoverable')
            assert out['metrics']['generated'] <= n
            c.retire()
        c=client()
        start=time.monotonic()
        stream=c.submit([dict(role='user',content='Write a detailed eight-section guide to writing reliable Python software.')],
                        options=dict(base,max_tokens=768))
        identity=c.identity
        # Header publication is not lease release or generation acknowledgement.
        for operation in (lambda:rt.internal_fenced_request(identity.session_id,identity.body,identity.sequence),
                          lambda:rt.internal_fenced_request(identity.session_id,identity.body,identity.sequence+1),
                          lambda:rt.close_session(identity.session_id),lambda:rt.create_session()):
            try:operation()
            except RuntimeHTTPError as exc:assert exc.status==409
            else:raise AssertionError('active overlap admitted')
        observed=0
        for event in stream:
            choices=event.get('choices') or []
            if choices and choices[0].get('delta',{}).get('content'):
                observed+=1
                if observed==4:stream.close();break
        out=resolve(c);record('active_overlap_and_stream_cancel',c,out,start)
        assert out['outcome_state'] in ('recoverable','unrecoverable') and out['metrics']['generated'] < 768
        try:rt.internal_fenced_request(identity.session_id,identity.body+b' ',identity.sequence)
        except RuntimeHTTPError as exc:assert exc.status==409
        else:raise AssertionError('changed exact bytes accepted')
        if out['outcome_state']=='recoverable':
            next_out=turn(c,'Say goodbye briefly.',name='cancel_retained_continuation')
            assert next_out['outcome_state']=='recoverable'
            try:rt.internal_fenced_request(identity.session_id,identity.body,identity.sequence)
            except RuntimeHTTPError as exc:assert exc.status==409
            else:raise AssertionError('expired outcome regenerated')
        c.retire()
        # Churn beyond the diagnostic budget; old IDs never regain authority.
        for _ in range(20):
            c=client();c.retire()
        reject('GET',f'/v1/sessions/{old}',None,expected=404)
        result.update(status='PASS',effects=len(effects),final_health=rt.health(),
                      inherited='M33-M39 native invariants; fresh composed HTTP/projection/helper gates above')
    except BaseException as exc:
        result.update(status='FAIL',error=repr(exc));raise
    finally:
        save()
        if process:
            process.terminate()
            try:process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.kill();process.wait();result['forced_shutdown']=True;save()
        if log:log.close()
    print(json.dumps(dict(status=result['status'],cases=len(result['cases']),admission=len(result['admission']))))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
