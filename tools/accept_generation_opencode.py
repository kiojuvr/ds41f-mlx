"""Real OpenCode coding/tool/cancel/recovery acceptance, never a client mock.

Requires the admitted production runtime with observe_generation_runtime.py and
byte-preserving observe_client_wire.py. The caller provisions a scratch coding
project and an OpenCode provider config; this collector changes neither client
system prompt nor tool behavior. All continuation requests come from OpenCode.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / 'artifacts/generation-policy/frontend-execution.jsonl'
WIRE = ROOT / 'artifacts/generation-policy/opencode-wire.jsonl'


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def status():
    with urlopen('http://127.0.0.1:18441/ds41f/status', timeout=3) as response:
        return json.load(response)


def wait_idle():
    for _ in range(600):
        if not status()['active_requests'] and not status()['queued_requests']:
            return
        time.sleep(.1)
    raise TimeoutError('Runtime has not settled; no resend')


def start(args, name, prompt, session=None):
    command = ['opencode', 'run', '--dir', str(args.directory), '--pure', '--print-logs', '--log-level', 'INFO', '--format', 'json', '--agent', 'build', '--model', 'ds41f/deepseek-v4.1-flash']
    if session: command += ['--session', session]
    command += [prompt]
    return subprocess.Popen(command, cwd=args.directory, env=os.environ | {'PWD': str(args.directory)}, stdin=subprocess.DEVNULL,
        stdout=(args.output / (name + '.jsonl')).open('w'), stderr=(args.output / (name + '.stderr')).open('w'),
        start_new_session=True)


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    trace_start, wire_start = len(rows(TRACES)), len(rows(WIRE))
    with urlopen('http://127.0.0.1:8000/health') as response: health = json.load(response)
    assert health['status'] == 'ready'
    prompt = ('Implement sum_column(path, column) in csv_total.py using only the Python standard library. '
        'Read UTF-8 CSV with csv.DictReader, support quoted commas and Unicode headers, and return a decimal.Decimal total. '
        'Ignore blank cells. Missing column and invalid/nonfinite decimal cells must raise ValueError; invalid cells must identify '
        'the 1-based data row. A header-only CSV returns Decimal zero. Add unittest coverage for these cases, run it, '
        'and briefly summarize the result. Do not change opencode.json or install dependencies.')
    child = start(args, 'coding', prompt)
    try: code = child.wait(timeout=600)
    except subprocess.TimeoutExpired:
        child.send_signal(signal.SIGINT); child.wait(timeout=90); wait_idle()
        raise TimeoutError('Coding workload bounded stop; inspect practical completion, never infer a policy failure')
    wait_idle(); assert code == 0, (code, 'coding.stderr')
    events = rows(args.output / 'coding.jsonl')
    session = next(event['sessionID'] for event in events if event.get('sessionID'))
    tools = [event for event in events if event.get('type') == 'tool_use']
    assert len(tools) >= 2, 'No real agent/tool continuation observed'
    assert any(event.get('type') == 'text' for event in events), 'No practical final answer'
    independent = subprocess.run(['python3', '-c', '''
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from csv_total import sum_column
with TemporaryDirectory() as d:
 p=Path(d)/'data.csv'
 p.write_text('名前,amount\\n"A,B",1.20\\nC,\\nD,2.30\\n',encoding='utf-8')
 assert sum_column(p,'amount') == Decimal('3.50')
 p.write_text('amount\\n',encoding='utf-8'); assert sum_column(p,'amount') == Decimal(0)
 for text in ('amount\\nbad\\n','amount\\nNaN\\n','amount\\nInfinity\\n'):
  p.write_text(text,encoding='utf-8')
  try: sum_column(p,'amount')
  except ValueError as e: assert '1' in str(e), str(e)
  else: raise AssertionError('invalid cell accepted')
 p.write_text('other\\n1\\n',encoding='utf-8')
 try: sum_column(p,'amount')
 except ValueError: pass
 else: raise AssertionError('missing column accepted')
print('INDEPENDENT_CODING_PASS')
'''], cwd=args.directory, capture_output=True, text=True)
    (args.output / 'independent.txt').write_text(independent.stdout + independent.stderr)
    assert independent.returncode == 0, independent.stderr
    cancel_base = len(rows(TRACES))
    child = start(args, 'cancel', 'CANCEL-PROBE: Do not call tools or modify files. Explain in detail how to design a robust CSV Decimal summation API, covering Unicode headers, quoting, missing columns, blanks, nonfinite values, row errors and tests. Give a substantial multi-section tutorial with examples.', session)
    for _ in range(1200):
        current = status().get('current_request') or {}
        if current.get('current_phase') == 'DECODING' and current.get('metrics', {}).get('generated', 0) >= 10:
            (args.output / 'active-before-stop.json').write_text(json.dumps(current, indent=2))
            break
        if child.poll() is not None: raise RuntimeError('OpenCode ended before streaming cancellation')
        time.sleep(.1)
    else: raise TimeoutError('No real active decode for cancellation')
    child.send_signal(signal.SIGINT)
    try: child.wait(timeout=90)
    except subprocess.TimeoutExpired:
        child.terminate(); child.wait(timeout=15)
    wait_idle()
    cancel = rows(TRACES)[cancel_base:]
    assert any(record['cancelled'] for record in cancel), 'No actual runtime cancellation'
    settled_count = len(rows(TRACES)); request_count = len(rows(WIRE))
    time.sleep(2)
    assert len(rows(TRACES)) == settled_count and len(rows(WIRE)) == request_count, 'Automatic resend after Stop'
    child = start(args, 'recovery', 'RECOVERY-AFTER-STOP: The tutorial output was intentionally stopped. Do not resume it. Run the existing unittest suite once and briefly summarize whether the CSV implementation is ready.', session)
    code = child.wait(timeout=300); wait_idle(); assert code == 0
    recovery = rows(args.output / 'recovery.jsonl')
    assert any(event.get('type') == 'text' for event in recovery)
    events += recovery
    calls = [event['part']['callID'] for event in events if event.get('type') == 'tool_use']
    assert len(calls) == len(set(calls)), 'Duplicate executed tool effect identity'
    execution, wire = rows(TRACES)[trace_start:], rows(WIRE)[wire_start:]
    statuses = rows(WIRE.with_suffix('.statuses.jsonl'))[wire_start:]
    assert len(execution) == len(wire) == len(statuses)
    assert all(record['status'] == 200 for record in statuses), statuses
    hashes = [hashlib.sha256(record['body'].encode()).hexdigest() for record in wire]
    assert len(hashes) == len(set(hashes)), 'Repeated exact request body'
    keys = tuple(health['generation_policy']['defaults'])
    def signature(policy): return tuple(None if policy[key] is None else round(policy[key], 6) for key in keys)
    admitted = []
    for record in wire:
        body = json.loads(record['body'])
        admitted.append(health['generation_policy']['defaults'] | {key: body[key] for key in keys if body.get(key) is not None})
    assert Counter(map(signature, admitted)) == Counter(signature(record['generation_policy']) for record in execution)
    for record in execution:
        assert record['prompt_replay'] == record['full_cache_repack'] == 0
        assert set(record['target_offsets'] + record['dspark_offsets']) == {record['canonical_frontier']}
        assert record['execution_sampling'] == ('greedy' if record['generation_policy']['temperature'] == 0 else 'stochastic')
    receipt = dict(status='REAL_OPENCODE_MODEL_PASS', version=subprocess.check_output(['opencode', '--version'], text=True).strip(),
        session=session, workload='stdlib CSV Decimal implementation + tests', practical_completion=True,
        independent_check=independent.stdout.strip(), request_count=len(wire), actual_executions=execution,
        client_policies=admitted, tool_effects=len(calls), duplicate_tool_effects=0, duplicate_request_bodies=0,
        cancellation=True, recovery=True, health=health)
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    print('REAL_OPENCODE_MODEL_PASS', session, len(wire), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.directory = args.directory.resolve(); args.output = args.output.resolve()
    main(args)
