"""Bounded phase-only observer of the unchanged admitted R1 MTP runtime."""
from __future__ import annotations
import argparse
import functools
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--r1-output', type=Path)
    p.add_argument('--port', type=int)
    p.add_argument('--joint-state', action='store_true', help='diagnostic: evaluate target cache and rings together')
    p.add_argument('--split-context', action='store_true', help='serialized diagnostic: pay target hidden debt before projection')
    a = p.parse_args()
    if a.r1_output:
        from reference.R1.mtp_model import main as run
        original = subprocess.Popen
        def launch(command, *args, **kwargs):
            if command[:4] == [sys.executable, '-m', 'ds41f_mlx.ops', 'start']:
                command = [sys.executable, '-m', 'tools.profile_mtp_startup', '--port',
                           command[command.index('--port')+1], '--output', str(a.output)]
                if a.joint_state:
                    command.append('--joint-state')
                if a.split_context:
                    command.append('--split-context')
            return original(command, *args, **kwargs)
        subprocess.Popen = launch
        try:
            return run(['--output', str(a.r1_output)])
        finally:
            subprocess.Popen = original
    import mlx.core as mx
    from ds41f_mlx.serving.internal_mtp import InternalMTPQualificationBackend
    from ds41f_mlx.prefill_fp8_mlx import DeferredPrefillAppend
    from ds41f_mlx.prefill_fp8_mlx import handoff
    from ds41f_mlx import prefill_fp8_mlx as prefill
    from ds41f_mlx.runtime.mtp_resources import MTPWiredLimitLease
    from ds41f_mlx.runtime.mtp_lifecycle import OMLXMTPGenerationSession
    from omlx.patches.deepseek_v41.mtp import DSparkMixin
    from ds41f_mlx.serve import main as serve
    from mlx_lm.generate import BatchGenerator, GenerationBatch
    from omlx.patches.deepseek_v41.language import LanguageModel
    from omlx.patches.mlx_lm_mtp import batch_generator, cache_rollback
    batch_generator.apply()
    cache_rollback.apply()
    rows, stack, undo = [], [], []
    current = []
    def patch(host, name):
        original = getattr(host, name)
        descriptor = inspect.getattr_static(host, name)
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            active = bool(stack) or name == '_start' or (name == '_next' and args[2]['generated'] == 0)
            if not active or len(rows) >= 3000:
                return original(*args, **kwargs)
            if name == '_start':
                current.append(args[1])
            if name == 'dspark_append_context' and a.split_context and len(stack) == 1:
                t = time.perf_counter()
                mx.eval(args[1])
                rows.append(dict(name='diagnostic.target_hidden_eval', parent=list(stack), start=t, wall_s=time.perf_counter()-t))
            if name == 'eval' and a.joint_state and len(stack) == 1 and current and current[-1].cache is not None:
                args = (*args, [c.state for c in current[-1].cache])
            label = host.__name__ + '.' + name
            parent = list(stack)
            stack.append(label)
            t = time.perf_counter()
            output = None
            memory = dict(active=mx.get_active_memory(), cache=mx.get_cache_memory()) if name == 'set_wired_limit' else None
            try:
                output = original(*args, **kwargs)
                return output
            finally:
                elapsed = time.perf_counter()-t
                stack.pop()
                row = dict(name=label, parent=parent, start=t, wall_s=elapsed)
                if name == '_start':
                    current.pop()
                    row.update(prefix=len(args[2].token_ids), committed=len(args[1].canonical), token_ids=list(args[2].token_ids))
                if name == 'set_wired_limit':
                    row.update(limit=args[0], previous=output, memory=memory)
                if name in ('eval', 'synchronize'):
                    frame = inspect.currentframe().f_back
                    row['caller'] = f'{frame.f_code.co_filename}:{frame.f_lineno}:{frame.f_code.co_name}'
                rows.append(row)
                if name == '_start':
                    a.output.parent.mkdir(parents=True, exist_ok=True)
                    a.output.write_text(json.dumps(dict(schema='ds41f.mtp.startup-phases.v1', rows=rows), indent=2)+'\n')
        setattr(host, name, wrapped)
        undo.append((host, name, descriptor))
    for host, names in [(InternalMTPQualificationBackend, ['_start', '_next']),
                        (DeferredPrefillAppend, ['execute_all']),
                        (DSparkMixin, ['dspark_append_context', 'mtp_install_committed_context', '__call__']),
                        (LanguageModel, ['_forward']),
                        (handoff, ['validate_committed_cache']),
                        (prefill, ['handoff_to_generation']),
                        (MTPWiredLimitLease, ['__enter__', 'transfer_to']),
                        (OMLXMTPGenerationSession, ['__post_init__', 'start']),
                        (BatchGenerator, ['__init__', 'insert', 'next']),
                        (GenerationBatch, ['__init__']),
                        (batch_generator, ['apply', '_post_init_mtp', '_dspark_next_drafts', '_call_backbone_captured']),
                        (cache_rollback, ['apply']),
                        (mx, ['eval', 'synchronize', 'set_wired_limit'])]:
        for name in names:
            patch(host, name)
    import signal
    previous_term = signal.signal(signal.SIGTERM, lambda *_: None)
    try:
        serve(['--host', '127.0.0.1', '--port', str(a.port), '--profile', 'mtp-singleton-v1'])
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        for host, name, descriptor in reversed(undo):
            setattr(host, name, descriptor)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(dict(schema='ds41f.mtp.startup-phases.v1', rows=rows), indent=2)+'\n')

if __name__ == '__main__':
    raise SystemExit(main())
