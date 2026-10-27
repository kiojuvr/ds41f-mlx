"""Model-free installed headless contract audit. Never submits a task or command."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
INSTALL = Path.home() / '.local/share/deepseek-harness/node_modules/@deepseek-ai'
AUDIT = Path('/tmp/ds41f-w1-harness-source')


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


def main():
    assert subprocess.check_output(['git', '-C', str(AUDIT), 'rev-parse', 'HEAD'], text=True).strip() == '687f282e8625c6b8d11ff58211a90524be2e4317'
    names = ['dsh', 'dsh-base', 'dsh-headless', 'dsh-commands', 'dsh-command-compact',
             'dsh-compaction', 'dsh-compaction-basic', 'dsh-session', 'dsh-token-meter',
             'dsh-llm-deepseek', 'dsh-llm-deepseek-api-key', 'dsh-tool-fs',
             'dsh-attachment-local']
    packages = {}
    for name in names:
        package = INSTALL / name
        manifest = json.loads((package / 'package.json').read_text())
        assert manifest['version'] == '0.2.1-rc.2'
        packages[name] = dict(version=manifest['version'], files={
            str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(package.rglob('*')) if p.is_file() and (
                p.suffix in {'.js', '.yml'} or p.name == 'package.json')})
    save('installed-identities.json', dict(packages=packages,
        node=subprocess.check_output(['node', '--version'], text=True).strip(),
        source_audit='687f282e8625c6b8d11ff58211a90524be2e4317',
        installed_build_git_sha='not established by package manifests'))
    temp = Path(tempfile.mkdtemp(prefix='ds41f-step3b-contract-'))
    try:
        workspace = temp / 'workspace'
        workspace.mkdir()
        profile = temp / 'home/profiles/headless'
        profile.mkdir(parents=True)
        (profile / 'package.json').write_text(json.dumps(dict(name='ds41f-step3b-contract',
            private=True, dependencies={}, dsh=dict(profile=dict(
                bundles=['@deepseek-ai/dsh-base', '@deepseek-ai/dsh-headless'])))))
        (profile / 'cordis.yml').write_text('[]\n')
        prior = (OUT.parent / 'phase3-step3a-evidence/profile.patch.yml').read_text()
        patch = re.sub(r'    workspaceRoot: .*', '    workspaceRoot: ' + str(workspace), prior)
        patch = patch.split('- insert:\n')[0]  # no diagnostic plugins, dispatchers or guards
        (profile / 'cordis.patch.yml').write_text(patch)
        (OUT / 'profile.patch.yml').write_text(patch)
        env = dict(os.environ, DSH_HOME=str(temp / 'home'), DSH_AGENTS_HOME=str(temp / 'agents'),
                   DS41F_LOCAL_API_KEY='local-only-placeholder')
        for flag, name in [('--dump-config', 'composed-config.yml'), ('--help', 'headless-help.txt')]:
            result = subprocess.run(['node', str(INSTALL / 'dsh/lib/bin.js'), '--profile',
                'headless', flag], env=env, cwd=workspace, capture_output=True, text=True, timeout=60)
            (OUT / name).write_text(result.stdout)
            (OUT / (name + '.stderr')).write_text(result.stderr)
            assert result.returncode == 0, result.stderr
        runner = (INSTALL / 'dsh-headless/lib/index.js').read_text()
        assert 'agent.followup(createUserMessage({' in runner
        assert 'text: task' in runner
        assert 'commands' not in runner
        command = (INSTALL / 'dsh-command-compact/lib/index.js').read_text()
        assert 'ctx.commands.register({' in command and 'ctx.compaction.compactNow(' in command
        save('contract-result.json', dict(
            disposition='BLOCKED', weights_loaded=0, commands_invoked=0, inference_requests=0,
            reason='Installed headless runner has no supported human-command dispatch interface; task text is always a model-facing user followup.',
            config_dump_succeeded=True, help_succeeded=True,
            human_command_registry_and_plugin_composed=True,
            registration_implementation_verified_by_source=True,
            human_command_adapter_in_headless=False,
            no_alternative_dispatch_or_event_injection_attempted=True))
    finally:
        shutil.rmtree(temp)
        save('preflight-cleanup.json', dict(temporary_root=str(temp), removed=not temp.exists(),
            task_owned_model_processes_started=0, task_owned_listeners_started=0))


if __name__ == '__main__':
    main()
