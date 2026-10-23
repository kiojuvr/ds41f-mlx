// External preflight only: installed official provider builds/enforces the profile.
// No model, tool dispatcher, replacement sandbox, or installed-package edits.
import {createRequire} from 'node:module';
import {spawnSync} from 'node:child_process';
import {writeFileSync, realpathSync} from 'node:fs';
const require = createRequire(process.env.HOME+'/.local/share/deepseek-harness/package.json');
const {Context} = await import(require.resolve('@deepseek-ai/cordis'));
const {LocalSandboxProvider} = await import(require.resolve('@deepseek-ai/dsh-sandbox-local'));
const {writableRoots} = await import(require.resolve('@deepseek-ai/dsh-sandbox'));
const ctx = new Context();
const provider = new LocalSandboxProvider(ctx, {runnerCommand:[], runnerFailureSignatures:[], probeTimeoutMs:5000});
const policy = {mode:'workspace-write',workspaceRoot:realpathSync(process.env.PHASE3_WORKSPACE)};
const command = '/tmp/ds41f-phase2-renderer/bin/python /Volumes/SDXC-512/ds41f-mlx/tools/phase3_render.py';
const results = {method:'installed LocalSandboxProvider.confine -> unmodified returned argv -> spawnSync', policy, writableRoots:writableRoots(policy), command, runs:[]};
for (const [label, cmd] of [
  ['renderer', command],
  ['repository-write-denial', `/tmp/ds41f-phase2-renderer/bin/python -c 'from pathlib import Path; Path("/Volumes/SDXC-512/ds41f-mlx/artifacts/visual-feedback-loop/phase-3-evidence/continuation-1/denial-probe.txt").write_text("must be denied")'`],
]) {
  const confined = await provider.confine(['bash','-c',cmd],policy);
  const r = spawnSync(confined.argv[0], confined.argv.slice(1), {cwd:policy.workspaceRoot,env:process.env,encoding:'utf8',timeout:25000});
  results.runs.push({label, confined, status:r.status, signal:r.signal, error:r.error?.message, stdout:r.stdout, stderr:r.stderr});
  writeFileSync(process.env.PHASE3_PREFLIGHT_RESULT,JSON.stringify(results,null,2)+'\n');
}
await ctx.dispose();
