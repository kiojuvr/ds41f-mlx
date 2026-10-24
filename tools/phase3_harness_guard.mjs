// Narrow Phase 3 deployment policy. Registers no tools and dispatches none.
// Official Harness owns scheduling, execution, attachments, and session results.
import {realpathSync, appendFileSync, readFileSync, existsSync} from 'node:fs';
import {resolve} from 'node:path';
export const name = 'phase3-bounded-policy';
export const inject = ['tools'];
export function apply(ctx) {
  const workspace = realpathSync(process.env.PHASE3_WORKSPACE);
  const policy = JSON.parse(readFileSync(process.env.PHASE3_POLICY, 'utf8'));
  let effects = 0, imageReads = 0;
  const maxRenders = policy.max_renders ?? 6;
  const sourcePath = workspace + '/index.html';
  const allowedPath = p => typeof p === 'string' && resolve(workspace, p) === sourcePath && (!existsSync(sourcePath) || realpathSync(sourcePath) === sourcePath);
  // A small shell grammar, not arbitrary shell: no substitutions, pipes, redirects,
  // alternate paths, network, mutation, or background jobs. Output remains capped
  // by the official bash sandbox. sed ranges cover at most 40 source lines.
  const inspection = command => {
    if (!allowedPath('index.html')) return false;
    if (['pwd', 'ls index.html', 'wc -lc index.html'].includes(command)) return true;
    const m = /^sed -n '([1-9][0-9]{0,3}),([1-9][0-9]{0,3})p' index\.html$/.exec(command);
    return !!m && Number(m[2]) >= Number(m[1]) && Number(m[2]) - Number(m[1]) < 40;
  };
  ctx.tools.guard(exec => {
    const a = exec.arguments ?? {};
    let reason;
    if (effects >= policy.tools || Date.now() > policy.deadline_ms) reason = 'Phase 3 execution budget exhausted';
    else if (a.sandbox_permissions !== undefined || a.justification !== undefined) reason = 'Phase 3 forbids escalation';
    else if (['write', 'edit', 'read'].includes(exec.name)) {
      if (!allowedPath(a.file_path)) reason = 'Phase 3 permits only workspace/index.html';
      if (exec.name === 'write' && (typeof a.content !== 'string' || Buffer.byteLength(a.content) > 65536)) reason = 'Phase 3 source-size limit';
      if (!reason && exec.name === 'edit') {
        const source = readFileSync(workspace + '/index.html', 'utf8');
        if (typeof a.old_string !== 'string' || typeof a.new_string !== 'string' || Buffer.byteLength(source.replace(a.old_string, a.new_string)) > 65536) reason = 'Phase 3 source-size limit';
      }
    } else if (exec.name === 'bash') {
      let workdirOK = a.workdir === undefined;
      try { if (a.workdir !== undefined) workdirOK = realpathSync(a.workdir) === workspace; } catch {}
      if ((a.command !== policy.command && !inspection(a.command)) || !workdirOK || a.run_in_background === true || (a.timeoutMs !== undefined && (!Number.isFinite(a.timeoutMs) || a.timeoutMs <= 0 || a.timeoutMs > 25000))) reason = 'Phase 3 permits only the approved foreground renderer or bounded read-only source inspection';
    } else if (exec.name === 'read_image') {
      const p = typeof a.file_path === 'string' ? resolve(workspace, a.file_path) : '';
      const index = /^render-(\d{2})\.png$/.exec(p.slice(workspace.length + 1));
      if (imageReads >= maxRenders || !p.startsWith(workspace + '/') || !index || Number(index[1]) >= maxRenders || !existsSync(p) || realpathSync(p) !== p) reason = 'Phase 3 permits only budgeted produced workspace screenshots';
    } else reason = 'Tool not admitted for Phase 3';
    appendFileSync(policy.audit, JSON.stringify({at:new Date().toISOString(), call_id:exec.callId, name:exec.name, decision:reason?'deny':'allow', reason})+'\n');
    if (!reason) { effects++; if (exec.name === 'read_image') imageReads++; }
    return reason;
  });
}
