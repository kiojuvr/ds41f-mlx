#!/usr/bin/env python3
"""M24 bounded operational agent soak harness.

This is a controlled client-side agent harness for the OpenAI-compatible ds41f
stateful Chat Completions endpoint.  It keeps tool execution on the client side,
records exact request/response/tool-result envelopes for selected turns, and
samples bounded runtime diagnostics.
"""
from __future__ import annotations

import argparse, hashlib, json, os, shutil, signal, subprocess, sys, tempfile, textwrap, time, urllib.error, urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def http(method: str, url: str, body: dict[str, Any] | None = None, timeout: float = 600.0) -> Any:
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    if not raw:
        return None
    return json.loads(raw.decode())


def try_http(method: str, url: str, body: dict[str, Any] | None = None, timeout: float = 60.0) -> tuple[bool, Any]:
    try:
        return True, http(method, url, body, timeout)
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read().decode())
        except Exception:
            payload = e.reason
        return False, {"status": e.code, "error": payload}
    except Exception as e:
        return False, {"error_type": type(e).__name__, "error": str(e)}


def rss_bytes(pid: int | None) -> int | None:
    if not pid:
        return None
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()
        return int(out) * 1024 if out else None
    except Exception:
        return None


def vm_stat() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        text = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()
        out["swapusage"] = text
    except Exception:
        pass
    try:
        pressure = subprocess.check_output(["memory_pressure", "-Q"], text=True, stderr=subprocess.STDOUT, timeout=5)
        out["memory_pressure_tail"] = pressure[-1000:]
    except Exception:
        pass
    return out


def sha_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_workspace(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "calc.py").write_text("""def add(a, b):\n    return a + b\n\ndef mul(a, b):\n    return a + b  # intentional bug for the agent to find\n""")
    (path / "test_calc.py").write_text("""from calc import add, mul\n\ndef test_add():\n    assert add(2, 3) == 5\n\ndef test_mul():\n    assert mul(3, 4) == 12\n""")
    (path / "README.md").write_text("Disposable M24 workspace. Fix calc.mul and keep tests passing.\n")


def run_tool(workspace: Path, name: str, args: dict[str, Any]) -> dict[str, Any]:
    try:
        root = workspace.resolve()
        if name == "list_files":
            return {"ok": True, "files": sorted(str(p.resolve().relative_to(root)) for p in workspace.rglob("*") if p.is_file())}
        if name == "read_file":
            p = (workspace / str(args.get("path", ""))).resolve()
            if not str(p).startswith(str(root)):
                raise ValueError("path escapes workspace")
            return {"ok": True, "path": str(p.relative_to(root)), "content": p.read_text()[:12000]}
        if name == "write_file":
            p = (workspace / str(args.get("path", ""))).resolve()
            if not str(p).startswith(str(root)):
                raise ValueError("path escapes workspace")
            p.write_text(str(args.get("content", "")))
            return {"ok": True, "path": str(p.relative_to(root)), "bytes": p.stat().st_size}
        if name == "run_tests":
            cmd = [sys.executable, "-m", "pytest", "-q"]
            proc = subprocess.run(cmd, cwd=workspace, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            return {"ok": proc.returncode == 0, "returncode": proc.returncode, "output": proc.stdout[-12000:]}
        if name == "intentional_missing_command":
            proc = subprocess.run(["./definitely_missing_m24_command"], cwd=workspace, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
            return {"ok": proc.returncode == 0, "returncode": proc.returncode, "output": proc.stdout[-2000:]}
        raise ValueError(f"unknown tool {name}")
    except Exception as e:
        return {"ok": False, "error_type": type(e).__name__, "error": str(e)}


def tools_schema() -> list[dict[str, Any]]:
    def fn(name: str, desc: str, props: dict[str, Any], required: list[str] = []) -> dict[str, Any]:
        return {"type": "function", "function": {"name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": required}}}
    return [
        fn("list_files", "List files in the disposable workspace.", {}),
        fn("read_file", "Read a workspace file.", {"path": {"type": "string"}}, ["path"]),
        fn("write_file", "Overwrite a workspace file.", {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
        fn("run_tests", "Run the workspace pytest suite.", {}),
        fn("intentional_missing_command", "Run a known-missing command once to prove failure recovery.", {}),
    ]


def assistant_message(resp: dict[str, Any]) -> dict[str, Any]:
    msg = dict(resp["choices"][0]["message"])
    msg.setdefault("role", "assistant")
    return msg


def call_ids(msg: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    out = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except Exception:
            args = {"_raw_arguments": fn.get("arguments")}
        out.append((tc.get("id") or f"call_{len(out)}", fn.get("name") or "", args))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("DS41F_BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--server-command", nargs="*", help="optional command to start the release server (use -- before command options, or prefer --server-command-string)")
    ap.add_argument("--server-command-string", help="optional shell command to start the release server")
    ap.add_argument("--output", type=Path, default=ROOT/"artifacts/m24/operational-soak.json")
    ap.add_argument("--workspace", type=Path)
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--persist-after", type=int, default=5)
    ap.add_argument("--cancel-turn", type=int, default=3)
    ap.add_argument("--diagnostic-endpoints", action="store_true")
    args = ap.parse_args(argv)

    started = time.time(); server = None; pid = None
    env = os.environ.copy(); env.setdefault("DS41F_ENABLE_DIAGNOSTIC_ENDPOINTS", "1"); env.setdefault("DS41F_TRACE_HISTORY_LIMIT", "128")
    server_cmd = (["/bin/bash", "-lc", args.server_command_string] if args.server_command_string else args.server_command)
    if server_cmd:
        server = subprocess.Popen(server_cmd, cwd=Path.cwd(), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        pid = server.pid
        for _ in range(240):
            ok, h = try_http("GET", args.base_url + "/health", timeout=2)
            if ok and h.get("model_ready"):
                break
            time.sleep(1)
    workspace = args.workspace or Path(tempfile.mkdtemp(prefix="ds41f-m24-work-"))
    make_workspace(workspace)
    session = http("POST", args.base_url + "/v1/sessions", {"id": "m24_soak"})
    sid = session["id"]
    messages: list[dict[str, Any]] = [{"role": "system", "content": textwrap.dedent("""
        You are driving a disposable coding workspace through tools. Work in small steps.
        First inspect files. At least once call intentional_missing_command, observe failure, and recover.
        Fix the bug, run tests, and then perform several verification/summary turns. Do not invent tool results.
    """).strip()}, {"role": "user", "content": "Start the M24 agent workload. Inspect the workspace and fix the tests."}]
    exact_samples: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    persisted = None; restored = False; cancellation = None
    tool_cycles = 0

    for i in range(1, args.turns + 1):
        body = {"model": "deepseek-v4.1-flash", "messages": messages, "tools": tools_schema(), "tool_choice": "auto", "temperature": 0, "max_tokens": args.max_tokens, "stream": False}
        recovered_after_cancel = False
        if i == args.cancel_turn:
            # Client-side interruption: short timeout against a valid turn.  If the server commits after the
            # client gives up, do not retry as new work; recover the canonical committed response from session
            # metadata and continue the transcript from that exact protocol response.
            ok, res = try_http("POST", f"{args.base_url}/v1/sessions/{sid}/chat/completions", body, timeout=0.001)
            cancellation = {"turn": i, "client_result": res, "classified_as": "client_interruption_waiting_for_idle_boundary"}
            recovered_resp = None
            for _ in range(1200):
                gok, srec = try_http("GET", f"{args.base_url}/v1/sessions/{sid}", timeout=10)
                if gok and srec.get("state") != "busy":
                    if int(srec.get("request_count", 0)) >= i and ((srec.get("last_turn") or {}).get("response_json")):
                        recovered_resp = (srec.get("last_turn") or {}).get("response_json")
                        recovered_after_cancel = True
                        cancellation["server_committed_after_client_timeout"] = True
                    break
                time.sleep(1)
            if recovered_resp is None:
                cancellation["server_committed_after_client_timeout"] = False
        t0 = time.time()
        if recovered_after_cancel:
            ok, resp = True, recovered_resp
            elapsed = 0.0
        else:
            ok, resp = try_http("POST", f"{args.base_url}/v1/sessions/{sid}/chat/completions", body, timeout=1200); elapsed = time.time() - t0
        diag_ok, diag = try_http("GET", args.base_url + "/_ds41f/diagnostics", timeout=20) if args.diagnostic_endpoints else (False, None)
        if not ok:
            turns.append({"turn": i, "ok": False, "error": resp, "elapsed_s": elapsed}); break
        amsg = assistant_message(resp); messages.append(amsg)
        calls = call_ids(amsg); tool_results = []
        for cid, name, targs in calls:
            result = run_tool(workspace, name, targs); tool_cycles += 1
            tool_msg = {"role": "tool", "tool_call_id": cid, "content": json.dumps(result, sort_keys=True)}
            messages.append(tool_msg); tool_results.append({"id": cid, "name": name, "args": targs, "result": result})
        turn_rec = {"turn": i, "ok": True, "elapsed_s": elapsed, "request_sha256": sha_obj(body), "response_sha256": sha_obj(resp), "tool_call_count": len(calls), "tool_results": tool_results, "rss_bytes": rss_bytes(pid), "diagnostics_tail": (diag.get("session_traces", [])[-1] if diag_ok else None)}
        turns.append(turn_rec)
        if i in {1, args.cancel_turn, args.persist_after, args.turns} or calls:
            exact_samples.append({"turn": i, "request": body, "response": resp, "tool_results": tool_results})
        if i == args.persist_after:
            p = http("POST", f"{args.base_url}/v1/sessions/{sid}/persist", {"artifact_root": str(ROOT/"artifacts/m24/kv")}, timeout=1200)
            persisted = p
            if server:
                server.terminate(); server.wait(timeout=120)
                server = subprocess.Popen(server_cmd, cwd=Path.cwd(), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True); pid = server.pid
                for _ in range(240):
                    okh, h = try_http("GET", args.base_url + "/health", timeout=2)
                    if okh and h.get("model_ready"): break
                    time.sleep(1)
            old_sid = sid; sid = "m24_soak_restored"
            http("POST", args.base_url + "/v1/sessions/restore", {"artifact_path": p["artifact"]["path"], "id": sid}, timeout=1200)
            restored = True
            turns[-1]["persist_restore"] = {"old_session": old_sid, "new_session": sid, "artifact": p}
        if not calls:
            messages.append({"role": "user", "content": "Continue the same task. If more evidence is needed, use a tool; otherwise summarize current status briefly."})

    final_diag_ok, final_diag = try_http("GET", args.base_url + "/_ds41f/diagnostics", timeout=20) if args.diagnostic_endpoints else (False, None)
    close_ok, close = try_http("DELETE", f"{args.base_url}/v1/sessions/{sid}", timeout=120)
    artifact = {
        "schema": "ds41f.m24.operational-soak.v1", "created_at": started, "duration_s": time.time()-started,
        "client_harness": "tools/run_m24_operational_soak.py controlled OpenAI-compatible tool-loop client",
        "base_url": args.base_url, "server_pid": pid, "workspace": str(workspace), "turn_target": args.turns,
        "turn_count": len([t for t in turns if t.get("ok")]), "tool_cycles": tool_cycles,
        "persisted": persisted, "restored": restored, "cancellation": cancellation,
        "turns": turns, "exact_transcript_samples": exact_samples[-12:], "final_diagnostics": final_diag if final_diag_ok else None,
        "close": {"ok": close_ok, "result": close}, "resource_samples": {"final_rss_bytes": rss_bytes(pid), **vm_stat()},
        "classification": {"repetition_incidents": [], "runtime_replay_or_repack": "see diagnostics prompt_replay_count/full_cache_repack_count", "notes": "No heuristic loop suppression or hidden retries are performed by this harness."},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True)+"\n")
    if server:
        server.terminate()
    print(json.dumps({"artifact": str(args.output), "turns": artifact["turn_count"], "tool_cycles": tool_cycles}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
