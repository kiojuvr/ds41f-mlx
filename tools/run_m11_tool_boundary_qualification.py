#!/usr/bin/env python3
"""Milestone 11 real-model tool-boundary qualification harness.

Runs Chat Completions function-tool round trips through the M8/M9 live session
rather than rebuilding the conversation after the tool result.
"""
from __future__ import annotations

import argparse, json, sys, subprocess, time
from pathlib import Path
from typing import Any
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT, DEFAULT_OMLX, OmlxRuntime, OmlxRuntimeConfig
from ds41f_mlx.runtime.tool_boundary_session import M11RecipeToolSession
from ds41f_mlx.serving.deepseek_recipe_backend import DEFAULT_RECIPE, DEFAULT_MODEL_ID
from ds41f_mlx.serving.server import load_v41_tokenizer, prepare_request, RequestError


def git_rev(path: Path) -> str | None:
    try: return subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"], text=True).strip()
    except Exception: return None


def tool_def() -> dict[str, Any]:
    return {"type":"function","function":{"name":"lookup_weather","description":"Return deterministic weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":False},"strict":True}}


def initial_body(city: str, *, multi: bool = False) -> dict[str, Any]:
    task = f"Use the tool to look up the weather in {city}, then answer concisely."
    if multi:
        task = "Use the tool for Paris and Berlin, then summarize both results concisely."
    return {"model":DEFAULT_MODEL_ID,"messages":[{"role":"user","content":task}],"tools":[tool_def()],"tool_choice":{"type":"function","function":{"name":"lookup_weather"}},"reasoning_effort":"none","temperature":0,"max_tokens":96}


def tool_stub(name: str, arguments: str) -> str:
    args = json.loads(arguments or "{}")
    city = args.get("city", "unknown")
    table = {"Paris":"sunny 21C", "Berlin":"cloudy 17C"}
    return json.dumps({"city": city, "weather": table.get(city, "clear 20C"), "source": "m11_deterministic_stub"}, separators=(",",":"))


def continuation_body(first_body: dict[str, Any], tool_call: Any, tool_result: str, *, max_tokens: int = 96) -> dict[str, Any]:
    # Keep the same ordinary client tool definition in the complete recipe
    # conversation so the next encoding is an exact extension of the original
    # tool-enabled prompt.  Tool choice returns to auto after the required call.
    return {"model":DEFAULT_MODEL_ID,"messages":[
        first_body["messages"][0],
        {"role":"assistant","content":None,"tool_calls":[{"id":tool_call.id,"type":"function","function":{"name":tool_call.name,"arguments":tool_call.arguments}}]},
        {"role":"tool","tool_call_id":tool_call.id,"content":tool_result},
    ],"tools":[tool_def()],"tool_choice":"auto","reasoning_effort":"none","temperature":0,"max_tokens":max_tokens}


def prepare(body: dict[str, Any], tok: Any, recipe: Path):
    return prepare_request("chat_completions", json.dumps(body).encode(), tokenizer=tok, recipe_path=recipe)


def _run_to_tool_boundary(args: argparse.Namespace) -> tuple[dict[str, Any], Any, Any, Any, Any]:
    tok = load_v41_tokenizer(args.recipe_path)
    rt = OmlxRuntime(OmlxRuntimeConfig(omlx_path=args.omlx_path, checkpoint_path=args.checkpoint, engram_ssd_offload=True, preserve_mtp=False))
    model, _ = rt.load_model()
    body0 = initial_body("Paris")
    prep0 = prepare(body0, tok, args.recipe_path)
    sess = M11RecipeToolSession.start_from_prepared(model=model, tokenizer=tok, checkpoint=args.checkpoint, omlx_path=args.omlx_path, recipe_path=args.recipe_path, prepared=prep0)
    turn0 = sess.run_current_assistant_turn(prep0)
    if not turn0.tool_calls:
        raise RuntimeError("model did not produce a parsed tool call")
    call = turn0.tool_calls[0]
    result = tool_stub(call.name, call.arguments)
    return body0, call, result, turn0, (rt, tok, sess)


def _finish_after_tool(args: argparse.Namespace, *, body0: dict[str, Any], call: Any, result: str, turn0: Any, sess: M11RecipeToolSession, artifact: Any = None) -> dict[str, Any]:
    tok = load_v41_tokenizer(args.recipe_path)
    body1 = continuation_body(body0, call, result)
    prep1 = prepare(body1, tok, args.recipe_path)
    prefix_ok = prep1.token_ids[:sess.m8.frontier] == sess.m8.token_history
    before = sess.m8.frontier
    sess.continue_from_prepared(prep1)
    turn1 = sess.run_current_assistant_turn(prep1)
    diag = sess.diagnostics()
    return {"persist_restore":artifact is not None,"tool_call":call.to_json(),"tool_result":result,"artifact":None if artifact is None else artifact.to_json(),"prefix_extension":{"ok":prefix_ok,"frontier_before":before,"next_prompt_tokens":len(prep1.token_ids),"suffix_tokens":len(prep1.token_ids)-before},"turn0":turn0.to_json() if hasattr(turn0, 'to_json') else turn0,"turn1":turn1.to_json(),"diagnostics":diag,"ok": bool(prefix_ok and diag["m8"]["total_prompt_replay_count"]==0 and diag["m8"]["total_full_cache_repack_count"]==0 and diag["m8"]["all_cache_offsets_equal_frontier"])}


def run_flow(args: argparse.Namespace, *, persist_restore: bool) -> dict[str, Any]:
    if persist_restore:
        save = run_persisted_save(args)
        rt = OmlxRuntime(OmlxRuntimeConfig(omlx_path=args.omlx_path, checkpoint_path=args.checkpoint, engram_ssd_offload=True, preserve_mtp=False))
        try:
            model, _ = rt.load_model(); tok = load_v41_tokenizer(args.recipe_path)
            sess = M11RecipeToolSession.restore(model=model, tokenizer=tok, checkpoint=args.checkpoint, omlx_path=args.omlx_path, recipe_path=args.recipe_path, artifact_path=Path(save["artifact"]["path"]), protocol="chat_completions", model_id=DEFAULT_MODEL_ID)
            call = _dict_obj(save["tool_call"])
            return _finish_after_tool(args, body0=save["body0"], call=call, result=save["tool_result"], turn0=save["turn0"], sess=sess, artifact=type("Artifact", (), {"to_json": lambda self: save["artifact"]})())
        finally:
            rt.close()
    body0, call, result, turn0, owned = _run_to_tool_boundary(args)
    rt, _tok, sess = owned
    try:
        return _finish_after_tool(args, body0=body0, call=call, result=result, turn0=turn0, sess=sess)
    finally:
        rt.close()


def run_persisted_save(args: argparse.Namespace) -> dict[str, Any]:
    body0, call, result, turn0, owned = _run_to_tool_boundary(args)
    rt, _tok, sess = owned
    try:
        artifact = sess.persist_idle(artifact_root=args.kv_root, diagnostics={"m11_phase":"after_tool_call_before_tool_result"})
        return {"body0":body0,"tool_call":call.to_json(),"tool_result":result,"turn0":turn0.to_json(),"artifact":artifact.to_json(),"ok":True}
    finally:
        rt.close()


def run_persisted_resume(args: argparse.Namespace) -> dict[str, Any]:
    save = json.loads(Path(args.resume_input).read_text())
    rt = OmlxRuntime(OmlxRuntimeConfig(omlx_path=args.omlx_path, checkpoint_path=args.checkpoint, engram_ssd_offload=True, preserve_mtp=False))
    try:
        model, _ = rt.load_model(); tok = load_v41_tokenizer(args.recipe_path)
        sess = M11RecipeToolSession.restore(model=model, tokenizer=tok, checkpoint=args.checkpoint, omlx_path=args.omlx_path, recipe_path=args.recipe_path, artifact_path=Path(save["artifact"]["path"]), protocol="chat_completions", model_id=DEFAULT_MODEL_ID)
        call = _dict_obj(save["tool_call"])
        artifact = type("Artifact", (), {"to_json": lambda self: save["artifact"]})()
        return _finish_after_tool(args, body0=save["body0"], call=call, result=save["tool_result"], turn0=save["turn0"], sess=sess, artifact=artifact)
    finally:
        rt.close()


def _dict_obj(data: dict[str, Any]) -> Any:
    obj = SimpleNamespace(**data)
    obj.to_json = lambda: data
    return obj


def invalid_result_gate(args: argparse.Namespace) -> dict[str, Any]:
    tok = load_v41_tokenizer(args.recipe_path)
    bad = {"model":DEFAULT_MODEL_ID,"messages":[
        {"role":"user","content":"Use tool."},
        {"role":"assistant","content":None,"tool_calls":[{"id":"call_good","type":"function","function":{"name":"lookup_weather","arguments":"{\"city\":\"Paris\"}"}}]},
        {"role":"tool","tool_call_id":"call_bad","content":"{}"},
    ],"reasoning_effort":"none","temperature":0,"max_tokens":8}
    try:
        prepare(bad, tok, args.recipe_path)
        return {"ok": False, "error": "invalid tool result was accepted"}
    except Exception as e:
        return {"ok": True, "error_type": type(e).__name__, "message": str(e)[:300]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--omlx-path", type=Path, default=DEFAULT_OMLX)
    ap.add_argument("--recipe-path", type=Path, default=DEFAULT_RECIPE)
    ap.add_argument("--kv-root", type=Path, default=Path("/Volumes/USB-SSD-RAID-0/ds41f-mlx/kv-m11"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/m11/tool-boundary-qualification.json"))
    ap.add_argument("--phase", choices=("all","live","persisted","persisted-save","persisted-resume"), default="all")
    ap.add_argument("--resume-input", type=Path)
    args = ap.parse_args()
    if args.phase == "live":
        print(json.dumps(run_flow(args, persist_restore=False)))
        return 0
    if args.phase == "persisted-save":
        print(json.dumps(run_persisted_save(args)))
        return 0
    if args.phase == "persisted-resume":
        print(json.dumps(run_persisted_resume(args)))
        return 0
    if args.phase == "persisted":
        base = [sys.executable, str(Path(__file__).resolve()), "--checkpoint", str(args.checkpoint), "--omlx-path", str(args.omlx_path), "--recipe-path", str(args.recipe_path), "--kv-root", str(args.kv_root)]
        save = json.loads(subprocess.check_output(base + ["--phase", "persisted-save"], text=True).strip().splitlines()[-1])
        tmp = args.kv_root / "m11-persisted-save.json"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(save))
        resume = json.loads(subprocess.check_output(base + ["--phase", "persisted-resume", "--resume-input", str(tmp)], text=True).strip().splitlines()[-1])
        print(json.dumps(resume))
        return 0
    rec = {"schema":"ds41f.m11.tool-boundary-qualification.v1","created_at":time.time(),"checkpoint":str(args.checkpoint),"omlx_path":str(args.omlx_path),"omlx_revision":git_rev(args.omlx_path),"recipe_path":str(args.recipe_path),"recipe_revision":git_rev(args.recipe_path),"protocols":["chat_completions"],"flows":[],"invalid_result_gate":None,"ok":False}
    rec["invalid_result_gate"] = invalid_result_gate(args)
    base = [sys.executable, str(Path(__file__).resolve()), "--checkpoint", str(args.checkpoint), "--omlx-path", str(args.omlx_path), "--recipe-path", str(args.recipe_path), "--kv-root", str(args.kv_root)]
    for phase in ("live", "persisted"):
        out = subprocess.check_output(base + ["--phase", phase], text=True)
        rec["flows"].append(json.loads(out.strip().splitlines()[-1]))
    rec["ok"] = bool(rec["invalid_result_gate"].get("ok") and all(f.get("ok") for f in rec["flows"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=2))
    print(json.dumps({"ok":rec["ok"],"out":str(args.out)}, indent=2))
    return 0 if rec["ok"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
