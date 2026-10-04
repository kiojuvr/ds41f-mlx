"""Local browser client for the qualified ds41f HTTP runtime.

Run separately from the model runtime, for example:
  python -m ds41f_mlx.serve
  python -m ds41f_mlx.web
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from ds41f_mlx.web_client import RuntimeClient, StatefulToolChatClient, RuntimeHTTPError
from ds41f_mlx.web_tools import registry_from_env, ToolError

STATIC_DIR = Path(__file__).with_name("web_static")


def create_app(*, runtime_base_url: str | None = None) -> FastAPI:
    runtime = RuntimeClient(runtime_base_url or os.environ.get("DS41F_RUNTIME_URL", "http://127.0.0.1:8000"))
    tools = registry_from_env()
    chat = StatefulToolChatClient(runtime, tools)
    app = FastAPI(title="ds41f-local-web-client", version="0.1.0")
    app.state.runtime = runtime
    app.state.tool_registry = tools

    @app.middleware('http')
    async def profile_handshake(request: Request, call_next):
        if request.method in ('POST','DELETE') and request.url.path.startswith('/api/'):
            try:
                if runtime.health().get('profile') == 'mtp-singleton-v1':
                    return JSONResponse(status_code=400, content={'error':{'code':'unsupported_capability',
                        'message':'Browser client does not support the local MTP profile'}})
            except RuntimeHTTPError as exc:
                return JSONResponse(status_code=502, content={'error':{'message':str(exc)}})
        return await call_next(request)

    @app.exception_handler(RuntimeHTTPError)
    async def runtime_error(_request: Request, exc: RuntimeHTTPError) -> Response:
        return JSONResponse(status_code=502 if exc.status >= 500 else exc.status, content={"error": {"message": str(exc), "runtime_status": exc.status, "runtime_body": exc.body}})

    @app.exception_handler(ToolError)
    async def tool_error(_request: Request, exc: ToolError) -> Response:
        return JSONResponse(status_code=400, content={"error": {"message": str(exc), "type": "tool_error"}})

    @app.get("/")
    async def index() -> Response:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/status")
    async def status(session_id: str | None = None) -> Response:
        data: dict[str, Any] = {"runtime_base_url": runtime.base_url, "tools": [t["function"]["name"] for t in tools.declarations]}
        try:
            data["runtime"] = runtime.health()
        except Exception as exc:
            data["runtime"] = {"status": "unreachable", "error": str(exc)}
        if session_id:
            try:
                data["session"] = runtime.get_session(session_id)
            except Exception as exc:
                data["session"] = {"id": session_id, "state": "unknown", "error": str(exc)}
        return JSONResponse(content=data)

    @app.post("/api/session")
    async def create_session(request: Request) -> Response:
        raw = await request.body()
        body = json.loads(raw.decode()) if raw else {}
        rec = runtime.create_session(body.get("id"))
        return JSONResponse(content=rec)

    @app.delete("/api/session/{session_id}")
    async def close_session(session_id: str) -> Response:
        return JSONResponse(content=runtime.close_session(session_id))

    @app.post("/api/chat")
    async def chat_turn(request: Request) -> Response:
        body = await request.json()
        session_id = body.get("session_id")
        message = body.get("message")
        transcript = body.get("transcript") or []
        if not isinstance(session_id, str) or not session_id:
            return JSONResponse(status_code=400, content={"error": {"message": "session_id is required"}})
        if not isinstance(message, str) or not message.strip():
            return JSONResponse(status_code=400, content={"error": {"message": "message is required"}})
        if not isinstance(transcript, list):
            return JSONResponse(status_code=400, content={"error": {"message": "transcript must be a list"}})
        out = chat.run_turn(session_id=session_id, transcript=transcript, user_message=message.strip(), max_tokens=int(body.get("max_tokens") or 512), temperature=float(body.get("temperature") or 0.0), tools_enabled=bool(body.get("tools_enabled", True)))
        return JSONResponse(content=out.to_json())

    @app.post("/api/session/{session_id}/persist")
    async def persist(session_id: str, request: Request) -> Response:
        raw = await request.body()
        body = json.loads(raw.decode()) if raw else {}
        return JSONResponse(content=runtime.persist(session_id, body.get("artifact_root")))

    @app.post("/api/session/restore")
    async def restore(request: Request) -> Response:
        body = await request.json()
        if not body.get("artifact_path"):
            return JSONResponse(status_code=400, content={"error": {"message": "artifact_path is required"}})
        return JSONResponse(content=runtime.restore(body["artifact_path"], body.get("id")))

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the local ds41f browser client")
    parser.add_argument("--host", default=os.environ.get("DS41F_WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DS41F_WEB_PORT", "8080")))
    parser.add_argument("--runtime-url", default=os.environ.get("DS41F_RUNTIME_URL", "http://127.0.0.1:8000"))
    args = parser.parse_args(argv)
    import uvicorn
    uvicorn.run(create_app(runtime_base_url=args.runtime_url), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
