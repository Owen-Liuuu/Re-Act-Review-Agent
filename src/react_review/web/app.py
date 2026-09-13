"""Minimal web UI: read run files, post checkpoint decisions, start one run."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from react_review.hitl.gate import Decision
from react_review.hitl.web import StaleDecision
from react_review.core.config import keys_required, load_config
from react_review.core.exceptions import ConfigError
from react_review.web import html as pages
from react_review.web.auth import BasicAuthMiddleware
from react_review.web.eta import decorate_live
from react_review.web.gears import (
    overlay_from_form,
    parse_gears,
    parse_keys,
    partial_customer_keys,
    redacted_gears,
)
from react_review.web.reads import (
    blocked_step,
    list_run_ids,
    load_live,
    load_steps,
    run_dir,
    run_status,
    safe_run_id,
)
from react_review.web.runner import RunRunner, default_start, make_gate, write_studies_csv


UPLOAD_RETENTION = """\
This folder holds the review PDF and any source PDFs uploaded for this run.

They stay on this machine only. ReAct-Review does not copy them to a remote
host, attach them to a shareable report, or put them in Git.

Keep the folder while you may re-run or inspect the audit. Delete
output/uploads/<run_id>/ and output/runs/<run_id>/ when you no longer need
them — that removes the PDFs, the extraction/semantic caches, and any model
replies that quoted the papers.

Caches written under output/runs/<run_id>/ for an upload run are marked
shareable=false. Do not redistribute those files.
"""


def create_app(
    *,
    runs_dir: Path,
    uploads_dir: Path,
    config_path: Path,
    basic_auth: tuple[str, str] | None,
    runner: RunRunner | None = None,
) -> Starlette:
    runs_dir = Path(runs_dir)
    uploads_dir = Path(uploads_dir)
    runner = runner or RunRunner(default_start)

    def _host():
        return load_config(config_path)

    def _home(*, error: str = "", status_code: int = 200) -> HTMLResponse:
        host = _host()
        return HTMLResponse(pages.home_page(
            list_run_ids(runs_dir), busy=runner.current, error=error,
            require_keys=keys_required(host), routing=host.routing),
            status_code=status_code)

    async def home(request: Request) -> Response:
        return _home()

    async def get_run(request: Request) -> Response:
        try:
            run_id = safe_run_id(request.path_params["run_id"])
        except ValueError:
            return Response("invalid run id", status_code=404)
        directory = run_dir(runs_dir, run_id)
        if not directory.is_dir():
            return Response("run not found", status_code=404)
        gate = runner.gates.get(run_id)
        waiting = gate.waiting if gate is not None else None
        steps = load_steps(directory)
        blocked = None
        if waiting is not None:
            blocked = waiting.model_dump(mode="json")
        elif gate is None:
            blocked = blocked_step(steps)
        status = run_status(
            directory,
            waiting_index=(waiting.index if waiting is not None else None),
        )
        if waiting is not None:
            status = "waiting"
        elif runner.current == run_id or runner.starting(run_id):
            # package.partial.json is a crash snapshot written after every
            # paper; it is not a stop. Only fail.json / package.json are
            # terminal while this worker still holds the run.
            if status not in {"complete", "failed"}:
                status = "running"
        elif runner.queued(run_id):
            status = "queued"
        live = None
        if status in {"running", "queued", "failed"}:
            live = decorate_live(load_live(directory), steps)
        body = {
            "run_id": run_id,
            "status": status,
            "blocked": blocked,
            "steps": steps,
            "live": live,
            "has_report": (directory / "report.html").is_file(),
            "has_log": (directory / "checkpoints.log").is_file(),
            "has_package": (directory / "package.json").is_file()
            or (directory / "package.partial.json").is_file(),
        }
        if "application/json" in (request.headers.get("accept") or ""):
            return JSONResponse(body)
        return HTMLResponse(pages.run_page(
            run_id, status=status, steps=steps, blocked=blocked,
            has_report=body["has_report"], has_log=body["has_log"],
            live=live, has_package=body["has_package"],
            poll=True))

    async def get_report(request: Request) -> Response:
        directory = _existing_dir(runs_dir, request)
        if isinstance(directory, Response):
            return directory
        path = directory / "report.html"
        if not path.is_file():
            return Response("report not ready", status_code=404)
        return FileResponse(path, media_type="text/html")

    async def get_package(request: Request) -> Response:
        directory = _existing_dir(runs_dir, request)
        if isinstance(directory, Response):
            return directory
        path = directory / "package.json"
        if not path.is_file():
            path = directory / "package.partial.json"
        if not path.is_file():
            return Response("package not ready", status_code=404)
        return FileResponse(path, media_type="application/json")

    async def get_log(request: Request) -> Response:
        directory = _existing_dir(runs_dir, request)
        if isinstance(directory, Response):
            return directory
        path = directory / "checkpoints.log"
        if not path.is_file():
            return Response("log not ready", status_code=404)
        return FileResponse(path, media_type="text/plain")

    async def post_decision(request: Request) -> Response:
        try:
            run_id = safe_run_id(request.path_params["run_id"])
        except ValueError:
            return Response("invalid run id", status_code=404)
        gate = runner.gates.get(run_id)
        if gate is None:
            return Response("no live gate for this run", status_code=409)
        payload = await _decision_payload(request)
        try:
            index = int(payload["index"])
            decision = Decision(str(payload["decision"]))
        except (KeyError, ValueError, TypeError):
            return Response("need index and a known decision", status_code=400)
        dropped = payload.get("dropped")
        if dropped is not None and not isinstance(dropped, list):
            return Response("dropped must be a list of ids", status_code=400)
        try:
            gate.submit(index, decision, dropped=dropped)
        except StaleDecision as exc:
            return Response(str(exc), status_code=409)
        if "text/html" in (request.headers.get("accept") or "text/html"):
            return Response(status_code=303, headers={"Location": f"/runs/{run_id}"})
        return JSONResponse({"ok": True, "index": index, "decision": decision.value})

    async def post_runs(request: Request) -> Response:
        form = await request.form()
        upload = form.get("review")
        host = _host()
        require = keys_required(host)
        if upload is None or not getattr(upload, "filename", ""):
            return _home(error="No review PDF detected.", status_code=400)
        keys = parse_keys(form)
        if partial_customer_keys(form) or (require and not all(keys)):
            return _home(
                error="API keys missing. Open Account (top right) and paste "
                      "the native vendor key for each gear, then Start. "
                      "Not an OpenRouter key.",
                status_code=400)
        try:
            overlay = overlay_from_form(host, form)
            gears = parse_gears(form)
        except ConfigError as exc:
            return _home(error=str(exc), status_code=400)
        run_id = uuid.uuid4().hex[:12]
        dest = uploads_dir / run_id
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "RETENTION.txt").write_text(UPLOAD_RETENTION, encoding="utf-8")
        # Exist before the 303 so a refresh does not 404 while the worker starts.
        run_path = runs_dir / run_id
        run_path.mkdir(parents=True, exist_ok=True)
        review_path = dest / _safe_name(upload.filename, "review.pdf")
        review_path.write_bytes(await upload.read())
        source_files = form.getlist("sources") if hasattr(form, "getlist") else []
        saved: list[Path] = []
        for item in source_files:
            if item is None or not getattr(item, "filename", ""):
                continue
            path = dest / _safe_name(item.filename, "source.pdf")
            path.write_bytes(await item.read())
            saved.append(path)
        argv = [
            "--pdf", str(review_path),
            "--config", str(config_path),
            "--out", str(runs_dir),
            "--run-id", run_id,
            "--non-interactive",
        ]
        if saved:
            csv_path = dest / "included_studies.csv"
            write_studies_csv(csv_path, saved)
            argv += ["--studies", str(csv_path), "--pdf-dir", str(dest)]
        record = redacted_gears(
            overlay, complex=gears[0], simple=gears[1], visual=gears[2],
            name=str(form.get("name") or "").strip(),
            email=str(form.get("email") or "").strip(),
        )
        (dest / "gears.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8")
        (run_path / "gears.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8")
        gate = make_gate()
        state = runner.enqueue((run_id, argv, gate, overlay))
        (dest / "launch.json").write_text(
            json.dumps({"run_id": run_id, "argv": argv, "state": state}, indent=2),
            encoding="utf-8")
        return Response(status_code=303, headers={"Location": f"/runs/{run_id}"})

    routes = [
        Route("/", home, methods=["GET"]),
        Route("/runs", post_runs, methods=["POST"]),
        Route("/runs/{run_id}", get_run, methods=["GET"]),
        Route("/runs/{run_id}/report", get_report, methods=["GET"]),
        Route("/runs/{run_id}/package", get_package, methods=["GET"]),
        Route("/runs/{run_id}/log", get_log, methods=["GET"]),
        Route("/runs/{run_id}/decision", post_decision, methods=["POST"]),
    ]
    app = Starlette(routes=routes)
    app.state.runner = runner
    app.state.runs_dir = runs_dir
    if basic_auth is not None:
        user, password = basic_auth
        app.add_middleware(BasicAuthMiddleware, username=user, password=password)
    return app


def _existing_dir(runs_dir: Path, request: Request) -> Path | Response:
    try:
        run_id = safe_run_id(request.path_params["run_id"])
    except ValueError:
        return Response("invalid run id", status_code=404)
    directory = run_dir(runs_dir, run_id)
    if not directory.is_dir():
        return Response("run not found", status_code=404)
    return directory


def _safe_name(name: str, fallback: str) -> str:
    stem = Path(name or "").name
    if not stem or stem in {".", ".."} or "/" in stem or "\\" in stem:
        return fallback
    if not stem.lower().endswith(".pdf"):
        return fallback
    return stem


async def _decision_payload(request: Request) -> dict:
    ctype = request.headers.get("content-type") or ""
    if "application/json" in ctype:
        data = await request.json()
        return data if isinstance(data, dict) else {}
    form = await request.form()
    keep = set(form.getlist("keep")) if hasattr(form, "getlist") else set()
    all_ids = form.getlist("all") if hasattr(form, "getlist") else []
    dropped = [i for i in all_ids if i not in keep] if all_ids else None
    return {
        "index": form.get("index"),
        "decision": form.get("decision"),
        "dropped": dropped,
        "note": form.get("note") or "",
    }
