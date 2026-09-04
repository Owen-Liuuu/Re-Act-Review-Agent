"""One live audit at a time. Each run is a background thread; state is on disk."""
from __future__ import annotations

import csv
import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

import structlog

from react_review.hitl.policy import CheckpointPolicy
from react_review.hitl.web import WebCheckpoint

logger = structlog.get_logger(__name__)


class RunRunner:
    """Serial launcher. A second POST waits until the current run finishes."""

    def __init__(self, start: Callable[..., None]) -> None:
        self._start = start
        self._lock = threading.Lock()
        self._current: str | None = None
        self._queued: deque[tuple] = deque()
        self.gates: dict[str, WebCheckpoint] = {}
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._jobs = threading.Semaphore(0)
        self._worker.start()

    @property
    def current(self) -> str | None:
        with self._lock:
            return self._current

    def queued(self, run_id: str) -> bool:
        with self._lock:
            jobs = list(self._queued)
            if self._current is None and jobs and jobs[0][0] == run_id:
                return False
            return any(job[0] == run_id for job in jobs)

    def starting(self, run_id: str) -> bool:
        """True when this id is next (or live) so the desk should not say Queued."""
        with self._lock:
            if self._current == run_id:
                return True
            jobs = list(self._queued)
            return self._current is None and bool(jobs) and jobs[0][0] == run_id

    def enqueue(self, job: tuple) -> str:
        """job = (run_id, argv, gate[, overlay]). Returns 'running' or 'queued'."""
        run_id, _argv, gate = job[0], job[1], job[2]
        with self._lock:
            self.gates[run_id] = gate
            if self._current is None and not self._queued:
                self._queued.append(job)
                state = "running"
            else:
                self._queued.append(job)
                state = "queued"
        self._jobs.release()
        return state

    def _loop(self) -> None:
        while True:
            self._jobs.acquire()
            with self._lock:
                if not self._queued:
                    continue
                item = self._queued.popleft()
                run_id, argv, gate = item[0], item[1], item[2]
                overlay = item[3] if len(item) > 3 else None
                self._current = run_id
            logger.info("web_run_start", run_id=run_id)
            try:
                self._invoke(argv, gate, overlay)
            except SystemExit as exc:
                logger.warning("web_run_exited", run_id=run_id, code=exc.code)
                write_run_fail(argv, _exit_message(exc.code))
            except Exception:
                logger.exception("web_run_crashed", run_id=run_id)
                write_run_fail(argv, "the worker crashed; see the serve terminal")
            finally:
                with self._lock:
                    self._current = None
                    self.gates.pop(run_id, None)

    def _invoke(self, argv: list[str], gate: WebCheckpoint, overlay) -> None:
        if overlay is not None:
            try:
                self._start(argv, gate, overlay)
                return
            except TypeError:
                pass
        self._start(argv, gate)


def _flag(argv: list[str], name: str) -> str:
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError):
        return ""


def _exit_message(code) -> str:
    if code == 3:
        return (
            "The provider refused the request (often HTTP 402: no remaining "
            "credit). Open Account, paste a funded native key, and start a new run."
        )
    if code == 2:
        return "The run was stopped at a checkpoint."
    return f"The run exited ({code})."


def write_run_fail(argv: list[str], message: str) -> None:
    """Leave fail.json + live.json so the desk is not stuck on Queued."""
    out = _flag(argv, "--out")
    run_id = _flag(argv, "--run-id")
    if not out or not run_id:
        return
    directory = Path(out) / run_id
    directory.mkdir(parents=True, exist_ok=True)
    error = (message or "the run failed").strip()[:800]
    (directory / "fail.json").write_text(
        json.dumps({"state": "failed", "error": error}, indent=2),
        encoding="utf-8")
    now = time.time()
    (directory / "live.json").write_text(
        json.dumps({
            "stage": "failed",
            "label": "failed",
            "caption": error,
            "state": "failed",
            "started_unix": now,
            "updated_unix": now,
        }),
        encoding="utf-8")


def write_studies_csv(path: Path, source_pdfs: list[Path]) -> None:
    """included_studies.csv for --studies / --pdf-dir (local PDFs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["study_id", "doi", "source_pdf"])
        for pdf in source_pdfs:
            w.writerow([pdf.stem, "", pdf.name])


def default_start(argv: list[str], gate: WebCheckpoint, overlay=None) -> None:
    from react_review.cli import _run_main
    from react_review.production import ProductionDependencies

    try:
        _run_main(
            argv, dependencies=ProductionDependencies(gate=gate, config=overlay))
    except SystemExit as exc:
        if exc.code not in (0, None):
            write_run_fail(argv, _exit_message(exc.code))
        # Do not re-raise: SystemExit would kill the serve worker thread.


def make_gate() -> WebCheckpoint:
    return WebCheckpoint(CheckpointPolicy.key_stages())
