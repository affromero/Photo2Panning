"""One bounded job queue with process-group cancellation and staged outputs."""

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextlib import suppress
from queue import Queue
from typing import BinaryIO, cast

from klogr.path import (
    path_is_file,
    path_join,
    path_mkdir,
    path_open,
    path_read_text,
)

from pic2panning.schemas.creative.project import ProjectRequest, ProjectResult
from pic2panning.studio.files import Workspace
from pic2panning.studio.schemas import JobRecord, StudioJob, WorkerResult
from pic2panning.tools.io import file_sha256
from pic2panning.tools.projects.cache import CachedArtifact, CacheEntry
from pic2panning.tools.projects.registry import (
    file_inputs,
    outputs,
    redirect_outputs,
)
from pic2panning.tools.projects.runtime import publish


def stop_group(process: subprocess.Popen[bytes]) -> None:
    """Interrupt the entire worker group and reap it before returning."""
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)


def log_tail(path: str) -> str:
    """Read bounded diagnostics even when a model emits a large log."""
    if not path_is_file(path):
        return ""
    with cast("BinaryIO", path_open(path, "rb")) as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - 8192))
        return handle.read(8192).decode("utf-8", errors="replace")


class JobQueue:
    """Serialize expensive jobs while CPU previews use a separate lock."""

    workspace: Workspace
    lock: threading.Lock
    pending: Queue[str | None]
    records: dict[str, JobRecord]
    requests: dict[str, StudioJob]
    logs: dict[str, str]
    process: subprocess.Popen[bytes] | None
    active: str | None
    closed: bool
    thread: threading.Thread

    def __init__(self, workspace: Workspace) -> None:
        """Start one daemon dispatcher without loading any model framework."""
        self.workspace = workspace
        self.lock = threading.Lock()
        self.pending = Queue()
        self.records = {}
        self.requests = {}
        self.logs = {}
        self.process = None
        self.active = None
        self.closed = False
        self.thread = threading.Thread(
            target=self.dispatch, name="studio-job-queue", daemon=True
        )
        self.thread.start()

    def submit(self, request: StudioJob) -> JobRecord:
        """Freeze a checked request and return its durable session identifier."""
        normalized = self.workspace.job(request)
        with self.lock:
            if self.closed:
                raise RuntimeError("Studio job queue is closed")
            identifier = uuid.uuid4().hex
            record = JobRecord(
                id=identifier, kind=normalized.kind, submitted=time.time()
            )
            self.records[identifier] = record
            self.requests[identifier] = normalized
            self.pending.put(identifier)
        return record

    def snapshot(self, identifier: str) -> JobRecord:
        """Read immutable state and a bounded live subprocess log."""
        with self.lock:
            record = self.records[identifier]
            path = self.logs.get(identifier)
        return record.model_copy(
            update={"log": log_tail(path) if path is not None else record.log}
        )

    def snapshots(self) -> tuple[JobRecord, ...]:
        """Keep the newest jobs first for the status drawer."""
        with self.lock:
            identifiers = tuple(reversed(self.records))
        return tuple(self.snapshot(identifier) for identifier in identifiers)

    def cancel(self, identifier: str) -> JobRecord:
        """Cancel queued work or interrupt the active worker process group."""
        with self.lock:
            record = self.records[identifier]
            if record.status in {"completed", "failed", "cancelled"}:
                return record
            self.records[identifier] = record.model_copy(
                update={"status": "cancelled", "finished": time.time()}
            )
            process = self.process if self.active == identifier else None
        if process is not None:
            stop_group(process)
        return self.snapshot(identifier)

    def dispatch(self) -> None:
        """Run one immutable snapshot at a time until server shutdown."""
        while True:
            identifier = self.pending.get()
            try:
                if identifier is None:
                    return
                with self.lock:
                    record = self.records[identifier]
                    if record.status == "cancelled":
                        continue
                    self.records[identifier] = record.model_copy(
                        update={"status": "running", "started": time.time()}
                    )
                    self.active = identifier
                self.execute(identifier)
            finally:
                self.pending.task_done()

    def execute(self, identifier: str) -> None:
        """Stage worker outputs, publish completed artifacts, and clean storage."""
        directory = path_join(
            self.workspace.root, ".studio", "runs", identifier
        )
        request = self.requests[identifier]
        source = path_join(directory, "request.json")
        destination = path_join(directory, "result.json")
        log = path_join(directory, "worker.log")
        process: subprocess.Popen[bytes] | None = None
        created = False
        try:
            directory = self.workspace.path(directory, internal=True)
            path_mkdir(directory, parents=True, exist_ok=False)
            created = True
            with self.lock:
                self.logs[identifier] = log
            staged = (
                request
                if isinstance(request, ProjectRequest)
                else redirect_outputs(request, path_join(directory, "stage"))
            )
            with path_open(source, "w") as handle:
                handle.write(staged.model_dump_json() + "\n")
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            with cast("BinaryIO", path_open(log, "wb")) as diagnostics:
                process = self.launch(
                    source, destination, diagnostics, environment
                )
                with self.lock:
                    self.process = process
                    cancelled = self.records[identifier].status == "cancelled"
                if cancelled:
                    stop_group(process)
                process.wait()
            with self.lock:
                if self.records[identifier].status != "cancelled":
                    response = WorkerResult.model_validate_json(
                        path_read_text(destination)
                    )
                    self.complete(identifier, request, staged, response)
        except (
            ValueError,
            OSError,
            RuntimeError,
            subprocess.SubprocessError,
        ) as error:
            with self.lock:
                if self.records[identifier].status != "cancelled":
                    self.records[identifier] = self.records[
                        identifier
                    ].model_copy(
                        update={
                            "status": "failed",
                            "finished": time.time(),
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
        finally:
            if process is not None:
                stop_group(process)
            tail = log_tail(log) if created else ""
            with self.lock:
                self.records[identifier] = self.records[identifier].model_copy(
                    update={"log": tail}
                )
                self.logs.pop(identifier, None)
                self.process = None
                self.active = None
            if created:
                shutil.rmtree(directory)

    def complete(
        self,
        identifier: str,
        request: StudioJob,
        staged: StudioJob,
        response: WorkerResult,
    ) -> None:
        """Publish a completed worker result while holding the queue lock."""
        if response.error is not None or response.result is None:
            raise RuntimeError(response.error or "Worker returned no result")
        result = response.result
        if not isinstance(request, ProjectRequest) and not isinstance(
            staged, ProjectRequest
        ):
            entry = CacheEntry(
                fingerprint=identifier,
                artifacts=tuple(
                    CachedArtifact(path=path, sha256=file_sha256(path))
                    for path in outputs(staged)
                ),
                result=result,
                models={},
            )
            result = publish(
                request,
                entry,
                file_inputs(request),
                overwrite=request.overwrite,
            )
        declared = (
            ProjectResult.model_validate(result).outputs
            if isinstance(request, ProjectRequest)
            else outputs(request)
        )
        self.records[identifier] = self.records[identifier].model_copy(
            update={
                "status": "completed",
                "finished": time.time(),
                "outputs": tuple(
                    self.workspace.relative(path)
                    for path in declared
                    if path_is_file(path)
                ),
                "result": result,
            }
        )

    def launch(
        self,
        source: str,
        destination: str,
        diagnostics: BinaryIO,
        environment: dict[str, str],
    ) -> subprocess.Popen[bytes]:
        """Start the canonical worker as the leader of a separate process group."""
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                "pic2panning.studio.worker",
                source,
                destination,
            ],
            cwd=self.workspace.root,
            env=environment,
            stdout=diagnostics,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def close(self) -> None:
        """Cancel all work and join the dispatcher during server shutdown."""
        with self.lock:
            if self.closed:
                return
            self.closed = True
            identifiers = tuple(self.records)
        for identifier in identifiers:
            self.cancel(identifier)
        self.pending.put(None)
        self.thread.join(timeout=15)
