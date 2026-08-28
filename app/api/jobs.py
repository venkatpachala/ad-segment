"""In-memory job store for async detect.

Downloads can take 30–60s; the API returns a job_id immediately and the client
polls GET /v1/jobs/{id}. Process restart loses in-flight jobs (take-home OK).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from uuid import uuid4

from app.models.schemas import DetectRequest, DetectResponse, JobError, JobRecord, utc_now
from app.pipeline.ingest import IngestError
from app.pipeline.run import run_detect

_lock = threading.Lock()
_sem = threading.Semaphore(2)
_jobs: dict[str, JobState] = {}


@dataclass
class JobState:
    job_id: str
    status: str
    created_at: str
    req: DetectRequest
    error_code: str | None = None
    error_detail: str | None = None
    result: DetectResponse | None = None

    def to_record(self) -> JobRecord:
        err = None
        if self.error_code:
            err = JobError(code=self.error_code, detail=self.error_detail or "")
        return JobRecord(
            job_id=self.job_id,
            status=self.status,  # type: ignore[arg-type]
            created_at=self.created_at,
            error=err,
            result=self.result,
        )


def reset_jobs() -> None:
    with _lock:
        _jobs.clear()


def get_job(job_id: str) -> JobRecord | None:
    with _lock:
        job = _jobs.get(job_id)
        return job.to_record() if job else None


def wait_for_job(job_id: str, timeout_s: float) -> JobRecord | None:
    """Block until the job is terminal or *timeout_s* elapses."""
    deadline = time.time() + timeout_s
    rec = get_job(job_id)
    while rec is not None and rec.status not in {"done", "failed", "ingest_failed"}:
        if time.time() >= deadline:
            return rec
        time.sleep(0.15)
        rec = get_job(job_id)
    return rec


def submit_job(req: DetectRequest) -> JobRecord:
    job_id = uuid4().hex[:12]
    state = JobState(
        job_id=job_id,
        status="queued",
        created_at=utc_now(),
        req=req,
    )
    with _lock:
        _jobs[job_id] = state
    t = threading.Thread(target=_run_job, args=(job_id,), name=f"job-{job_id}", daemon=True)
    t.start()
    return state.to_record()


def _set_status(job_id: str, status: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        if job.status in {"done", "failed", "ingest_failed"}:
            return
        job.status = status


def _run_job(job_id: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        req = job.req

    def on_status(s: str) -> None:
        _set_status(job_id, s)

    try:
        with _sem:
            result = run_detect(req, run_id=job_id, on_status=on_status)
        with _lock:
            job = _jobs[job_id]
            job.result = result
            job.status = "done"
    except IngestError as e:
        with _lock:
            job = _jobs[job_id]
            job.status = "ingest_failed"
            job.error_code = e.code
            job.error_detail = str(e)
    except FileNotFoundError as e:
        with _lock:
            job = _jobs[job_id]
            job.status = "ingest_failed"
            job.error_code = "not_found"
            job.error_detail = str(e)
    except Exception as e:
        with _lock:
            job = _jobs[job_id]
            job.status = "failed"
            job.error_code = "pipeline_failed"
            job.error_detail = str(e)
