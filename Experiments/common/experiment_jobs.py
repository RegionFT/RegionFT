"""Run independent experiment jobs in a process pool."""

import multiprocessing
import sys
import traceback
from typing import Any, NamedTuple


class ExperimentJob(NamedTuple):
    dataset: str
    model: str
    protected_pair: tuple
    method: str
    runtime: int
    repeat_id: int
    out_root: str
    config: dict
    cut_dir: Any = None


def run_job(job):
    """Run one job through the same entry point as the single-run CLI."""
    from runner import run_cell

    run_cell(
        job.dataset,
        job.model,
        job.protected_pair,
        job.method,
        job.runtime,
        1,
        job.out_root,
        start_label=job.repeat_id,
        regionft_config=job.config,
        cut_dir=job.cut_dir,
    )


def _job_tag(job):
    return (
        f"{job.method}-{job.model}-{job.dataset}-"
        f"{job.protected_pair[0]}-{job.repeat_id}"
    )


def run_and_extract(job):
    """Run a job and collect its compact analysis artifacts."""
    from Experiments.common.result_extraction import extract_after_run

    tag = _job_tag(job)
    try:
        run_job(job)
    except Exception:
        sys.stderr.write(f"[run] failed: {tag}\n")
        traceback.print_exc()

    try:
        return extract_after_run(job)
    except Exception:
        sys.stderr.write(f"[extract] failed: {tag}\n")
        traceback.print_exc()
        return None


def run_only(job):
    run_job(job)
    return None


def run_jobs(jobs, processes, extract=True):
    """Run jobs in parallel, optionally extracting each result in its worker."""
    worker = run_and_extract if extract else run_only
    with multiprocessing.Pool(processes=processes) as pool:
        return pool.map(worker, jobs)
