"""Bounded process-parallel evaluation of tied species-scale assignments.

Each task processes one complete assignment, including its anchors and scores.
Integration and ordered result selection remain in the parent process.
The executor runs in a package-owned interpreter, so spawned workers do not
import the calling application. Structural summaries stay in the caller.
"""
from __future__ import annotations
from itertools import islice
from contextlib import closing
import multiprocessing as mp
import os
from pathlib import Path
import time
from ._assignment_pool import pool_results

_STATE = None
_THREAD_LIMITER = None


def cpu_capacity():
    """Return the CPU capacity allowed by affinity and detected cgroup quotas."""
    counts = [os.cpu_count() or 1]
    if hasattr(os, 'process_cpu_count'):
        counts.append(os.process_cpu_count() or 1)
    if hasattr(os, 'sched_getaffinity'):
        counts.append(len(os.sched_getaffinity(0)))
    # Check cgroup v2 ancestors, including the root of a cgroup namespace.
    roots = [Path('/sys/fs/cgroup')]
    try:
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            if line.startswith('0::'):
                roots.append(Path('/sys/fs/cgroup') / line[3:].lstrip('/'))
    except OSError:
        pass
    visited = set()
    for root in roots:
        for parent in (root, *root.parents):
            if str(parent) in visited or not str(parent).startswith('/sys/fs/cgroup'):
                continue
            visited.add(str(parent))
            try:
                quota, period = (parent/'cpu.max').read_text().split()
                if quota != 'max':
                    counts.append(max(1, int(quota)//int(period)))
            except (OSError, ValueError):
                pass
    # Common cgroup v1 layout.
    try:
        root = Path('/sys/fs/cgroup/cpu')
        quota = int((root/'cpu.cfs_quota_us').read_text())
        period = int((root/'cpu.cfs_period_us').read_text())
        if quota > 0:
            counts.append(max(1, quota//period))
    except (OSError, ValueError):
        pass
    return max(1, min(counts))


def resolve_workers(n_jobs):
    """Return the process limit after reserving one available CPU.

    Args:
        n_jobs: Positive process limit, or None for automatic selection.

    Returns:
        At least one, respecting CPU capacity and the Windows limit of 61.
        The evaluator further caps this count by the available assignments;
        a count of one runs in the parent without creating a pool.
    """
    if n_jobs is not None and (not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs < 1):
        raise ValueError('n_jobs must be None (automatic) or a positive integer.')
    capacity = max(1, cpu_capacity() - 1)
    wanted = capacity if n_jobs is None else n_jobs
    return min(wanted, capacity, 61 if os.name == 'nt' else wanted)


def _init_worker(state):
    global _STATE, _THREAD_LIMITER
    _STATE = state
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    # Limit existing native thread pools when threadpoolctl is available.
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        _THREAD_LIMITER = None
    else:
        _THREAD_LIMITER = threadpool_limits(limits=1)


def _evaluate(index, scales, state):
    from .process_CRN_given_nondimensionalisation import process_CRN_given_nondimensionalisation
    S, V, X, basis_type, dimensionally_consistent_sets = state
    started, cpu_started = time.perf_counter(), time.process_time()
    try:
        output = process_CRN_given_nondimensionalisation(
            S, V, X, nondim=dict(zip(X, scales)), basis_type=basis_type,
            return_all_families=True,
            dimensionally_consistent_sets=dimensionally_consistent_sets,
        )
    except Exception as exc:
        raise RuntimeError(f'Tied scale assignment {index} failed; no minimum is returned.') from exc
    finished = time.perf_counter()
    # Return only the retained local minima, not the repeated model matrices.
    return {'families': output.get('families', [])}, {
        'index': index, 'pid': os.getpid(), 'start': started, 'end': finished,
        'wall_seconds': finished-started, 'cpu_seconds': time.process_time()-cpu_started,
        'local_scores': [tuple(int(f[k]) for k in ('L1_cost', 'gen_L1_cost', 'time_L1_cost'))
                         for f in output.get('families', [])],
    }


def _evaluate_worker(job):
    if _STATE is None:
        raise RuntimeError('Assignment worker was not initialized.')
    return _evaluate(job[0], job[1], _STATE)


def evaluate_assignments(jobs, state, n_jobs, start_method, diagnostics, stats):
    """Yield assignment results in input order with bounded lookahead.

    Args:
        jobs: Iterable of (choice, complete_species_scale_tuple) pairs.
        state: Shared model tuple (S, V, X, basis_type, dimensional_sets).
        n_jobs: Positive process limit, or None for automatic selection.
        start_method: Multiprocessing context, such as "spawn".
        diagnostics: Whether to record per-task timings and process IDs.
        stats: Dictionary populated with execution counts and timing data.

    Yields:
        (choice, result) pairs in input order, preserving deterministic ties.

    At most twice the effective worker count is submitted ahead. The count is
    capped by available CPUs minus one, n_jobs and the number of assignments.
    One CPU or one assignment uses the parent; empty input performs no work.
    A dedicated coordinator creates the process pool without importing the
    application. This prevents repeated setup and output under spawn, including
    when the caller invokes the search at module scope. Ordinary model and scale
    objects must still be pickleable. Frozen executables should use n_jobs=1.
    """
    workers = resolve_workers(n_jobs)
    iterator = iter(jobs)
    initial = list(islice(iterator, 2*workers))
    workers = min(workers, len(initial))
    stats.update(requested_workers=n_jobs, cpu_capacity=cpu_capacity(),
                 effective_workers=workers, assignments_completed=0,
                 process_ids=[], backend='serial', start_method=None,
                 max_in_flight=0, sum_assignment_cpu_seconds=0.0,
                 sum_assignment_wall_seconds=0.0)
    if diagnostics:
        stats['tasks'] = []
    if not initial:
        return
    seen_pids = set()

    def record(data):
        stats['assignments_completed'] += 1
        seen_pids.add(data['pid'])
        stats['process_ids'] = sorted(seen_pids)
        stats['sum_assignment_cpu_seconds'] += data['cpu_seconds']
        stats['sum_assignment_wall_seconds'] += data['wall_seconds']
        if diagnostics:
            stats['tasks'].append(data)

    if workers == 1:
        from itertools import chain
        for index, (choice, scales) in enumerate(chain(initial, iterator)):
            result, data = _evaluate(index, scales, state)
            record(data)
            yield choice, result
        return
    if mp.current_process().daemon:
        raise RuntimeError('A daemonic process cannot create the assignment pool; use n_jobs=1 there.')
    mp.get_context(start_method)  # Validate before launching the coordinator.
    stats.update(backend='process', start_method=start_method)
    with closing(pool_results(initial, iterator, state, workers,
                              start_method, stats)) as results:
        for choice, result, data in results:
            record(data)
            yield choice, result
