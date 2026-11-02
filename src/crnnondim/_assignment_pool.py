"""Process-pool bootstrap isolated from the application's main module.

The caller streams assignments to a private coordinator. Only the coordinator
creates the ProcessPoolExecutor, so spawned workers never import the caller's
script. Two inherited anonymous pipes carry requests and results; stdout and stderr
remain available for diagnostics. No network address or listener is used.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import struct
import subprocess
import sys
import threading
import time
import traceback

_HEADER = struct.Struct('!Q')
_STARTUP_TIMEOUT = 60.0
_WINDOWS = os.name == 'nt'
_HANDLE_LOCK = threading.Lock()
TRANSPORT = 'anonymous-pipes'


class _PipeConnection:
    """Binary request/result streams owned by one parent/coordinator pair."""

    def __init__(self, reader, writer):
        self.reader = reader
        self.writer = writer

    def read(self, size):
        return self.reader.read(size)

    def write_all(self, data):
        view = memoryview(data)
        while view:
            written = self.writer.write(view)
            if not written:
                raise BrokenPipeError('The assignment pipe stopped accepting data.')
            view = view[written:]

    def close(self):
        try:
            self.writer.close()
        finally:
            self.reader.close()


@contextmanager
def _pipe_pair():
    """Own both unbuffered ends of one anonymous OS pipe."""
    read_fd, write_fd = os.pipe()
    try:
        reader = os.fdopen(read_fd, 'rb', buffering=0)
    except BaseException:
        os.close(read_fd)
        os.close(write_fd)
        raise
    try:
        writer = os.fdopen(write_fd, 'wb', buffering=0)
    except BaseException:
        reader.close()
        os.close(write_fd)
        raise
    try:
        yield reader, writer
    finally:
        try:
            writer.close()
        finally:
            reader.close()



def _read_exact(connection, size):
    """Read a complete local protocol record, raising on unexpected closure."""
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.read(min(size - len(chunks), 1024 * 1024))
        if not chunk:
            raise EOFError('The assignment coordinator closed its connection.')
        chunks.extend(chunk)
    return bytes(chunks)


def _send(connection, value):
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    connection.write_all(_HEADER.pack(len(payload)))
    connection.write_all(payload)


def _receive(connection):
    size, = _HEADER.unpack(_read_exact(connection, _HEADER.size))
    # Only the parent and its child inherit these pipe ends. Messages contain
    # trusted local Python objects; do not reuse this protocol for untrusted input.
    return pickle.loads(_read_exact(connection, size))


def _python_path():
    """Preserve import paths, putting this package's checkout first."""
    paths = [str(Path(__file__).resolve().parents[1])]
    paths.extend(os.path.abspath(path or os.curdir) for path in sys.path
                 if isinstance(path, str))
    return list(dict.fromkeys(paths))


def _launch_coordinator(read_fd, write_fd, paths, environment):
    """Inherit only the coordinator's two protocol ends and standard output."""
    bootstrap = (
        'import sys\nsys.path[:0] = ' + repr(paths) + '\n'
        'from crnnondim._assignment_pool import main\n'
        'if __name__ == "__main__":\n    main()\n'
    )
    command = [sys.executable, '-u', '-c', bootstrap]
    options = dict(stdin=subprocess.DEVNULL, env=environment, close_fds=True)
    if not _WINDOWS:
        return subprocess.Popen(command + [str(read_fd), str(write_fd)],
                                pass_fds=(read_fd, write_fd), **options)
    import msvcrt
    handles = [msvcrt.get_osfhandle(fd) for fd in (read_fd, write_fd)]
    startup = subprocess.STARTUPINFO()
    startup.lpAttributeList = {'handle_list': handles}
    # Windows requires inheritable handles for its explicit child handle list.
    # Restore their flags on success and failure; serialise launches in this module.
    with _HANDLE_LOCK:
        previous = [os.get_handle_inheritable(handle) for handle in handles]
        try:
            for handle in handles:
                os.set_handle_inheritable(handle, True)
            return subprocess.Popen(command + [str(handle) for handle in handles],
                                    startupinfo=startup, **options)
        finally:
            for handle, flag in zip(handles, previous):
                os.set_handle_inheritable(handle, flag)


def _wait_ready(connection, process):
    """Bound startup time without socket polling or platform-specific pipe waits."""
    expired = threading.Event()

    def stop_startup():
        if process.poll() is None:
            expired.set()
            try:
                process.kill()
            except ProcessLookupError:
                pass

    timer = threading.Timer(_STARTUP_TIMEOUT, stop_startup)
    timer.daemon = True
    timer.start()
    try:
        try:
            message = _receive(connection)
        except (EOFError, OSError) as exc:
            if expired.is_set():
                raise RuntimeError('Timed out starting the assignment coordinator.') from exc
            raise RuntimeError(
                f'The assignment coordinator exited before starting. Python: {sys.executable}; '
                f'package: {Path(__file__).resolve().parent}. '
                'Check that this interpreter can import the installed dependencies.'
            ) from exc
        if expired.is_set():
            raise RuntimeError('Timed out starting the assignment coordinator.')
        if message != ('ready',):
            raise RuntimeError('Invalid assignment coordinator startup message.')
    finally:
        timer.cancel()
        timer.join()


@contextmanager
def _coordinator():
    """Start an isolated Python entry point over two inherited anonymous pipes."""
    if getattr(sys, 'frozen', False):
        raise RuntimeError(
            'Parallel assignments require a regular Python interpreter. '
            'Use n_jobs=1 in a frozen executable.'
        )
    environment = os.environ.copy()
    paths = _python_path()
    environment['PYTHONPATH'] = os.pathsep.join(paths)
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[name] = '1'
    with _pipe_pair() as (requests, request_writer), _pipe_pair() as (results, result_writer):
        process = None
        ready = False
        connection = _PipeConnection(results, request_writer)
        try:
            process = _launch_coordinator(requests.fileno(), result_writer.fileno(), paths, environment)
            # Close the child's copies in the parent so closure is detectable.
            requests.close()
            result_writer.close()
            _wait_ready(connection, process)
            ready = True
            yield connection, process
        finally:
            # Pipe closure cancels pending work on early iterator exit.
            connection.close()
            if process is not None:
                if not ready and process.poll() is None:
                    process.terminate()
                # Match executor shutdown: cancel queued work, join running jobs.
                process.wait()


def pool_results(initial, iterator, state, workers, start_method, stats):
    """Yield ordered results with at most two queued assignments per worker.

    Choice labels stay in the caller. Only scale tuples, the shared model, and
    exact result objects cross the private pipes. Worker errors include their
    original traceback and prevent selection of a partial minimum.
    """
    pending = deque(initial)
    next_index = len(initial)
    expected_index = 0
    started = time.perf_counter()
    try:
        with _coordinator() as (connection, process):
            stats.update(coordinator_pid=process.pid,
                         worker_bootstrap='isolated',
                         transport=TRANSPORT,
                         max_in_flight=len(pending))
            _send(connection, ('start', state, workers, start_method,
                               [(i, scales) for i, (_, scales) in enumerate(initial)]))
            while pending:
                message = _receive(connection)
                if message[0] == 'error':
                    raise RuntimeError(message[1])
                if message[0] != 'result' or message[1] != expected_index:
                    raise RuntimeError('Invalid assignment result order from the coordinator.')
                _, _, result, data = message
                choice, _ = pending.popleft()
                expected_index += 1
                following = next(iterator, None)
                if following is not None:
                    pending.append(following)
                    _send(connection, ('next', (next_index, following[1])))
                    next_index += 1
                else:
                    _send(connection, ('next', None))
                yield choice, result, data
            message = _receive(connection)
            if message[0] == 'error':
                raise RuntimeError(message[1])
            if message != ('done',):
                raise RuntimeError('The assignment coordinator did not finish normally.')
        if process.returncode != 0:
            raise RuntimeError(f'The assignment coordinator exited with code {process.returncode}.')
    except (EOFError, ConnectionError) as exc:
        raise RuntimeError(
            'The assignment coordinator disconnected before returning all results; '
            'no minimum is returned. Check the subprocess error output.'
        ) from exc
    finally:
        stats['pool_wall_seconds'] = time.perf_counter() - started


def _serve(connection):
    """Create the sole executor in the isolated interpreter, then stream results."""
    from ._parallel_assignments import _evaluate_worker, _init_worker
    message = _receive(connection)
    if len(message) != 5 or message[0] != 'start':
        raise RuntimeError('Invalid assignment coordinator initialization.')
    _, state, workers, start_method, initial = message
    context = mp.get_context(start_method)
    pending = deque()
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                   initializer=_init_worker, initargs=(state,))
    try:
        for index, scales in initial:
            pending.append((index, executor.submit(_evaluate_worker, (index, scales))))
        while pending:
            index, future = pending.popleft()
            result, data = future.result()
            _send(connection, ('result', index, result, data))
            message = _receive(connection)
            if len(message) != 2 or message[0] != 'next':
                raise RuntimeError('Invalid assignment coordinator continuation.')
            if message[1] is not None:
                index, scales = message[1]
                pending.append((index, executor.submit(_evaluate_worker, (index, scales))))
    finally:
        for _, future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
    _send(connection, ('done',))


def _inherited_stream(value, mode):
    """Convert one inherited descriptor/Windows handle to a binary stream."""
    descriptor = int(value)
    if _WINDOWS:
        import msvcrt
        flags = os.O_BINARY | (os.O_RDONLY if mode == 'rb' else os.O_WRONLY)
        descriptor = msvcrt.open_osfhandle(descriptor, flags)
    try:
        os.set_inheritable(descriptor, False)
        return os.fdopen(descriptor, mode, buffering=0)
    except BaseException:
        os.close(descriptor)
        raise


def main():
    """Run the internal coordinator; not a user-facing command-line interface."""
    mp.freeze_support()
    if len(sys.argv) != 3:
        raise RuntimeError('The assignment coordinator requires its two inherited pipe ends.')
    with _inherited_stream(sys.argv[1], 'rb') as reader, _inherited_stream(sys.argv[2], 'wb') as writer:
        connection = _PipeConnection(reader, writer)
        if hasattr(os, 'register_at_fork'):
            # A fork worker must not keep the parent/coordinator protocol alive.
            os.register_at_fork(after_in_child=connection.close)
        try:
            _send(connection, ('ready',))
            _serve(connection)
        except (EOFError, ConnectionError):
            return  # The caller closed the iterator or exited.
        except Exception:
            message = 'Assignment process failure; no minimum is returned.\n' + traceback.format_exc()
            _send(connection, ('error', message))


if __name__ == '__main__':
    main()
