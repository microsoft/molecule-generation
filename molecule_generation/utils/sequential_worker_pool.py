from multiprocessing.pool import Pool


class ImmediateResult:
    def __init__(self, result):
        self._result = result

    def ready(self):
        return True

    def successful(self):
        return True

    def wait(self, timeout=None):
        return

    def get(self, timeout=None):
        return self._result


class SequentialWorkerPool(Pool):
    """Stub class implementing the multiprocessing.Pool interface, but without
    separate processes, to facilitate easier interactive debugging."""

    def __init__(self):
        pass  # We have no state

    def close(self):
        pass

    def join(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def apply(self, func, args=(), kwds={}):
        return func(*args, **kwds)

    def apply_async(self, func, args=(), kwds={}):
        return ImmediateResult(func(*args, **kwds))

    def map(self, func, iterable, chunksize=None):
        return [func(arg) for arg in iterable]

    def map_async(self, func, iterable, chunksize=None):
        return ImmediateResult([func(arg) for arg in iterable])

    def starmap(self, func, iterable, chunksize=None):
        return [func(*args) for args in iterable]

    def starmap_async(self, func, iterable, chunksize=None):
        return ImmediateResult([func(*args) for args in iterable])

    def imap(self, func, iterable, chunksize=1):
        for arg in iterable:
            yield func(arg)

    def imap_unordered(self, func, iterable, chunksize=1):
        for arg in iterable:
            yield func(arg)


def get_worker_pool(num_processes: int):
    if num_processes <= 1:
        return SequentialWorkerPool()
    else:
        return Pool(num_processes)
