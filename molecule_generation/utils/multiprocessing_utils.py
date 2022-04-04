from multiprocessing import Process, Queue
from typing import Callable, TypeVar


ResultType = TypeVar("ResultType")


def _run_and_push_result(res_queue: Queue, fun: Callable[..., ResultType], *args) -> None:
    res_queue.put(fun(*args))


def run_in_separate_process(fun: Callable[..., ResultType], *args) -> ResultType:
    res_queue = Queue(1)
    p = Process(
        target=_run_and_push_result,
        args=(res_queue, fun, *args),
    )

    p.start()
    result = res_queue.get()
    p.join()

    return result
