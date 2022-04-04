"""Iterator to read data shards in parallel using multiple processes."""

import os
import random
import time
import types
from itertools import cycle
from multiprocessing import Queue, Event, Process
from queue import Empty
from typing import Iterable, Iterator, List, TypeVar, Union
from typing_extensions import Literal

from dpu_utils.utils import RichPath

T = TypeVar("T")


class ShardedDataReader(Iterable[T]):
    """Reader of data stored in sharded files."""

    def __init__(
        self,
        shard_paths: List[RichPath],
        max_queue_size: int = 500,
        num_workers: int = 20,
        shuffle_data: bool = False,
        repeat_data: bool = False,
        no_parallelism: bool = False,
    ):
        self._shard_paths = shard_paths
        self._max_queue_size = max_queue_size
        self._num_workers = min(num_workers, len(shard_paths))
        self._shuffle_data = shuffle_data
        self._repeat_data = repeat_data
        self._no_parallelism = no_parallelism
        self._created_parallel_iterators: List[ParallelShardedDataIterator] = []

    def __iter__(
        self,
    ) -> Union["SequentialShardedDataIterator[T]", "ParallelShardedDataIterator[T]"]:
        if self._no_parallelism:
            return SequentialShardedDataIterator(self)
        else:
            reader = ParallelShardedDataIterator(self)
            self._created_parallel_iterators.append(reader)
            return reader

    def __enter__(self) -> "ShardedDataReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> Literal[False]:
        self.cleanup_resources()
        return False  # Signal that exceptions should be re-raised, if needed

    def cleanup_resources(self) -> None:
        for reader in self._created_parallel_iterators:
            reader.cleanup_workers()

    def get_example_datum(self) -> T:
        if len(self._shard_paths) == 0:
            raise ValueError("No shards passed to ParallelShardReader. No example datum available")
        path = self._shard_paths[0]
        data = path.read_by_file_suffix()
        datum = next(iter(data))
        return datum


class SequentialShardedDataIterator(Iterator[T]):
    def __init__(self, context_iterable: ShardedDataReader[T]):
        self._shard_paths = context_iterable._shard_paths
        self._shuffle_data = context_iterable._shuffle_data
        self._repeat_data = context_iterable._repeat_data
        self._current_file_data: List[T] = []
        self._next_path_idx = 0

    def __next__(self) -> T:
        if len(self._current_file_data) == 0:
            if self._next_path_idx >= len(self._shard_paths):
                if not self._repeat_data:
                    raise StopIteration
                self._next_path_idx = 0
                if self._shuffle_data:
                    random.shuffle(self._shard_paths)
            next_path = self._shard_paths[self._next_path_idx]
            self._next_path_idx += 1
            self._current_file_data = next_path.read_by_file_suffix()
            if self._shuffle_data:
                random.shuffle(self._current_file_data)
            else:
                self._current_file_data = list(self._current_file_data)
        next_datum = self._current_file_data.pop()
        return next_datum


class ParallelShardedDataIterator(Iterator[T]):
    def __init__(self, context_iterable: ShardedDataReader[T]):
        self._context_iterable = context_iterable
        self._shard_paths = context_iterable._shard_paths
        self._shuffle_data = context_iterable._shuffle_data
        self._repeat_data = context_iterable._repeat_data
        self._num_workers = context_iterable._num_workers

        self._processes: List[Process] = []
        self._shards_to_read_queues: List[Queue] = []
        self._output_queues: List[Queue] = []
        self._initialised_workers = False

    def initialise_workers(self):
        if self._initialised_workers:
            return

        # Terminate event to get set when __del__ function is called.
        self._termination_signal = Event()

        # Set up queues communicating shard paths to the workers/collecting results:
        num_shards = len(self._shard_paths)
        max_num_shards_per_worker = (
            2 + num_shards // self._num_workers
        )  # + 1 for rounding, + 1 for Empty
        self._shards_to_read_queues: List[Queue[str]] = [
            Queue(max_num_shards_per_worker) for _ in range(self._num_workers)
        ]
        self._output_queues: List[Queue[T]] = [
            Queue(self._context_iterable._max_queue_size) for _ in range(self._num_workers)
        ]

        # Set off the processes reading their shards.
        self._processes = [
            Process(
                target=read_shard,
                args=(
                    self._shards_to_read_queues[worker_id],
                    self._output_queues[worker_id],
                    worker_id,
                    self._shuffle_data,
                    self._repeat_data,
                    self._termination_signal,
                ),
            )
            for worker_id in range(self._num_workers)
        ]

        # Variables used to control the loop in the __next__ method.
        self._next_worker_idx = -1
        self._worker_is_finished_flags = [False] * len(self._processes)

        self.__populate_shards_to_read_queues()
        for worker in self._processes:
            worker.start()

        self._initialised_workers = True

    def __enter__(self) -> "ParallelShardedDataIterator":
        self.initialise_workers()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> Literal[False]:
        self.cleanup_workers()
        return False  # Signal that exceptions should be re-raised, if needed

    def __populate_shards_to_read_queues(self):
        # First, shuffle the shards.
        if self._shuffle_data:
            random.Random(0).shuffle(self._shard_paths)

        for worker_id, shard_path in zip(
            cycle(range(self._context_iterable._num_workers)), self._shard_paths
        ):
            self._shards_to_read_queues[worker_id].put(shard_path)
        # Put tombstone value at the end of each queue.
        for input_queues in self._shards_to_read_queues:
            input_queues.put(Empty)

    def cleanup_workers(self):
        # Do nothing if we are already cleaned up:
        if not self._initialised_workers:
            return

        # Set the terminate flag:
        self._termination_signal.set()

        # Empty the output queues to unblock the processes:
        for queue in self._output_queues:
            while not queue.empty():
                try:
                    queue.get_nowait()  # queue.empty is not totally reliable, so get_nowait can raise errors.
                except Empty:
                    pass

        # Just in case the process is blocking on getting from its input queue:
        for queue in self._shards_to_read_queues:
            queue.put(Empty)

        # Run over all workers, try to join them back:
        is_any_alive = False
        for worker in self._processes:
            worker.join(timeout=0.5)
            is_any_alive |= worker.is_alive()

        # The nuclear option to kill the stragglers:
        if is_any_alive:
            # Give them one more second to catch up:
            time.sleep(1)
            try:
                for worker in self._processes:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join(timeout=0.1)
            except ValueError as e:
                raise e

        # Clean up the processes. Try one last time to join the child process. If this
        # fails, .close() will raise an Exception, but we have no other recourse at this point.
        for worker in self._processes:
            worker.join(timeout=60)
            worker.close()

        # Also clean up the remaining queues now:
        delattr(self, "_termination_signal")
        for shard_queue in self._shards_to_read_queues:
            shard_queue.close()
        self._shards_to_read_queues = []
        for output_queue in self._output_queues:
            output_queue.close()
        self._output_queues = []

        self._initialised_workers = False

    def __del__(self):
        self.cleanup_workers()

    def __next__(self) -> T:
        if not self._initialised_workers:
            self.initialise_workers()

        while True:
            # Easy out if all of the data has already been read.
            if all(self._worker_is_finished_flags):
                if self._repeat_data:
                    self._worker_is_finished_flags = [False] * len(self._processes)
                    self.__populate_shards_to_read_queues()
                else:
                    break

            # Find which worker thread we should be polling next, going through the
            # list until we find one that is not finished yet (at least one exists, see above):
            self._next_worker_idx = (self._next_worker_idx + 1) % self._num_workers
            while self._worker_is_finished_flags[self._next_worker_idx]:
                self._next_worker_idx = (self._next_worker_idx + 1) % self._num_workers

            queue = self._output_queues[self._next_worker_idx]

            # We have to wait for the next element now, but regularly check that the child process
            # didn't die (due to programming error, OOM, ...)
            next_element = None
            while not next_element:
                try:
                    next_element = queue.get(timeout=5)
                except Empty:
                    worker_process = self._processes[self._next_worker_idx]
                    # Check if our dear child is still alive, if not, bail out:
                    if not worker_process.is_alive():
                        raise Exception(f"Worker {worker_process.pid} got killed; giving up.")

            # Check if the input queue is finished.
            if next_element is Empty:
                self._worker_is_finished_flags[self._next_worker_idx] = True
                continue
            # Check if we have some other error.
            if isinstance(next_element, tuple) and isinstance(next_element[0], Exception):
                raise next_element[0].with_traceback(next_element[1])

            # If everything is good, we have found our next element.
            return next_element

        # Reached if we break out of the while loop because we ran out of data:
        self.cleanup_workers()
        raise StopIteration


def read_shard(
    input_paths: Queue,
    output_queue: Queue,
    seed: int,
    shuffle_data: bool,
    repeat: bool,
    termination_signal: Event,
):
    """Read a list of shards and put the data contained in them in the output queue.
    When there is no data left, it will put an Empty object in the queue."""
    rng = random.Random(seed)
    try:
        while not termination_signal.is_set():
            path = input_paths.get()
            if path is Empty:
                output_queue.put(Empty)
                if repeat:
                    # Will block in the next loop on input_paths.get()
                    continue
                else:
                    break
            try:
                data = path.read_by_file_suffix()
            except Exception as e:
                print(
                    f"Process {os.getpid()} threw exception \n{e}\n"
                    f" when trying to read file {path}."
                )
                continue
            if shuffle_data:
                # shuffling requires data to be a list-like type:
                if isinstance(data, types.GeneratorType):
                    data = list(data)
                rng.shuffle(data)
            for datum in data:
                if not termination_signal.is_set():
                    output_queue.put(datum)
                else:
                    break
    except Exception as e:
        print(f"Process {os.getpid()} threw exception \n{e}")
        import traceback

        traceback.print_exc()
