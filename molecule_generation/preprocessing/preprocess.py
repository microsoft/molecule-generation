from collections import Counter
import logging
import os
from gzip import GzipFile
from itertools import chain
from typing import Any, Dict, List, Optional, Type

from dpu_utils.utils import RichPath
from tqdm import tqdm

from molecule_generation.preprocessing.graph_sample import GraphSample
from molecule_generation.chem.motif_utils import MotifVocabulary
from molecule_generation.chem.atom_feature_utils import AtomTypeFeatureExtractor
from molecule_generation.preprocessing.data_conversion_utils import (
    convert_graph_samples_to_traces,
    convert_jsonl_file_to_graph_samples,
)
from molecule_generation.preprocessing.generation_order import BFSOrder, GenerationOrder
from molecule_generation.utils.sequential_worker_pool import get_worker_pool, Pool

logger = logging.getLogger(__name__)


def preprocess_jsonl_files(
    jsonl_directory: RichPath,
    output_directory: RichPath,
    tie_fwd_bkwd_edges: bool,
    num_processes: int = 10,
    chunksize: int = 10,
    shardsize: int = 200,
    generation_order_cls: Type[GenerationOrder] = BFSOrder,
    MoLeR_style_trace: bool = True,
    quiet: bool = False,
):
    assert tie_fwd_bkwd_edges, "Separate forward and backward edges are not supported."

    # First copy the metadata, if needed.
    metadata_filename = "metadata.pkl.gz"
    output_metadata_path = output_directory.join(metadata_filename)
    if not output_metadata_path.exists():
        input_metadata_path = jsonl_directory.join(metadata_filename)
        assert (
            input_metadata_path.exists()
        ), f"Metadata file {metadata_filename} does not exist in {jsonl_directory}"
        output_metadata_path.copy_from(input_metadata_path)

    # Read the atom featurisers from the metadata.
    metadata = output_metadata_path.read_by_file_suffix()
    atom_featurisers = metadata["feature_extractors"]

    fold_names = ["train", "test", "valid"]
    input_paths = [jsonl_directory.join(f"{fold}.jsonl.gz") for fold in fold_names]
    output_paths = [
        output_directory.join(f"{fold}_{{shard_directory}}").join(
            f"{fold}_{{shard_counter}}.pkl.gz"
        )
        for fold in fold_names
    ]

    # Make sure that call does not re-calculate things if it does not need to.
    all_exist = all(output_path.exists() for output_path in output_paths)
    if all_exist:
        return

    # Make sure input exists.
    for input_path in input_paths:
        assert input_path.exists(), f"File not found at {input_path}"

    motif_vocabulary = metadata.get("motif_vocabulary")

    metadata["train_next_node_type_distribution"] = Counter()
    metadata["generation_order"] = generation_order_cls

    # Preprocess everything. Allow to update the metadata when processing the training datafold.
    for input_path, output_path, metadata_to_update in zip(
        input_paths, output_paths, [metadata, None, None]
    ):
        _preprocess_datafold(
            input_path,
            output_path,
            metadata_to_update,
            tie_fwd_bkwd_edges,
            atom_featurisers,
            motif_vocabulary,
            num_processes,
            chunksize,
            shardsize,
            generation_order_cls,
            MoLeR_style_trace,
            quiet=quiet,
        )

    # Save the (possibly updated) metadata.
    output_metadata_path.save_as_compressed_file(metadata)


def _preprocess_datafold(
    input_path: RichPath,
    output_path: RichPath,
    metadata: Optional[Dict[str, Any]],
    tie_fwd_bkwd_edges: bool,
    atom_featurisers: List[AtomTypeFeatureExtractor],
    motif_vocabulary: Optional[MotifVocabulary],
    num_processes: int,
    chunksize: int,
    shardsize: int,
    generation_order_cls: Type[GenerationOrder],
    MoLeR_style_trace: bool,
    quiet: bool = False,
):
    logger.info(f"Processing data from {input_path}")
    data_generator = input_path.read_by_file_suffix()

    # These next two are counters to use with tqdm in the _preprocess_subset function.
    def file_len(path: str):
        total = 0
        with GzipFile(path, "rb") as file:
            for _ in file:
                total += 1
            return total

    num_datapoints = file_len(input_path.path)
    counter = tqdm(total=num_datapoints, smoothing=0, disable=quiet)

    # We have to batch the data to stop out of memory errors. We run all of the preprocessing
    # on shards which have at most shardsize datapoints in them.
    data_remaining = True
    shard_counter = 0
    worker_pool = get_worker_pool(num_processes)
    while data_remaining:
        data: List[Dict[str, Any]] = []
        for i in range(shardsize):
            try:
                data.append(next(data_generator))
            except StopIteration:
                data_remaining = False
                break
        logger.debug("Successfully read data.")
        # Shards 0 - 999 will be stored in directory 0, 1000 - 1999 in directory 1000, etc.
        shard_directory = shard_counter - (shard_counter % 1000)
        _preprocess_subset(
            data,
            RichPath.create(
                output_path.path.format(
                    shard_directory=shard_directory, shard_counter=shard_counter
                )
            ),
            metadata,
            tie_fwd_bkwd_edges,
            atom_featurisers,
            motif_vocabulary,
            counter,
            num_processes,
            chunksize,
            worker_pool,
            shardsize,
            generation_order_cls,
            MoLeR_style_trace,
        )
        shard_counter += 1
    counter.close()
    worker_pool.close()
    worker_pool.join()


def _preprocess_subset(
    data: List[Dict[str, Any]],
    output_path: RichPath,
    metadata: Optional[Dict[str, Any]],
    tie_fwd_bkwd_edges: bool,
    atom_featurisers: List[AtomTypeFeatureExtractor],
    motif_vocabulary: Optional[MotifVocabulary],
    counter: tqdm,
    num_processes: int,
    chunksize: int,
    worker_pool: Pool,
    shardsize: int,
    generation_order_cls: Type[GenerationOrder],
    MoLeR_style_trace: bool,
):
    # Easy out when there's nothing to do.
    if len(data) == 0:
        return

    # Easy out when we've already processed this shard.
    if output_path.is_file():
        counter.update(shardsize)
        return

    graph_samples = convert_jsonl_file_to_graph_samples(data, num_processes, chunksize)
    logger.debug("Converted json dicts to graph samples.")
    # If graph samples is empty at this point, then something has gone wrong somewhere.
    first_graph: GraphSample = next(graph_samples)
    assert first_graph is not None, "Could not convert any graph samples."
    logger.debug("Converted at least one graph sample.")

    trace_samples = convert_graph_samples_to_traces(
        chain([first_graph], graph_samples),
        num_fwd_edge_types=first_graph.num_edge_types,
        atom_feature_extractors=atom_featurisers,
        motif_vocabulary=motif_vocabulary,
        tie_fwd_bkwd_edges=tie_fwd_bkwd_edges,
        add_self_loop_edges=False,  # Added on the fly in the dataset batching, if needed.
        num_processes=num_processes,
        chunksize=chunksize,
        MoLeR_style_trace=MoLeR_style_trace,
        generation_order_cls=generation_order_cls,
    )
    logger.debug("Converted to trace samples.")

    def _counter_update(x):
        counter.update()
        return x

    trace_samples = [_counter_update(trace_sample) for trace_sample in trace_samples]
    logger.debug("Materialised trace samples")

    if metadata is not None:
        # Update metadata based on the trace samples.
        for sample in trace_samples:
            for node_types in sample.correct_node_type_choices:
                if node_types is None:
                    continue

                if len(node_types) > 0:
                    metadata["train_next_node_type_distribution"].update(node_types)
                else:
                    # An empty list indicates the "no new nodes" prediction class.
                    metadata["train_next_node_type_distribution"]["None"] += 1

    # Make sure we are saving to a directory that exists.
    directory = os.path.split(output_path.path)[0]
    if not os.path.exists(directory):
        os.makedirs(directory)

    logger.debug(f"Saving to {output_path}")
    worker_pool.apply_async(output_path.save_as_compressed_file, [trace_samples])
