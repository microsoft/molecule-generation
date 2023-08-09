# MoLeR: A Model for Molecule Generation

[![CI](https://github.com/microsoft/molecule-generation/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/microsoft/molecule-generation/actions/workflows/ci.yml)
[![license](https://img.shields.io/github/license/microsoft/molecule-generation.svg)](https://github.com/microsoft/molecule-generation/blob/main/LICENSE)
[![pypi](https://img.shields.io/pypi/v/molecule-generation.svg)](https://pypi.org/project/molecule-generation/)
[![python](https://img.shields.io/pypi/pyversions/molecule_generation)](https://www.python.org/downloads/)
[![code style](https://img.shields.io/badge/code%20style-black-202020.svg)](https://github.com/ambv/black)

This repository contains training and inference code for the MoLeR model introduced in [Learning to Extend Molecular Scaffolds with Structural Motifs](https://arxiv.org/abs/2103.03864). We also include our implementation of CGVAE, but without integration with the high-level model interface.

## Quick start

`molecule_generation` can be installed via `pip`, but it additionally depends on `rdkit` and (if one wants to use a GPU) on setting up CUDA libraries. One can get both through `conda`:

```bash
conda env create -f environment.yml
conda activate moler-env
```

Our package was tested with `python>=3.7`, `tensorflow>=2.1.0` and `rdkit>=2020.09.1`; see the `environment*.yml` files for the exact configurations tested in CI.

To then install the latest release of `molecule_generation`, run
```bash
pip install molecule-generation
```

Alternatively, `pip install -e .` within the root folder installs the latest state of the code, including changes that were merged into `main` but not yet released.

A MoLeR checkpoint trained using the default hyperparameters is available [here](https://figshare.com/ndownloader/files/34642724). This file needs to be saved in a fresh folder `MODEL_DIR` (e.g., `/tmp/MoLeR_checkpoint`) and be renamed to have the `.pkl` ending (e.g., to `GNN_Edge_MLP_MoLeR__2022-02-24_07-16-23_best.pkl`). Then you can sample 10 molecules by running

```bash
molecule_generation sample MODEL_DIR 10
```

See below for how to train your own model and run more advanced inference.

### Troubleshooting

> Q: Installing `tensorflow` on my system does not work, or it works but GPU is not being used.
>
> A: Please refer to [the tensorflow website](https://www.tensorflow.org/install) for guidelines. In particular, with recent versions of `tensorflow` one may get a "libdevice not found" error; in that case please follow the instructions at the bottom of [this page](https://www.tensorflow.org/install/pip#step-by-step_instructions).

> Q: My particular combination of dependency versions does not work.
>
> A: Please submit an issue and default to using one of the pinned configurations from `environment-py*.yml` in the meantime.

> Q: I am in China and so the figshare checkpoint link does not work for me.
>
> A: You can try [this link](https://pan.baidu.com/s/1lkiWK9-d5MvNyzqRrusGXA?pwd=4hij) instead.

## Workflow

Working with MoLeR can be roughly divided into four stages:
- *data preprocessing*, where a plain text list of SMILES strings is turned into `*.pkl` files containing descriptions of the molecular graphs and generation traces;
- *training*, where MoLeR is trained on the preprocessed data until convergence;
- *inference*, where one loads the model and performs batched encoding, decoding or sampling; and (optionally)
- *fine-tuning*, where a previously trained model is fine-tuned on new data.

Additionally, you can visualise the decoding traces and internal action probabilities of the model, which can be useful for debugging.

### Data Preprocessing

To run preprocessing, your data has to follow a simple GuacaMol format (files `train.smiles`, `valid.smiles` and `test.smiles`, each containing SMILES strings, one per line). Then, you can preprocess the data by running

```
molecule_generation preprocess INPUT_DIR OUTPUT_DIR TRACE_DIR
```

where `INPUT_DIR` is the directory containing the three `*.smiles` files, `OUTPUT_DIR` is used for intermediate results, and `TRACE_DIR` for final preprocessed files containing the generation traces. Additionally, the `preprocess` command accepts command-line arguments to override various preprocessing hyperparameters (notably, the size of the motif vocabulary).
This step roughly corresponds to applying Algorithm 2 from our paper to each molecule in the input data.

After running the above, you should see an output similar to

```
2022-03-10 11:22:15,927 preprocess.py:239 INFO 1273104 train datapoints, 79568 validation datapoints, 238706 test datapoints loaded, beginning featurization.
2022-03-10 11:22:15,927 preprocess.py:245 INFO Featurising data...
2022-03-10 11:22:15,927 molecule_dataset_utils.py:261 INFO Turning smiles into mol
2022-03-10 11:22:15,927 molecule_dataset_utils.py:79 INFO Initialising feature extractors and motif vocabulary.
2022-03-10 11:44:17,864 motif_utils.py:158 INFO Motifs in total: 99751
2022-03-10 11:44:25,755 motif_utils.py:182 INFO Removing motifs with less than 3 atoms
2022-03-10 11:44:25,755 motif_utils.py:183 INFO Motifs remaining: 99653
2022-03-10 11:44:25,764 motif_utils.py:190 INFO Truncating the list of motifs to 128 most common
2022-03-10 11:44:25,764 motif_utils.py:192 INFO Motifs remaining: 128
2022-03-10 11:44:25,764 motif_utils.py:199 INFO Finished creating the motif vocabulary
2022-03-10 11:44:25,764 motif_utils.py:200 INFO | Number of motifs: 128
2022-03-10 11:44:25,764 motif_utils.py:203 INFO | Min frequency: 3602
2022-03-10 11:44:25,764 motif_utils.py:204 INFO | Max frequency: 1338327
2022-03-10 11:44:25,764 motif_utils.py:205 INFO | Min num atoms: 3
2022-03-10 11:44:25,764 motif_utils.py:206 INFO | Max num atoms: 10
2022-03-10 11:44:25,862 preprocess.py:255 INFO Completed initializing feature extractors; featurising and saving data now.
 Wrote 1273104 datapoints to /guacamol/output/train.jsonl.gz.
 Wrote 79568 datapoints to /guacamol/output/valid.jsonl.gz.
 Wrote 238706 datapoints to /guacamol/output/test.jsonl.gz.
 Wrote metadata to /guacamol/output/metadata.pkl.gz.
(...proceeds to compute generation traces...)
```

After the preprocessed graphs are saved into `OUTPUT_DIR`, they will be turned into concrete generation traces, which is typically the most compute-intensive part of preprocessing. During that part, the preprocessing code may print errors, noting molecules that could not have been parsed or failed other assertions; MoLeR's preprocessing is robust to such cases, and will simply skip any problematic samples.

### Training

Having stored some preprocessed data under `TRACE_DIR`, MoLeR can be trained by running

```
molecule_generation train MoLeR TRACE_DIR
```


The `train` command accepts many command-line arguments to override training and architectural hyperparameters, most of which are accessed through passing `--model-params-override`. For example, the following trains a MoLeR model using `GGNN`-style message passing (instead of the default `GNN_Edge_MLP`) and using fewer layers in both the encoder and the decoder GNNs:

```
molecule_generation train MoLeR TRACE_DIR \
    --model GGNN \
    --model-params-override '{"gnn_num_layers": 6, "decoder_gnn_num_layers": 6}'
```

As [tf2-gnn](https://github.com/microsoft/tf2-gnn) is highly flexible, MoLeR supports a vast space of architectural configurations.

After running `molecule_generation train`, you should see an output similar to

```
(...tensorflow messages, hyperparameter dump...)
Initial valid metric:
Avg weighted sum. of graph losses:  122.1728
Avg weighted sum. of prop losses:   0.4712
Avg node class. loss:                 35.9361
Avg first node class. loss:           27.4681
Avg edge selection loss:              1.7522
Avg edge type loss:                   3.8963
Avg attachment point selection loss:  1.1227
Avg KL divergence:                    7335960.5000
Property results: sa_score: MAE 11.23, MSE 1416.26 (norm MAE: 13.89) | clogp: MAE 10.87, MSE 4620.69 (norm MAE: 5.98) | mol_weight: MAE 407.42, MSE 185524.38 (norm MAE: 3.70).
   (Stored model metadata and weights to trained_model/GNN_Edge_MLP_MoLeR__2022-03-01_18-15-14_best.pkl).
(...training proceeds...)
```

By default, training proceeds until there is no improvement in validation loss for 3 consecutive mini-epochs, where a mini-epoch is defined as 5000 training steps; this can be controlled through the `--patience` flag and the `num_train_steps_between_valid` model parameter, respectively.

### Inference

After a model has been trained and saved under `MODEL_DIR`, we provide two ways to load it: from CLI or directly from Python.
Currently, CLI-based loading does not expose all useful functionalities, and is mostly meant for simple tests.

To sample molecules from the model using the CLI, simply run

```
molecule_generation sample MODEL_DIR NUM_SAMPLES
```

and, similarly, to encode a list of SMILES stored under `SMILES_PATH` into latent vectors, and store them under `OUTPUT_PATH`

```
molecule_generation encode MODEL_DIR SMILES_PATH OUTPUT_PATH
```

In all cases `MODEL_DIR` denotes the directory containing the model checkpoint, not the path to the checkpoint itself.
The model loader will only look at `*.pkl` files under `MODEL_DIR`, and expect there is _exactly one_ such file, corresponding to the trained checkpoint.

You can load a model directly from Python via

```python
from molecule_generation import load_model_from_directory

model_dir = "./example_model_directory"
example_smiles = ["c1ccccc1", "CNC=O"]

with load_model_from_directory(model_dir) as model:
    embeddings = model.encode(example_smiles)
    print(f"Embedding shape: {embeddings[0].shape}")

    # Decode without a scaffold constraint.
    decoded = model.decode(embeddings)

    # The i-th scaffold will be used when decoding the i-th latent vector.
    decoded_scaffolds = model.decode(embeddings, scaffolds=["CN", "CCC"])

    print(f"Encoded: {example_smiles}")
    print(f"Decoded: {decoded}")
    print(f"Decoded with scaffolds: {decoded_scaffolds}")
```

which should yield an output similar to

```
Embedding shape: (512,)
Encoded: ['c1ccccc1', 'CNC=O']
Decoded: ['C1=CC=CC=C1', 'CNC=O']
Decoded with scaffolds: ['C1=CC=C(CNC2=CC=CC=C2)C=C1', 'CNC(=O)C(C)C']
```

As shown above, MoLeR is loaded through a context manager.
Behind the scenes, the following things happen:
- First, an appropriate wrapper class is chosen: if the provided directory contains a `MoLeRVae` checkpoint, the returned wrapper will support `encode`, `decode` and `sample`, while `MoLeRGenerator` will only support `sample`.
- Next, parallel workers are spawned, which await queries for encoding/decoding; these processes continue to live as long as the context is active.
The degree of paralellism can be configured using a `num_workers` argument.

### Fine-tuning

Fine-tuning proceeds similarly to training from scratch, with a few adjustments.
First, data intended for fine-tuning has to be preprocessed accordingly, by running

```
molecule_generation preprocess INPUT_DIR OUTPUT_DIR TRACE_DIR \
    --pretrained-model-path CHECKPOINT_PATH
```

Where `CHECKPOINT_PATH` points to the file (not directory) corresponding to the model that will later be fine-tuned.

The `--pretrained-model-path` argument is necessary, as otherwise preprocessing would infer various metadata (e.g. set of atom/motif types) solely from the provided set of SMILES, whereas for fine-tuning this has to be aligned with the metadata that the model was originally trained with.

After preprocessing, fine-tuning is run as
```
molecule_generation train MoLeR TRACE_DIR \
    --load-saved-model CHECKPOINT_PATH \
    --load-weights-only
```

When fine-tuning on a small dataset, it may not be desirable to update the model until convergence.
Training duration can be capped by passing `--model-params-override '{"num_train_steps_between_valid": 100}'` (to shorten the mini-epochs) and `--max-epochs` (to limit the number of mini-epochs).

### Visualisation

We support two subtly different modes of visualisation: decoding a given latent vector, and decoding a latent vector created by encoding a given SMILES string. In the former case, the decoder runs as normal during inference; in the latter case we know the ground-truth input, so we teacher-force the correct decoding decisions.

To enter the visualiser, run either

```
molecule_generation visualise cli MODEL_DIR SMILES_OR_SAMPLES_PATH
```

to get the result printed as plain text in the CLI, or

```
molecule_generation visualise html MODEL_DIR SMILES_OR_SAMPLES_PATH OUTPUT_DIR
```

to get the result saved under `OUTPUT_DIR` as a static HTML webpage.

## Code Structure

All of our models are implemented in [Tensorflow 2](https://www.tensorflow.org/), and are meant to be easy to extend and build upon. We use [tf2-gnn](https://github.com/microsoft/tf2-gnn) for the core Graph Neural Network components.

The MoLeR model itself is implemented as a `MoLeRVae` class, inheriting from `GraphTaskModel` in `tf2-gnn`; that base class encapsulates the encoder GNN. The decoder GNN is instantiated as an external `MoLeRDecoder` layer; it also includes batched inference code, which forces the maximum likelihood choice at every step.

## Authors

* [Krzysztof Maziarz](mailto:krzysztof.maziarz@microsoft.com)
* [Henry Jackson-Flux](mailto:hrjackson@gmail.com)
* [Marc Brockschmidt](mailto:mabrocks@microsoft.com)
* [Pashmina Cameron](mailto:Pashmina.Cameron@microsoft.com)
* [Sarah Lewis](mailto:sarahlewis@microsoft.com)
* [Marwin Segler](mailto:marwinsegler@microsoft.com)
* [Megan Stanley](mailto:meganstanley@microsoft.com)
* [Paweł Czyż](mailto:pawelpiotr.czyz@ai.ethz.ch)
* [Ashok Thillaisundaram](mailto:ashok@cantab.net)

_Note: as git history was truncated at the point of open-sourcing, GitHub's statistics do not reflect the degree of contribution from some of the authors. All listed above had an impact on the code, and are (approximately) ordered by decreasing contribution._

The code is maintained by the [Generative Chemistry](https://www.microsoft.com/en-us/research/project/generative-chemistry/)
group at Microsoft Research, Cambridge, UK.
We are [hiring](https://www.microsoft.com/en-us/research/project/generative-chemistry/opportunities/).

MoLeR was created as part of our collaboration with Novartis Research. In particular, its design was guided by [Nadine Schneider](mailto:nadine-1.schneider@novartis.com), [Finton Sirockin](mailto:finton.sirockin@novartis.com), [Nikolaus Stiefl](mailto:nikolaus.stiefl@novartis.com), as well as others from Novartis.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

### Style Guide

- For code style, use [black](https://pypi.org/project/black/) and [flake8](https://pypi.org/project/flake8/).
- For commit messages, use imperative style and follow the [semmantic commit messages](https://gist.github.com/joshbuchea/6f47e86d2510bce28f8e7f42ae84c716) template; e.g.
    > feat(moler_decoder): Improve masking of invalid actions

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
