# Changelog
All notable changes to the project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2022-10-18

### Added
- Support for fine-tuning a pretrained model on new data ([#30](https://github.com/microsoft/molecule-generation/pull/30))
- `__version__` attribute to make the package version easily accessible at runtime ([#35](https://github.com/microsoft/molecule-generation/pull/35))

### Changed
- Dropped the exact version requirements for `python` and `tensorflow` in order to support entire ranges of versions, including modern ones ([#35](https://github.com/microsoft/molecule-generation/pull/35))

### Removed
- Unused `GraphMultitaskModel` which ended up in the open-source release by accident ([#34](https://github.com/microsoft/molecule-generation/pull/34))

### Fixed
- Made the inference server check the status of child processes every 10 seconds, so that it can exit gracefully in case of errors instead of hanging ([#33](https://github.com/microsoft/molecule-generation/pull/33))

## [0.2.0] - 2022-07-01

### Added
- `MoLeRGenerator`, which uses the MoLeR decoder (without the encoder) as an autoregressive policy ([#6](https://github.com/microsoft/molecule-generation/pull/6))
- `load_model_from_directory`, which can load any model by automatically picking the right wrapper class (either `VaeWrapper` or `GeneratorWrapper`) ([#24](https://github.com/microsoft/molecule-generation/pull/24))
- An option for `encode` to return not only the mean latent code but also the variance ([#26](https://github.com/microsoft/molecule-generation/pull/26))

### Changed
- Improved how the MoLeR visualisers handle node selection steps ([#10](https://github.com/microsoft/molecule-generation/pull/10))
- Refactored how MoLeR keeps track of generation steps during decoding and included partial molecules in the step info classes ([#27](https://github.com/microsoft/molecule-generation/pull/27))

### Fixed
- One-off errors in the latent-based visualisation mode ([#10](https://github.com/microsoft/molecule-generation/pull/10))
- Constrained the version of `protobuf` to avoid pulling in a breaking release ([#25](https://github.com/microsoft/molecule-generation/pull/25))

## [0.1.0] - 2022-04-11

This is the first public release, matching what was used for [the original paper](https://arxiv.org/abs/2103.03864).

### Added
- Full implementation of MoLeR as introduced in the paper
- Reference implementation of CGVAE, not yet supported by the high-level model API

[Unreleased]: https://github.com/microsoft/molecule-generation/compare/v0.3.0...HEAD
[0.1.0]: https://github.com/microsoft/molecule-generation/releases/tag/v0.1.0
[0.2.0]: https://github.com/microsoft/molecule-generation/releases/tag/v0.2.0
[0.3.0]: https://github.com/microsoft/molecule-generation/releases/tag/v0.3.0
