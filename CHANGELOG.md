# Changelog

All notable changes to the project are documented in this file.

The format follows [Common Changelog](https://common-changelog.org/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.1] - 2024-01-03

### Changed
- Relax `protobuf` version requirement ([#62](https://github.com/microsoft/molecule-generation/pull/62))

### Fixed
- Fix hydrogen handling in scaffolds with explicit attachment points ([#70](https://github.com/microsoft/molecule-generation/pull/70))
- Avoid memory leaks and other `tensorflow` issues ([#68](https://github.com/microsoft/molecule-generation/pull/68))

## [0.4.0] - 2023-06-16

### Added
- Add an option for `decode` to return the entire generation trace ([#51](https://github.com/microsoft/molecule-generation/pull/51))

### Changed
- Reformat with `black==23.1.0` and pin it in CI to avoid further unexpected updates ([#50](https://github.com/microsoft/molecule-generation/pull/50))

### Fixed
- Remove deprecated `numpy` types to make `molecule_generation` work with `numpy>=1.24.0` ([#49](https://github.com/microsoft/molecule-generation/pull/49))
- Patch `GetSSSR` for compatibility with `rdkit>=2022.09.1` ([#58](https://github.com/microsoft/molecule-generation/pull/58))

## [0.3.0] - 2022-10-18

### Added
- Add support for fine-tuning a pretrained model on new data ([#30](https://github.com/microsoft/molecule-generation/pull/30))
- Add a `__version__` attribute to make the package version easily accessible at runtime ([#35](https://github.com/microsoft/molecule-generation/pull/35))

### Changed
- Drop the exact version requirements for `python` and `tensorflow` to support entire ranges of versions ([#35](https://github.com/microsoft/molecule-generation/pull/35))

### Removed
- Drop unused `GraphMultitaskModel` which ended up in the open-source release by accident ([#34](https://github.com/microsoft/molecule-generation/pull/34))

### Fixed
- Make the inference server check the status of child processes every 10 seconds, so that it can exit gracefully in case of errors instead of hanging ([#33](https://github.com/microsoft/molecule-generation/pull/33))

## [0.2.0] - 2022-07-01

### Added
- Add `MoLeRGenerator`, which uses the MoLeR decoder (without the encoder) as an autoregressive policy ([#6](https://github.com/microsoft/molecule-generation/pull/6))
- Add `load_model_from_directory`, which can load any model by automatically picking the right wrapper class ([#24](https://github.com/microsoft/molecule-generation/pull/24))
- Implement an option for `encode` to return not only the mean latent code but also the variance ([#26](https://github.com/microsoft/molecule-generation/pull/26))

### Changed
- Improve how the MoLeR visualisers handle node selection steps ([#10](https://github.com/microsoft/molecule-generation/pull/10))
- Refactor how MoLeR keeps track of generation steps during decoding and include partial molecules in the step info classes ([#27](https://github.com/microsoft/molecule-generation/pull/27))

### Fixed
- Fix one-off errors in the latent-based visualisation mode ([#10](https://github.com/microsoft/molecule-generation/pull/10))
- Constrain `protobuf` version to avoid pulling in a breaking release ([#25](https://github.com/microsoft/molecule-generation/pull/25))

## [0.1.0] - 2022-04-11

:seedling: First public release, matching what was used for [the original paper](https://arxiv.org/abs/2103.03864).

### Added
- Add full implementation of MoLeR as introduced in the paper
- Add reference implementation of CGVAE, not yet supported by the high-level model API

[Unreleased]: https://github.com/microsoft/molecule-generation/compare/v0.4.1...HEAD
[0.1.0]: https://github.com/microsoft/molecule-generation/releases/tag/v0.1.0
[0.2.0]: https://github.com/microsoft/molecule-generation/releases/tag/v0.2.0
[0.3.0]: https://github.com/microsoft/molecule-generation/releases/tag/v0.3.0
[0.4.0]: https://github.com/microsoft/molecule-generation/releases/tag/v0.4.0
[0.4.1]: https://github.com/microsoft/molecule-generation/releases/tag/v0.4.1
