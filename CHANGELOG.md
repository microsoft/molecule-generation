# Changelog
All notable changes to the project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `MoLeRGenerator`, which uses the MoLeR decoder (without the encoder) as an autoregressive policy ([#6](https://github.com/microsoft/molecule-generation/pull/6))

### Changed
- Improved how the MoLeR visualisers handle node selection steps and fixed the "visualise from latents" mode ([#10](https://github.com/microsoft/molecule-generation/pull/10))

### Fixed
- Constrained the version of `protobuf` to avoid pulling in a breaking release ([#25](https://github.com/microsoft/molecule-generation/pull/25))

## [0.1.0] - 2022-04-11

This is the first public release, matching what was used for [the original paper](https://arxiv.org/abs/2103.03864).

### Added
- Full implementation of MoLeR as introduced in the paper
- Reference implementation of CGVAE, not yet supported by the high-level model API

[Unreleased]: https://github.com/microsoft/molecule-generation/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/microsoft/molecule-generation/releases/tag/v0.1.0
