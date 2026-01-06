# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-12-26

### Added
- Initial release
- `FlattenedModel` class for parameter flattening/unflattening
- `Mask` dataclass for efficient binary mask representation
- `MaskSelector` abstract base class with implementations:
  - `RandomMaskSelector`: Per-round random selection
  - `TopKMagnitudeSelector`: Magnitude-based selection
  - `GradientBasedSelector`: Gradient-magnitude selection
  - `StructuredMaskSelector`: Layer-aware selection
  - `FixedMaskSelector`: User-defined indices
  - `ClientSpecificMaskSelector`: Per-client different masks
- Flower integration:
  - `PSFedAvg` strategy
  - `PSFedClient` base class
- Examples:
  - MNIST basic example
  - CIFAR-10 advanced example
- Comprehensive test suite
- Documentation

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A
