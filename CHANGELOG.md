# Changelog

## [0.2.0] - 2026-10-25

### Added

- Local quantized FLUX 3 generation and reference editing, with FLUX 1,
  masked Fill, LoRA adapters, and small CPU segmentation models.
  ([#1](https://github.com/affromero/Photo2Panning/pull/1))
- Local LTX video generation with first and last image guides, filtered
  pinned checkpoints, and transformer streaming for Apple Silicon.
  ([#1](https://github.com/affromero/Photo2Panning/pull/1))
- Ordered picture edits, explicit export quality, and JSON jobs with
  generated schemas and documented fields for internal agents.
  ([#1](https://github.com/affromero/Photo2Panning/pull/1))
- Sampled shutter exposure for photo motion, with linear-light blending
  to reduce temporal aliasing while retaining bright detail.
  ([#1](https://github.com/affromero/Photo2Panning/pull/1))
- Runnable public photo demos, an agent tool guide, and a soundtrack
  example that preserves the encoded video without recompression.
  ([#1](https://github.com/affromero/Photo2Panning/pull/1))

### Fixed

- CLI dispatch, aspect-preserving motion, video timing, source audio,
  independent title-card timing, and frame cleanup after failures.
  ([#1](https://github.com/affromero/Photo2Panning/pull/1))
- Transparent image edits, input aliases, incomplete HTTP downloads,
  repeated audio offsets, GIF timing, and non-square video pixels.
  ([#1](https://github.com/affromero/Photo2Panning/pull/1))

### Changed

- Use uv, Python 3.14.9, and a locked Apple Silicon environment. Optional
  model dependencies load when their tools execute.
  ([#1](https://github.com/affromero/Photo2Panning/pull/1))
- Use tyro for the CLI, one validated processing API, streaming FFmpeg
  frames, strict mypy, ty, Ruff, and repository quality hooks.
  ([#1](https://github.com/affromero/Photo2Panning/pull/1))

[0.2.0]: https://github.com/affromero/Photo2Panning/releases/tag/v0.2.0
