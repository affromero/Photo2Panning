# Photo2Panning contributor instructions

Photo2Panning turns images into motion videos and provides image editing,
video processing, and local model tools. Preserve each capability when
refactoring. This is a personal project with one current API; document that
API directly.

## Environment

- Target Apple Silicon macOS and Python 3.14.9.
- Use `uv sync` and `uv lock` for dependencies. Commit the lockfile.
- Keep model dependencies in the optional `models` extra and YouTube
  support in the optional `youtube` extra.
- Keep local generative video dependencies in the optional `video` extra.
- Keep scene reconstruction and native rendering in the optional `camera`
  extra.
- Core imports, CLI help, schema discovery, and validation must work without
  model packages or downloaded weights.
- Use MLX quantized generation models and small CPU models for
  segmentation. Surface model and device failures.

## Structure and APIs

- `pic2panning/schemas/` defines validated request and result models.
- `pic2panning/tools/` owns processing and filesystem operations.
- `pic2panning/backends/` adapts external model libraries.
- `pic2panning/api.py` exposes the library functions.
- `pic2panning/cli.py` maps commands to the same functions.
- `pic2panning/demos/` contains runnable examples using the public CLI.
- Keep tests under root `tests/`.

Use existing functions before introducing new helpers. Keep processing out
of CLI handlers and demos. Generate the JSON schema from runtime models.
Update examples and the tool guide when request semantics change.

## Python rules

- Use the project's Pydantic base, with explicit fields and validation.
- Fully annotate functions. Use absolute imports.
- Use tyro for CLI interfaces. Keep CLI command models typed and separate
  from processing functions.
- Use `klogr` for logging and path helpers.
- Keep 79-character formatting, at most 1000 lines per source file, and at
  most 10 Python or shell files per directory.
- Declare instance attributes at class scope.
- Array and tensor signatures require shaped jaxtyping annotations and
  `@jaxtyped(typechecker=beartype)`.
- Keep shape strings quoted. Preserve the Ruff `UP037` protection.
- Avoid `Any`, standard dataclasses, type or lint suppressions, runtime
  import-path edits, inline imports, and unfinished markers.
- Keep model-library typing at explicit external boundaries.

## Media and output safety

Stream frames and keep memory bounded as video length grows. Preserve
aspect ratio when reframing. Width precedes height in every size.

Validate request options before work begins. Protect source files,
including symlink and hardlink aliases. Write outputs through the shared
atomic output helper and require explicit overwrite for existing files.
Clean up temporary files and external processes after failures.

Keep audio, title-card, speed, and reverse timing explicit. Never change a
backend, model, device, or user option silently after failure.

## Verification

```bash
uv sync --frozen --extra camera
uv run --extra camera pre-commit install
uv run --extra camera pre-commit run --all-files
uv run --extra camera mypy
uv run --extra camera pytest
uv build
```

Run affected tests after source changes. Test output content and failure
behavior. Mock only filesystem, network, subprocess, and model boundaries.
Do not mock pure helpers or assert internal call counts. Inspect generated
images or video frames when changing visual behavior.

Use the repository hooks locally. Fix hook failures without skipping them.
Before committing, review `git diff --cached` and check for debug logging.
Keep documentation concise and specific, with no em dashes.
