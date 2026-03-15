AGENTS
======

This file captures actionable build/lint/test commands and repository conventions so automated agents (and humans) can operate consistently in this repo.

- Repo root: `.`
- Primary language: Python (JAX/Flax + PyTorch mixed tooling)

Commands
--------
- Install (recommended): `poetry install` (uses `pyproject.toml`). If you don't use Poetry, create a venv and `pip install -r requirements.txt` if available.
- Run training (example from `.github/copilot-instructions.md`):

  ```bash
  conda activate jaxTorchTrans
  JAX_COMPILATION_CACHE_DIR="jax_cache" CUDA_VISIBLE_DEVICES="2" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 \
    python qwen_fineTuning_PtoC_fixCps.py --batch-size=4 --cp-fix 0 0 0
  ```

- Run all tests: `poetry run pytest` or `pytest` in an activated env.
- Run a single test file: `pytest -q src/test_subspace_curve.py` (or path to any test file).
- Run a single test function inside a file: `pytest -q src/test_subspace_curve.py::test_some_behavior`.
- Run tests by keyword: `pytest -q -k "keyword"`.
- Run with xdist parallelism: `pytest -n auto` (requires `pytest-xdist`).
- Run a single test with verbose/trace: `pytest -q -k "pattern" -s -vv`.
- Run coverage: `pytest --cov=src` (repo may require `pytest-cov`).

Linting / Formatting
--------------------
- Formatting: use `black`. Example: `poetry run black .` and to check: `poetry run black --check .`.
- Imports sorting: use `isort` or `ruff`'s isort integration. Example: `poetry run isort .`.
- Linting: `ruff` is preferred when available: `poetry run ruff check .` or `ruff check src tests`.
- Type checking: run `mypy` when present: `poetry run mypy src` (optional; repo has type hints in places).
- Pre-commit: if the repo has pre-commit set up, run `pre-commit run --all-files`.

Quick developer tips
--------------------
- Use `poetry run <cmd>` to ensure commands run inside the project's environment.
- When running code that compiles with JAX, set XLA/JAX env vars used by training scripts (see `.github/copilot-instructions.md`).
- Tests sometimes require GPU/JAX-capable environment for heavy code; many unit tests avoid device-heavy operations.

Code Style Guidelines (for agents)
---------------------------------
Follow these rules strictly to keep code readable and consistent.

- Formatting
  - Use `black` formatting defaults (line length 88). Run `black .` before committing.
  - Keep expressions simple; break long expressions onto subsequent indented lines.

- Imports
  - Order: stdlib, third-party, local (one blank line between groups).
  - Use absolute imports within the repository (e.g. `from src.module import Foo`) unless a local relative import is clearer for closely coupled modules.
  - Avoid wildcard imports (`from ... import *`).

- Typing & Types
  - Prefer explicit type annotations on function signatures for public functions and methods.
  - Use `Optional[T]` instead of `Union[T, None]` when appropriate.
  - Use `TypedDict` for structured dict-like configs if keys are stable.
  - Add a short docstring or inline comment when using complex typing constructs (e.g. pytree types used for JAX params).

- Naming Conventions
  - Modules / files: lowercase_with_underscores (e.g. `subspace_curve.py`).
  - Functions & variables: `snake_case`.
  - Classes & exceptions: `PascalCase`.
  - Constants: `UPPER_SNAKE_CASE` (module-level only).
  - Tests: `test_` prefix for function names and files (e.g. `test_*.py`).

- Functions & APIs
  - Small functions preferred; keep functions focused and short.
  - Public APIs should have an informative docstring describing arguments, return type, and side effects.
  - For code that runs under JAX/Flax, keep pure functions pure: avoid global state and side effects inside jitted functions.

- Error Handling
  - Prefer explicit exception types over `Exception` for raising and catching (e.g. `ValueError`, `TypeError`, custom exceptions inheriting from `Exception`).
  - Avoid bare `except:` or `except Exception:` unless you re-raise; always handle specific errors.
  - When re-raising, preserve context: `raise NewError(...) from exc`.
  - Use `logging` instead of prints for library code. Keep logs at appropriate levels (`debug`, `info`, `warning`, `error`, `critical`).

- Logging
  - Initialize loggers via `logger = logging.getLogger(__name__)` and do not configure global handlers in libraries—tests and entrypoints should configure handlers.

- JAX / Flax specifics
  - Keep JIT functions pure and side-effect free; pass mutable state explicitly.
  - Prefer `jax.lax.scan` for loops you want to avoid recompiling.
  - Keep dtype management explicit; document when using bfloat16/float32 mixed precision.
  - Use meaningful names for pytrees and masks; when converting/flattening, add a helper function and small docstring explaining the structure.

- LoRA / Masking Conventions (repo-specific)
  - Masks and LoRA parameter wrappers are represented as structured pytrees/dicts. When introducing new mask formats, add helper converters and tests.
  - Follow patterns from `src/subspace_curve.py` and `src/qwen_jax.py` for mask application and param conversions.

Testing Guidelines
------------------
- Test style
  - Use `pytest` fixtures for reusable setup.
  - Prefer small, deterministic unit tests. Mock out heavy device/GPU operations where possible.
  - Test names should describe the behavior tested: `test_function_does_x_given_y`.
  - Add a short docstring or comment for non-obvious tests.

- Adding tests
  - Place tests near the code under `tests/` or `src/` test files as the repository currently uses `src/test_subspace_curve.py`.
  - Include parameterized tests for multiple data shapes and corner cases.

- Running a single test
  - File: `pytest -q path/to/test_file.py`
  - Function: `pytest -q path/to/test_file.py::test_name`
  - Use `-k` to filter by expression: `pytest -q -k "substring"`.

Repository Conventions & Where to Look
------------------------------------
- Key files:
  - `src/subspace_curve.py` — core subspace math and LoRA mixing patterns.
  - `src/qwen_jax.py` — model implementation, weight mapping, (de)serialization helpers.
  - `qwen_fineTuning_PtoC_fixCps.py` — training orchestration and entry point used by sweep YAMLs.
  - `configs/*.yaml` — example configuration files for sweeps and runs.

- Serialization
  - Use `save_params_and_config` / `load_params_and_config` helpers where present; saved params use `jax_params.npy` and `jax_config.json` in training utilities.

Git / Agent Safety Rules
-----------------------
- Do not run destructive git commands (no `--force` push, no `reset --hard`) without explicit human approval.
- Avoid committing secrets: do not commit `.env`, API keys, or credential JSON files.
- When creating commits, follow existing commit style and do not amend user's existing commits unless explicitly asked.

Cursor / Copilot rules
----------------------
- Cursor rules: none found in repository (no `.cursor/rules/` or `.cursorrules`).
- Copilot rules: this repo contains `.github/copilot-instructions.md`. Include and follow those instructions — they contain high-level domain guidance and example commands used by the original author.

Embedded excerpt from `.github/copilot-instructions.md` (keep when making changes):

```
### Purpose
Time Transformer research framework: concise agent guidance for Bézier curve parameterized neural networks, LoRA fine-tuning, and experiment orchestration.

### Framework overview (core concepts)
- Subspace exploration: train curves (k+1 control points) in parameter space rather than single points to enforce mode connectivity.
- Uncertainty quantification: sample parameter configurations along curves to compute predictive distributions (mutual information, entropy, ECE).
- LoRA integration: low-rank adapters mixed with control-point subspaces for efficient fine-tuning of large models.

### Essential patterns (what to change and where)
- `SubspaceBaseModel` (src/subspace_curve.py): central pattern—models expose `nll`, `__call__`, `train_step`, and provide `init_mask`, `init_params_from_point` helpers. Follow its API when adding new task models.
- Wrapper classes: external PyTorch/JAX models are wrapped with a small adapter exposing `init` and `apply` compatible with `SubspaceBaseModel` (see wrappers in training scripts).

### Training & config patterns
- Entry point: `train()` in `qwen_fineTuning_PtoC_fixCps.py` — it loads datasets, prepares masks, initializes curve params (from point or random), creates optimizer schedules, and runs the curve training loop.

### Run / development workflows
- Typical environment: JAX/Flax + PyTorch (for weight conversion). The repo expects a GPU/JAX-capable environment; some configs set `XLA_FLAGS` to force host device counts.

Concretely (local dev):
  conda activate jaxTorchTrans
  JAX_COMPILATION_CACHE_DIR="jax_cache" CUDA_VISIBLE_DEVICES="2" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 python "qwen_fineTuning_PtoC_fixCps.py" --batch-size=4 --cp-fix 0 0 0
```

If you change model/LoRA/mask formats, update the helper converters and add unit tests. When in doubt, follow the patterns in `src/subspace_curve.py` and `src/qwen_jax.py`.

Next steps for agents
---------------------
1. Run formatting and linting: `poetry run black . && poetry run ruff check .`.
2. Run unit tests (single file or function) while iterating: `pytest -q path/to/test::test_name`.
3. Add tests for any new public helper or mask conversion you introduce.
