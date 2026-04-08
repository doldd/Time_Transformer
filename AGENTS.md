**AGENTS**

This file gives concise, actionable build/lint/test commands and repository conventions for automated agents operating in this repository.

- Repo root: `.`
- Primary language: Python (JAX/Flax + PyTorch mixed tooling)

1) Safety mandates (READ FIRST)
--------------------------------
- **W&B READ-ONLY (MANDATORY):** never delete runs, update attributes, create reports, or write to the remote Weights & Biases account. Treat all wandb access as strictly read-only.
- **Environment Stability:** NEVER change, install, or update python packages (no `pip install`, `poetry install`, etc.). Use the existing `jaxTorchTrans` conda environment.
- **Non-destructive git:** do not use destructive git (no `--force` push, no `git reset --hard`) without explicit human approval.
- **Secrets:** never commit `.env`, credentials, or keys.

2) Install / environment
-------------------------
- Preferred: `poetry install` (uses `pyproject.toml`). If not using Poetry: create a venv and `pip install -r requirements.txt`.
- Typical dev env:
  - `conda activate jaxTorchTrans`
  - Set JAX/XLA env vars for JAX runs (examples below).

3) Canonical training example
-----------------------------
Use this when reproducing experiments on a GPU node:

```bash
conda activate jaxTorchTrans
JAX_COMPILATION_CACHE_DIR="jax_cache" \
  CUDA_VISIBLE_DEVICES="2" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 \
  python qwen_fineTuning_PtoC_fixCps.py --batch-size=4 --smoke-test
```

4) Tests (quick commands)
-------------------------
- Run all tests: `poetry run pytest` or `pytest` in an activated env.
- Run a single test file: `pytest -q src/test_subspace_curve.py`.
- Run a single test function: `pytest -q path/to/test_file.py::test_name`.
- Filter by keyword: `pytest -q -k "keyword"`.
- Run with verbosity/trace: `pytest -q path/to/test_file.py::test_name -s -vv`.
- Parallel: `pytest -n auto` (requires `pytest-xdist`).
- Coverage: `pytest --cov=src` (requires `pytest-cov`).

5) Linting & formatting
------------------------
- Format: `poetry run black .` (line length 88). Check with `black --check .`.
- Sort imports: `poetry run isort .` or `ruff --fix`.
- Lint: `poetry run ruff check .` or `ruff check src tests`.
- Type checking: `poetry run mypy src` (optional).
- Pre-commit: if configured, `pre-commit run --all-files`.

6) Code style guidelines (for agents)
-------------------------------------
- Formatting: use `black` defaults (88 cols). Keep expressions simple and break long expressions.
- Imports: order as stdlib, third-party, local; one blank line between groups; prefer absolute imports (`from src.module import Foo`).
- Avoid wildcard imports (`from ... import *`).
- Typing: prefer explicit annotations on public functions; use `Optional[T]` not `Union[T, None]`; use `TypedDict` for stable dict configs; document complex pytrees.
- Naming: modules `lowercase_with_underscores.py`; functions/vars `snake_case`; classes/exceptions `PascalCase`; constants `UPPER_SNAKE_CASE`.
- Functions/APIs: small and focused; public APIs must have docstrings describing args, returns, and side effects.

7) Error handling & logging
---------------------------
- Use specific exceptions (`ValueError`, `TypeError` or custom). Avoid bare `except:`.
- Preserve exception context: `raise NewError(...) from exc`.
- Use `logging` (library code) not `print()`; initialize via `logger = logging.getLogger(__name__)` and avoid configuring global handlers in libraries.

8) JAX / Flax specifics
-----------------------
- Keep jitted functions pure and side-effect free; pass mutable state explicitly.
- Use `jax.lax.scan` when possible to avoid recompilation inside loops.
- Manage dtypes explicitly; document bfloat16/float32 mixed precision use.
- Name pytrees and masks clearly; provide helper converters and docstrings for flatten/pack utilities.

9) Repo-specific conventions
---------------------------
- LoRA and masks: represent adapters and masks as structured pytrees/dicts. When adding new formats provide conversion helpers and tests.
- Follow patterns in `src/subspace_curve.py` and `src/qwen_jax.py` for mask application and weight mapping.

10) Key files and where to look
------------------------------
- `src/subspace_curve.py` — core subspace math and LoRA mixing.
- `src/qwen_jax.py` — model implementation and weight mapping helpers.
- `qwen_fineTuning_PtoC_fixCps.py` — training entrypoint used by sweeps.
- `configs/*.yaml` — example config files for sweeps and runs.

11) Serialization
-----------------
- Use repository helpers `save_params_and_config` / `load_params_and_config` when present. Saved params use `jax_params.npy` and `jax_config.json`.

12) Testing guidelines (detailed)
--------------------------------
- Use `pytest` fixtures for reusable setup.
- Keep tests small, deterministic and fast; mock heavy GPU/JAX parts when appropriate.
- Name tests like `test_function_does_x_given_y` and add short docstrings for non-obvious tests.
- Place tests under `tests/` or alongside module under `src/` (see `src/test_subspace_curve.py`).

13) Cursor / Copilot rules
-------------------------
- Cursor rules: none found (no `.cursor/rules/` or `.cursorrules`).
- Copilot rules: this repo contains `.github/copilot-instructions.md` — include and follow its guidance for domain-specific examples.

14) Excerpted guidance (short)
-----------------------------
Purpose: Time Transformer research framework—Bézier-curve parameterized neural nets, LoRA fine-tuning, and experiment orchestration.

Essential pattern: `SubspaceBaseModel` (in `src/subspace_curve.py`) is central — expose `nll`, `__call__`, `train_step`, and helpers `init_mask`, `init_params_from_point`.

15) Next steps for agents
-------------------------
1. Run formatting and linting: `poetry run black . && poetry run ruff check .`.
2. Run a focused unit test while iterating: `pytest -q src/test_subspace_curve.py::test_name`.
3. When changing LoRA/mask formats, add converter helpers and unit tests; follow `src/subspace_curve.py` and `src/qwen_jax.py`.

16) Recent Progress (Context for following agents)
--------------------------------------------------
### Goal
Finalize ICML 2026/NeurIPS 2026 paper text for PtoC method (Overleaf project: `default`).

### Accomplished
- **NeurIPS Migration:** Migrated `main.tex` to NeurIPS 2026 style.
- **BMA Narrative:** Framed PtoC as a "path-integral BMA" bridging point-estimates and MCMC.
- **MFVI Insight:** Updated user story: the LoRA loss landscape is **not** overly complex, as MFVI works surprisingly well.
- **Figures:** Integrated Figure 1 (TikZ) comparing MAP, MFVI, DE, Curve, and PSMILE.
- **Experimental Results:** Updated `main.tex` with empirical insights (PtoC > DE(5) on ARC-Challenge LL).

### Work In Progress
- **PSMILE Analysis:** Finalize Figure 2/Analysis for particle-based MCMC vs Curve BMA.
- **MFVI Theory:** Refine text explaining the efficacy of MFVI in the LoRA subspace.
- **Consistency:** Ensure $\nseg$ ($N_{\text{seg}}$) and $\segdeg$ notation remains uniform.

Notes
-----
- This file is written for automated agents. It emphasizes safety (wandb read-only, no destructive git) and practical developer commands for working with the repo.
