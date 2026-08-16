# AGENTS.md — timeseries-gan (package `tsg`)

> **This repository is superseded and unmaintained.** Synthetic time-series
> generation now lives in
> [harveybc/synthetic-datagen](https://github.com/harveybc/synthetic-datagen),
> which owns the `sdg` CLI. Use that for any new work. This repository is kept
> for historical reference and is not required by any current deployment.

## Project overview

`timeseries-gan` trains a Sequential Conditional VAE-GAN on multi-feature
financial time series and generates synthetic sequences. The installed package
is `tsg` and it provides a `tsg` console script with train, generate and
GA-based hyperparameter-optimization modes, built on plugins loaded through
setuptools entry points from `tsg_plugins/`.

It does not serve models, execute trades, or acquire data. Its outputs are
Keras checkpoints and synthetic sequence arrays.

**Status: incomplete as shipped — runnable only in part.** The code itself is
substantially complete: `tsg_plugins/` contains roughly 46 modules, all seven
entry-point targets in `setup.py` resolve to files that exist, and all six
plugin classes import cleanly under TensorFlow 2.21 / Keras 3.15 with full
`plugin_params` dictionaries. What does not work is the pipeline around them.
The CLI cannot load a single plugin without a package install, the entry-point
groups are unqualified and collide with sibling repositories, the model files
the default config points at were never committed, and the test suite crashes
during collection. See [Verified state](#verified-state).

The one thing that does work end to end is **generation directly from the
committed checkpoints**, which is what the quickstart below does.

## Agent quickstart (load a checkpoint → generate → show the user results)

There is no working install-and-run path. Do not attempt to fix one before
reading [Do not touch](#do-not-touch) — installing this package into a shared
environment is actively harmful.

All commands were executed from the repository root on Python 3.12.13,
TensorFlow 2.21.0, Keras 3.15.0, and left the working tree clean.

### 1. Environment

Use an existing Python 3 environment with `tensorflow`, `keras`, `numpy` and
`pandas`. `requirements.txt` lists twelve packages with **no version pins** and
targets the original 2025-era environment.

Do **not** `pip install` this package or its requirements into a shared
environment — see [Do not touch](#do-not-touch).

Set an output directory outside the repository, and hide the GPUs so nothing
here can disturb other work:

```bash
export TSG_OUT="${TMPDIR:-/tmp}/timeseries-gan-agent"
mkdir -p "$TSG_OUT"
export CUDA_VISIBLE_DEVICES=""
```

`pandas_ta` is listed in `requirements.txt` but is missing from current
environments, and version 0.3.14b0 does `from numpy import NaN`, so it cannot
install against NumPy 2.x at all. `tsg_plugins/generator_plugin/pandas_ta_compat.py`
degrades gracefully — imports succeed with a warning — but technical-indicator
generation is unavailable.

### 2. Smoke test

```bash
PYTHONPATH=. python -c "import app.main; print('import OK')"
PYTHONPATH=. python app/main.py --help
```

Both succeed. `--help` prints roughly 90 arguments and exits 0.

Now confirm the actual defect, so you do not waste time on the CLI:

```bash
PYTHONPATH=. python -c "
from importlib.metadata import entry_points
for g in ['feeder.plugins','generator.plugins','discriminator.plugins',
          'evaluator.plugins','trainer.plugins','optimizer.plugins']:
    print(g, len(list(entry_points(group=g))))"
```

Observed: `0` for every group except `optimizer.plugins`, which returns **2
entries belonging to other projects** — the unqualified group name is squatted
by sibling repositories in this stack. `app/plugin_loader.py` resolves plugins
only through `importlib.metadata`, with no filesystem fallback, so `PYTHONPATH`
cannot help and there is no `*.egg-info` in the repository.

Consequently the CLI pipeline fails:

```
Failed to find plugin default_feeder in group feeder.plugins
...
Loading Optimizer Plugin: default_optimizer
Successfully loaded plugin: default_optimizer   <- another project's class
CRITICAL ERROR: Core plugin 'feeder_plugin' failed to load or initialize.
Error: FeederPlugin is required for generate mode but was not loaded.
```

Note the middle line: the only plugin that loads is the wrong one, silently
cross-wired from a different repository. That is the collision the README
warns about, happening for real.

### 3. Representative run — generate a synthetic sequence from a committed checkpoint

Training is not attempted: it is heavy and the plugin pipeline does not load.
Instead, load the committed generator checkpoint directly. It is a real trained
model from the original experiments and it still runs. Takes about four seconds
on CPU.

```bash
python - "$TSG_OUT" <<'PYEOF'
import sys, numpy as np, pandas as pd, keras

OUT = sys.argv[1]
keras.config.enable_unsafe_deserialization()   # checkpoint contains Lambda layers
g = keras.models.load_model("generator_epoch_500.keras", compile=False, safe_mode=False)

print("generator inputs:", [(t.name, tuple(t.shape)) for t in g.inputs])
print("generator output:", tuple(g.outputs[0].shape), "| params:", g.count_params())

rng = np.random.default_rng(42)
n = 4
seq = g.predict([rng.normal(size=(n, 100)).astype("float32"),   # noise_input
                 rng.normal(size=(n, 10)).astype("float32"),    # conditional_input_to_vae
                 rng.normal(size=(n, 64)).astype("float32")],   # context_input_to_vae
                verbose=0)

print("generated:", seq.shape, seq.dtype)
print(f"  min={seq.min():.4f} max={seq.max():.4f} mean={seq.mean():.4f}")

df = pd.DataFrame(seq[0], columns=[f"f{i:02d}" for i in range(seq.shape[2])])
df.insert(0, "step", range(len(df)))
df.to_csv(f"{OUT}/tsg_generated_sequence.csv", index=False)
print("wrote", f"{OUT}/tsg_generated_sequence.csv",
      f"({len(df)} steps x {seq.shape[2]} features)")
PYEOF
```

Observed output:

```
generator inputs: [('noise_input', (None, 100)), ('conditional_input_to_vae', (None, 10)), ('context_input_to_vae', (None, 64))]
generator output: (None, 144, 51) | params: 122759
generated: (4, 144, 51) float32
  min=-6.0205 max=4.2359 mean=0.0082
wrote .../tsg_generated_sequence.csv (144 steps x 51 features)
```

`safe_mode=False` is mandatory: the checkpoints embed `Lambda` layers holding
Python lambdas, which Keras 3 refuses to deserialize by default.
`discriminator_epoch_500.keras` loads without it (575,361 params, input
`(None, 144, 51)`); `generator_epoch_500.keras` (122,759 params) and
`gan_epoch_500.keras` (698,120 params) do not.

### 4. Analytics step — compare generated against real distributions

The output is 51 features in a normalized space. The largest committed real
dataset, `examples/data/phase_3/normalized_d4.csv`, has **45 columns** (44
features plus `DATE_TIME`, 30,425 rows: `RSI`, `MACD`, `EMA`, `ADX`, `ATR`,
`OPEN`/`HIGH`/`LOW`/`CLOSE`, `S&P500_Close`, `vix_close`, `CLOSE_15m_tick_*`,
`CLOSE_30m_tick_*`, calendar features). **The widths do not match and no
committed config maps the checkpoint's 51 channels to names**, so any column
alignment is a guess. Compare distributions in aggregate rather than pretending
to a per-feature mapping:

```bash
python - "$TSG_OUT" <<'PYEOF'
import sys, pandas as pd, numpy as np
OUT = sys.argv[1]
gen = pd.read_csv(f"{OUT}/tsg_generated_sequence.csv").drop(columns=["step"])
real = pd.read_csv("examples/data/phase_3/normalized_d4.csv").drop(columns=["DATE_TIME"])
for name, d in (("generated", gen), ("real normalized_d4", real)):
    v = d.to_numpy(dtype="float64").ravel()
    v = v[np.isfinite(v)]
    print(f"{name:22s} cols={d.shape[1]:3d}  mean={v.mean():8.4f}  std={v.std():7.4f}"
          f"  p01={np.quantile(v,.01):8.4f}  p99={np.quantile(v,.99):8.4f}")
PYEOF
```

Observed output:

```
generated              cols= 51  mean= -0.0053  std= 2.4567  p01= -6.0201  p99=  4.2031
real normalized_d4     cols= 44  mean=  0.3565  std= 0.2295  p01=  0.0022  p99=  0.9369
```

The two are not on the same scale at all: the committed real data is min-max
scaled into `[0, 1]`, while the generator emits roughly zero-centred values
with a standard deviation near 2.5. So this checkpoint was trained against a
different normalization than the dataset committed here, on top of the
44-versus-51 width mismatch. Report that rather than trying to force an
overlay.

### 5. Final message to the user

Report exactly this, with `<out>` replaced by the value of `$TSG_OUT`:

> Done — with an important caveat. **This repository is superseded and only
> partly runnable.** Its normal CLI cannot load a single plugin, so I generated
> directly from the committed trained checkpoint instead.
>
> Results are in `<out>/` (outside the repository; nothing in the repo was
> modified):
>
> - `<out>/tsg_generated_sequence.csv` — one synthetic sequence,
>   **144 timesteps × 51 features**, from `generator_epoch_500.keras`
>   (122,759 parameters). Values sit in a normalized range, roughly -6.0 to
>   +4.2, mean ≈ 0.
>
> What is actually broken: the package is not installed and there is no
> `egg-info`, so all six plugin groups resolve to nothing — except
> `optimizer.plugins`, which returns two plugins belonging to *other*
> repositories, because the group names are not namespaced. The generator and
> discriminator files the default config points at
> (`examples/results/phase_4_3/phase_4_3_*_model.keras`) were never committed,
> and the test suite crashes during collection.
>
> There is no web UI; these are files on disk.
>
> **Analysis to try first:** compare the generated features against the
> committed real dataset `examples/data/phase_3/normalized_d4.csv` — overall
> mean, standard deviation and the 1st/99th percentiles across all feature
> columns. I already ran it, and the two do not line up: the real data is
> min-max scaled into `[0, 1]` (mean 0.36, std 0.23) while the generator emits
> zero-centred values with std 2.46 spanning -6.0 to +4.2, and the widths
> differ too (51 generated channels versus 44 real features). Working out
> whether this checkpoint predates the committed normalization or was trained
> on a different feature set is the first thing worth knowing here. If you
> need working synthetic data rather than an archaeology exercise, use
> [synthetic-datagen](https://github.com/harveybc/synthetic-datagen) instead,
> where the equivalent task runs end to end in about two seconds.

## Build, test and lint commands

```bash
# CLI help (works)
PYTHONPATH=. python app/main.py --help

# Tests — collection crashes; the ignore is mandatory
python -m pytest --collect-only -q --ignore=tests/unit/test_trainer_fix.py
```

Without the ignore, collection dies with
`INTERNALERROR> SystemExit: 1` — `tests/unit/test_trainer_fix.py` calls a bare
`exit(1)` at module import, because it is a hand-run script, not a test. With
it ignored, **120 tests collect with 5 errors**, all of them stale imports of
APIs that no longer exist: `app.cli.main`, `app.config.Config`,
`app.config.CONFIG`, `tsg_plugins.generator_plugin.normalization_handler`, and
`tsg_plugins.discriminator_plugin` as a package. `tests/conftest.py` also
inserts a `../feature-extractor` path, a cross-repo dependency.

There is no linter, formatter or CI configuration.

## Layout

| Path | Contents |
|---|---|
| `app/main.py` | CLI entry. Dispatches on `operation_mode` (`train` / `generate` / `optimize`), default `generate`. |
| `app/cli.py`, `app/config.py`, `app/config_merger.py`, `app/config_handler.py` | ~90 arguments, defaults, config precedence |
| `app/pipeline/` | `train_pipeline.py`, `generate_pipeline.py`, `optimize_pipeline.py` |
| `app/data_generation/`, `app/evaluation/`, `app/utils/` | Real-data processing and synthetic generation, metrics, latent-shape inference, logging, output management |
| `app/plugin_loader.py` | Entry-point plugin resolution. Still ships four debug `print()` calls in the hot path. |
| `tsg_plugins/feeder_plugin/` | 11 modules: encoder handling, latent validation, conditioning, sampling, preprocessing |
| `tsg_plugins/generator_plugin/` | 15 modules: VAE-GAN generator, model load/save, feature processing and validation, indicators |
| `tsg_plugins/gan_trainer_plugin/` | 11 modules: training coordination, model building, persistence, metrics |
| `tsg_plugins/optimizer_plugin/` | 9 modules: genetic optimization |
| `tsg_plugins/discriminator_plugin.py`, `evaluator_plugin.py` | Single-module plugins |
| `examples/config/` | ~105 configs across 12 phase directories |
| `examples/data/` | Committed experiment data by phase. `phase_3/base_d4.csv` (`DATE_TIME,OPEN,LOW,HIGH,CLOSE`, 30,425 rows) and `phase_3/normalized_d4.csv` (45 columns, 30,425 rows) are the most useful. |
| `examples/results/` | Outputs of past experiments, including 22 `.keras` encoder/decoder models |
| `examples/scripts/` | 16 shell drivers. The phase_4 ones invoke `sdg.sh`, which does not exist in this repository. |
| `tests/` | 120 collectable tests, 5 import errors, 1 collection-crashing module |
| `*_epoch_500.keras` at the root | The three trained checkpoints. Committed residue, but the only thing that still runs. |
| `REFERENCE*.md`, `ARCHITECTURE_23_FEATURES.md`, `23_FEATURE_*.md`, `TASK_COMPLETION_SUMMARY.md` | Historical design documentation |
| `debug_*.py`, `test_*.py`, `analyze_*.py`, `*_sequential*.py` at the root | One-off debug scripts committed to the root |

## Verified state

Established by direct execution, not inference:

**Works**

- `import app.main` and `PYTHONPATH=. python app/main.py --help` both exit 0.
- All seven `setup.py` entry-point targets resolve to files that exist.
- All six plugin classes import and expose `plugin_params` (9 to 44 keys each).
- All three root checkpoints load under Keras 3.15 with no architecture or
  weight errors — `discriminator_epoch_500.keras` directly, the other two with
  `safe_mode=False`.
- `generator_epoch_500.keras` produces `(N, 144, 51)` float32 output from
  random latents in about four seconds on CPU.

**Broken**

- **Plugin discovery returns nothing.** `tsg` is not installed, there is no
  `*.egg-info`, and `app/plugin_loader.py` has no filesystem fallback.
- **Entry-point groups are unqualified** (`feeder.plugins`, `generator.plugins`,
  `discriminator.plugins`, `evaluator.plugins`, `optimizer.plugins`,
  `trainer.plugins`). `optimizer.plugins` is already claimed by two sibling
  repositories, and querying it returns *their* plugins.
- **The `app` package name is generic** and is also shipped by sibling repos.
  Where `tsg` is installed alongside them, its console script loads a foreign
  `app.main` and dies on `ModuleNotFoundError: No module named 'config_merger'`.
- **The default generate-mode models were never committed.**
  `app/config.py` points at `examples/results/phase_4_3/phase_4_3_generator_model.keras`
  and `..._discriminator_model.keras`; that directory contains only
  `phase_4_3_cnn_small_{encoder,decoder}_model.keras` and plots.
- **No config references the root checkpoints.** Grep across all 105 configs,
  16 scripts and the test tree finds zero matches for `*_epoch_500.keras`.
- **There is no `--mode` flag.** `operation_mode` is settable only through
  `--load_config` or through `app/config_merger.py:process_unknown_args`, which
  pairs unknown argv tokens blindly and silently discards the whole set when
  their count is odd.
- **`tsg_plugins/optimizer_plugin.py` is dead code**, permanently shadowed by
  the `optimizer_plugin/` package of the same name.
- **Committed residue:** `plugin_api.py` is 0 bytes; three `*_backup.py`
  duplicates; `__pycache__` directories; debug scripts at the root.
- **`pandas_ta` is missing and uninstallable** against NumPy 2.x.

## Conventions and constraints

- **Plugin interface.** A plugin is a class with a class-level `plugin_params`
  dict; `app/plugin_loader.py` reads it to derive required parameters.
- **Config precedence.** Defaults in `app/config.py`, then a config file via
  `--load_config`, then CLI flags, then unknown args merged as plugin
  parameters.
- **Column contract.** Committed data uses uppercase `DATE_TIME,OPEN,HIGH,LOW,CLOSE`
  — note `phase_3/base_d4.csv` orders them `OPEN,LOW,HIGH,CLOSE`.
- **Generator signature.** Three inputs: `noise_input` (100),
  `conditional_input_to_vae` (10), `context_input_to_vae` (64). Output
  `(batch, 144, 51)`.
- **Naming.** The repository is `timeseries-gan`; the package and command are
  `tsg`. Earlier READMEs showed `sdg ...` commands — that name belongs to
  [synthetic-datagen](https://github.com/harveybc/synthetic-datagen) and never
  worked here.

## Do not touch

- **Never `pip install` this package into a shared environment.** This is the
  most important constraint in this file. Its entry-point groups are
  unqualified and its top-level package is named `app`; installing it alongside
  the sibling repositories cross-wires plugin discovery non-deterministically
  and can break projects that are currently working. If you must run it, use a
  dedicated throwaway virtual environment.
- **Do not install `pandas_ta`** into a shared environment. It requires
  NumPy 1.x and will force a downgrade that breaks everything else.
- **Do not start training runs.** They are heavy, they need GPUs that are in
  use elsewhere, and the plugin pipeline they depend on does not load anyway.
  Export `CUDA_VISIBLE_DEVICES=""` before running anything here.
- **Do not modify `examples/data/`, `examples/results/` or `examples/config/`.**
  Roughly 250 MB of committed experiment data and outputs from the original
  runs. Read-only.
- **Do not delete the root `*_epoch_500.keras` checkpoints.** They are
  committed residue by intent, but they are also the only part of this
  repository that still produces output.
- **Do not write outputs into the repository.** Use `$TSG_OUT` outside it.
- **Do not invest in repairs.** This repository is superseded. Fixing the
  entry-point namespacing, the `app` package name, the missing model files and
  the stale tests is a real project with no consumer. New work belongs in
  [synthetic-datagen](https://github.com/harveybc/synthetic-datagen), whose
  `sdg.*` groups are namespaced precisely to avoid these problems.
- **Do not touch other repositories** from here.
