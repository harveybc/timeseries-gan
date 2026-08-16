# timeseries-gan (package: `tsg`)

> **⚠️ SUPERSEDED — this repository is legacy and no longer maintained (last substantive commit: 2025-06-21).**
>
> Synthetic time-series generation now lives in [harveybc/synthetic-datagen](https://github.com/harveybc/synthetic-datagen), which owns the `sdg` CLI and the "Synthetic Data Generator" name.
>
> Use synthetic-datagen for any new work. This repository is retained for historical reference only and is not required by any current deployment.

## What this was

A plugin-based framework for training a Sequential Conditional VAE-GAN
(SC-VAE-GAN) on multi-feature financial time series and generating synthetic
sequences (OHLC prices, technical indicators, date features). The installed
package is named `tsg` and it provides a `tsg` console script with train,
generate, and GA-based hyperparameter-optimization modes.

The pipeline in [`app/main.py`](app/main.py) dispatches on `operation_mode`
(`train` / `generate` / `optimize`, default `generate`) over six plugins loaded
through setuptools entry-point groups from [`tsg_plugins/`](tsg_plugins/): a
feeder, a generator, a discriminator, a GAN trainer, an evaluator and a genetic
optimizer. Deeper historical documentation is in [REFERENCE.md](REFERENCE.md),
[REFERENCE_Functionality.md](REFERENCE_Functionality.md),
[REFERENCE_Config_FileTree.md](REFERENCE_Config_FileTree.md) and
[ARCHITECTURE_23_FEATURES.md](ARCHITECTURE_23_FEATURES.md).

## Status

**Incomplete as shipped — runnable only in part.**

The code itself is substantially complete, not a hollow prototype:
[`tsg_plugins/`](tsg_plugins/) holds roughly 46 modules, all seven entry-point
targets declared in [`setup.py`](setup.py) resolve to files that exist, and all
six plugin classes import cleanly under TensorFlow 2.21 / Keras 3.15 with full
`plugin_params` dictionaries. Around 250 MB of genuine experiment data and
results are committed under [`examples/`](examples/).

What does not work is the pipeline around it. Verified by execution on
2026-08-16 (Python 3.12.13, TensorFlow 2.21.0, Keras 3.15.0):

| Check | Result |
|---|---|
| `import app.main`, `PYTHONPATH=. python app/main.py --help` | exit 0 |
| Plugin discovery for the six groups | `feeder`, `generator`, `discriminator`, `evaluator`, `trainer`: **0 entries**. `optimizer.plugins`: 2 entries **belonging to other repositories**. |
| CLI `generate` mode | fails — `Error: FeederPlugin is required for generate mode but was not loaded.` |
| Loading the three root `*_epoch_500.keras` checkpoints | all three load; two require `safe_mode=False` |
| Generating from `generator_epoch_500.keras` | works — `(N, 144, 51)` float32 in ~4 s on CPU |
| `python -m pytest --collect-only -q` | crashes: `INTERNALERROR> SystemExit: 1`. With `tests/unit/test_trainer_fix.py` ignored: 120 collected, 5 errors. |

See [`AGENTS.md`](AGENTS.md) for the full verified state and the one working
run recipe.

## Run this with an AI agent

Paste this into Claude Code, Cursor, Codex, GitHub Copilot or any coding agent with shell access:

> Read `AGENTS.md` in this repository and follow the **Agent quickstart** section end to end: set up the environment, run the smoke test, execute the example generation-from-checkpoint run, then tell me the exact file paths where I can see the results and one analysis I should try first.

`AGENTS.md` is the [agents.md](https://agents.md) convention, read natively by most coding agents.

## Known defects and naming hazards

- **Plugin discovery is dead without an install.** `tsg` is not installed in
  current environments and no `*.egg-info` is committed.
  [`app/plugin_loader.py`](app/plugin_loader.py) resolves plugins only through
  `importlib.metadata`, with no filesystem fallback, so `PYTHONPATH` does not
  help and every plugin group comes back empty.
- **Colliding entry-point groups.** Plugins are registered under unqualified
  groups (`feeder.plugins`, `generator.plugins`, `discriminator.plugins`,
  `evaluator.plugins`, `optimizer.plugins`, `trainer.plugins`). These names are
  not namespaced to this project. This is not theoretical: querying
  `optimizer.plugins` today returns two plugins registered by sibling
  repositories, so installing `tsg` alongside them corrupts plugin discovery in
  both directions.
- **The top-level package is named `app`,** which sibling repositories also
  ship. Where `tsg` is installed next to them, its console script loads a
  foreign `app.main` and dies on
  `ModuleNotFoundError: No module named 'config_merger'`.
- **Naming mismatch:** the repository is `timeseries-gan` but the package is
  `tsg`. Earlier versions of this README were titled "Synthetic Data Generator
  (SDG)" and showed `sdg ...` commands — that name and CLI now belong to
  [synthetic-datagen](https://github.com/harveybc/synthetic-datagen); the
  command installed by this repository has always been `tsg`.
- **The default generate-mode models were never committed.**
  [`app/config.py`](app/config.py) points at
  `examples/results/phase_4_3/phase_4_3_{generator,discriminator}_model.keras`;
  that directory contains only `phase_4_3_cnn_small_{encoder,decoder}_model.keras`
  and plots. Conversely, no committed config references the root
  `*_epoch_500.keras` checkpoints.
- **No `--mode` flag exists.** `operation_mode` is settable only through
  `--load_config` or through
  [`app/config_merger.py`](app/config_merger.py)`:process_unknown_args`, which
  pairs unknown argv tokens blindly and discards the whole set when their count
  is odd.
- **Stale tests.** `tests/unit/test_trainer_fix.py` calls a bare `exit(1)` at
  import and crashes pytest collection outright. Five further modules import
  APIs deleted in a refactor (`app.cli.main`, `app.config.Config`,
  `app.config.CONFIG`, `tsg_plugins.generator_plugin.normalization_handler`,
  `tsg_plugins.discriminator_plugin` as a package). `tests/conftest.py` inserts
  a `../feature-extractor` path.
- **`pandas_ta` is missing and uninstallable** against NumPy 2.x (0.3.14b0 does
  `from numpy import NaN`). Technical-indicator generation is unavailable;
  imports degrade gracefully with a warning.
- **Committed residue:** trained weights (`gan_epoch_500.keras`,
  `generator_epoch_500.keras`, `discriminator_epoch_500.keras`), debug and
  one-off scripts at the repository root, backup plugin files, a 0-byte
  `tsg_plugins/plugin_api.py`, a dead `tsg_plugins/optimizer_plugin.py`
  shadowed by the package of the same name, and `__pycache__` directories.
  `examples/scripts/phase_4_*.sh` invoke an `sdg.sh` that does not exist here.

## What still runs

Generation directly from the committed checkpoint, bypassing the CLI:

```python
import keras, numpy as np
keras.config.enable_unsafe_deserialization()   # the checkpoints embed Lambda layers
g = keras.models.load_model("generator_epoch_500.keras", compile=False, safe_mode=False)
rng = np.random.default_rng(42)
seq = g.predict([rng.normal(size=(4, 100)).astype("float32"),   # noise_input
                 rng.normal(size=(4, 10)).astype("float32"),    # conditional_input_to_vae
                 rng.normal(size=(4, 64)).astype("float32")])   # context_input_to_vae
# -> (4, 144, 51) float32
```

Set `CUDA_VISIBLE_DEVICES=""` first; it runs on CPU in about four seconds.
Full recipe, including a distribution comparison against the committed real
data, in [`AGENTS.md`](AGENTS.md).

## Historical usage — unverified in current environments

The commands below reflect how the tool was originally used (with the correct
`tsg` command). They have **not** been re-verified, and they cannot work
without first installing the package — which you should not do in a shared
environment, for the collision reasons above.

```bash
git clone https://github.com/harveybc/timeseries-gan.git
cd timeseries-gan
pip install -r requirements.txt   # historical — unverified in current environments
pip install .

tsg --trainer gan_trainer --gan_epochs 1000            # train — historical, unverified
tsg --n_samples 1000 --output_file synthetic_data.csv  # generate — historical, unverified
```

Sample EUR/USD data, results, and scripts from the original experiments are
under [`examples/`](examples/): ~105 configs across 12 phase directories, 62
data files, and 715 result files including 22 `.keras` models. The most useful
committed datasets are `examples/data/phase_3/base_d4.csv`
(`DATE_TIME,OPEN,LOW,HIGH,CLOSE`, 30,425 rows) and
`examples/data/phase_3/normalized_d4.csv` (45 columns, 30,425 rows).

## Limitations

- No maintenance, no issue support, no compatibility work is planned.
- TensorFlow/Keras and CUDA entries in [`requirements.txt`](requirements.txt)
  carry no version pins at all and target the original 2025-era environment.
- The successor, synthetic-datagen, uses properly namespaced `sdg.*` plugin
  groups and different generation methods; nothing here is interchangeable
  with it.

## License

MIT — see [LICENSE.txt](LICENSE.txt).
