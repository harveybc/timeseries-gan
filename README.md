# timeseries-gan (package: `tsg`)

> **⚠️ SUPERSEDED — this repository is legacy and no longer maintained (last substantive commit: 2025-06-21).**
>
> Synthetic time-series generation now lives in [harveybc/synthetic-datagen](https://github.com/harveybc/synthetic-datagen), which owns the `sdg` CLI and the "Synthetic Data Generator" name.
>
> Use synthetic-datagen for any new work. This repository is retained for historical reference only and is not required by any current deployment.

## What this was

A plugin-based framework for training a Sequential Conditional VAE-GAN (SC-VAE-GAN) on multi-feature financial time series and generating synthetic sequences (OHLC prices, technical indicators, date features). The installed package is named `tsg` and it provides a `tsg` console script with train, generate, and (GA-based) hyperparameter-optimization modes. Plugins are loaded through setuptools entry-point groups from [`tsg_plugins/`](tsg_plugins/) (see [`setup.py`](setup.py)); deeper historical documentation is in [REFERENCE.md](REFERENCE.md).

## Known defects and naming hazards

- **Naming mismatch:** the repository is `timeseries-gan` but the package is `tsg`. Earlier versions of this README were titled "Synthetic Data Generator (SDG)" and showed `sdg ...` commands — that name and CLI now belong to [synthetic-datagen](https://github.com/harveybc/synthetic-datagen); the command installed by this repository has always been `tsg`.
- **Colliding entry-point groups:** plugins are registered under unqualified groups (`feeder.plugins`, `generator.plugins`, `discriminator.plugins`, `evaluator.plugins`, `optimizer.plugins`, `trainer.plugins`). These names are not namespaced to this project and collide with other packages that claim the same groups (e.g. `feeder.plugins` is also used elsewhere), so installing `tsg` alongside such packages in one environment corrupts plugin discovery.
- **Committed residue:** trained weights (`gan_epoch_500.keras`, `generator_epoch_500.keras`, `discriminator_epoch_500.keras`), debug/one-off scripts at the repository root, backup plugin files, and `__pycache__` directories were committed. Treat them as leftovers, not reference artifacts.

## Historical usage — unverified in current environments

The commands below reflect how the tool was originally used (with the correct `tsg` command). They have **not** been re-verified in current environments.

```bash
git clone https://github.com/harveybc/timeseries-gan.git
cd timeseries-gan
pip install -r requirements.txt   # historical — unverified in current environments
pip install .

tsg --trainer gan_trainer --gan_epochs 1000            # train — historical, unverified
tsg --n_samples 1000 --output_file synthetic_data.csv  # generate — historical, unverified
```

Sample EUR/USD data, results, and scripts from the original experiments are under [`examples/`](examples/).

## Limitations

- No maintenance, no issue support, no compatibility work is planned.
- TensorFlow/Keras and CUDA pins in [`requirements.txt`](requirements.txt) target the original 2025-era environment and are unverified today.
- The successor, synthetic-datagen, uses properly namespaced `sdg.*` plugin groups and different generation methods; nothing here is interchangeable with it.

## License

MIT — see [LICENSE.txt](LICENSE.txt).
