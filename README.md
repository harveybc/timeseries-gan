# Synthetic Data Generator (SDG)

## Description

The **Synthetic Data Generator (SDG)** is a plugin-based framework for training and generating multi-feature time series data using a Sequential Conditional Variational Autoencoder–Generative Adversarial Network (SC-VAE-GAN). It supports:

- Training the SC-VAE-GAN on real data (e.g., EUR/USD hourly price dataset).
- Generating arbitrary-length synthetic sequences, including base market features (OHLC, log-returns, external fundamentals) and a full suite of technical indicators.
- Hyperparameter optimization via genetic algorithms.
- Plugin extensibility for feeder, generator, trainer, evaluator, optimizer, and preprocessors.

All configuration defaults reside in `app/config.py`. Command-line arguments accept every configuration key without default values (defaults loaded from `config.py`).

## Architecture & Modules

The project is organized into the following high-level components:

```
app/                   # Core orchestrator and config
core/                  # Shared utilities, model builders, pipelines
tsg_plugins/           # Plugin implementations (feeder, generator, Z-Gen, trainer, discriminator, evaluator, optimizer)
docs/                  # Sphinx documentation sources
tests/                 # Unit and integration tests
```

### app/

- **main.py**: Entry point. Loads configuration, initializes plugins, and dispatches to training or generation pipelines.
- **config.py**: Defines `DEFAULT_VALUES` for every CLI parameter.
- **cli.py**: Argparse definitions for all config keys (no defaults).
- **plugin_loader.py**: Dynamic loading based on setuptools entry points.
- **config_merger.py**: Merges defaults, file-based configs, remote configs, and CLI overrides.

### core/

- **data_io.py**: CSV loading/saving, window slicing, datetime utilities.
- **models/**: Keras model builders for Z-Generator (LSTM-based), VAE Decoder loader, Discriminator builder.
- **technical_indicators/**: Optional calculators for TIs (pandas-ta wrapper, TensorFlow layer).
- **pipelines/**: `TrainingPipeline` and `GenerationPipeline` orchestrating end-to-end flows.

### tsg_plugins/

Each plugin lives in its own subfolder with:

- `plugin.py`: Thin class delegating to helper modules.
- Supporting modules: noise providers, conditional builders, model handlers, training loops, fitness evaluators, metrics calculators.

Plugins:
- **feeder**: Supplies initial noise and conditional inputs (date features, autoregressive contexts).
- **generator_wrapper**: Wraps the pre-trained VAE decoder (main generator block).
- **z_generator_plugin**: Trainable latent sequence generator (feeding VAE decoder).
- **discriminator_plugin**: Trainable window-based discriminator model.
- **trainer**: Coordinates GAN training of Z-Gen + VAE-Dec vs. Discriminator.
- **evaluator**: Computes distributional, temporal, correlation, stylized facts, predictive, and visual metrics.
- **optimizer**: Genetic algorithm hyperparameter tuner using DEAP.
- **preprocessor**: (Optional) initial data transforms for upstream VAE training.

## Configuration Reference

All default values are defined in `app/config.py`. Below is a complete list of keys and their descriptions:

<!-- Configuration Keys and Descriptions -->

### Plugin Selection
- `feeder` (str): Feeder plugin name.
- `generator` (str): Generator plugin name (decoder wrapper).
- `evaluator` (str): Evaluator plugin name.
- `optimizer` (str): Optimizer plugin name.

### Data Sources & Generation
- `x_train_file`, `y_train_file`: CSV paths for training data.
- `x_validation_file`, `y_validation_file`: CSV paths for validation data.
- `x_test_file`, `y_test_file`: CSV paths for test data.
- `n_samples` (int): Number of synthetic samples/windows to generate.
- `max_steps_train` (int): Number of real data rows to prepend during generation.
- `seq_len` (int): Sequence length input/output for the VAE decoder.
- `latent_shape` ([int, int]): `(sequence_length, latent_dim)` shape for latent sequences.
- `batch_size` (int): Batch size for training/generation.

### Feeder Plugin Parameters
- `feeder_sampling_method` ("standard_normal"/"from_encoder"): How to sample noise.
- `feeder_encoder_sampling_technique` ("direct"/"kde"/"copula"): If sampling from encoder latent space.
- `encoder_model_file` (str): Path to pre-trained VAE encoder (.keras).
- `feeder_feature_columns_for_encoder` (List[str]): Input features for encoder sampling.
- `feeder_real_data_file_has_header` (bool): Header presence in CSV.
- `feeder_datetime_col_in_real_data` (str): Datetime column name.
- `feeder_date_features_for_conditioning` (List[str]): e.g., ["day_of_month","hour_of_day","day_of_week","day_of_year"].
- `feeder_fundamental_features_for_conditioning` (["S&P500_Close","vix_close"]).
- `feeder_max_day_of_*` (int): Ranges for date scaling.
- `feeder_context_vector_dim` (int): Dimensionality of optional context vector.
- `feeder_context_vector_strategy` ("zeros"/"random").
- `feeder_copula_kde_bw_method` (Optional[float]).

### Generator (VAE Decoder) Plugin Parameters
- `generator_sequential_model_file` (str): Path to the pre-trained decoder model.
- `discriminator_sequential_model_file` (str): Path to saved discriminator.
- `save_generator_sequential_model_file` (str).
- `save_discriminator_sequential_model_file` (str).
- `generator_decoder_input_window_size` (int): Expected input sequence length.
- `generator_full_feature_names_ordered` (List[str]): Final CSV columns order (including DATE_TIME, base features, TIs, date features).
- `generator_decoder_output_feature_names` (List[str]): Subset of features output by decoder.
- `generator_ohlc_feature_names` (List[str]): ["OPEN","HIGH","LOW","CLOSE"].
- `generator_ti_feature_names` (List[str]): TI names for optional post-processing.
- `generator_date_conditional_feature_names` (List[str]).
- `generator_feeder_conditional_feature_names` (List[str]).
- `generator_ti_calculation_min_lookback` (int).
- `generator_ti_params` (Dict[str,float]): Parameters for TI calculations (RSI, EMA, MACD, Bollinger Bands, etc.).
- `generator_normalization_params_file` (str): JSON with normalization min/max.
- `generator_decoder_input_name_*` (str): Name of each decoder input layer (latent, window, conditions, context).
- `context_vector_dim` (int): Must match feeder.

### Discriminator (Trainer) Parameters
- `base_feature_names_ordered` (List[str]): Features produced by VAE decoder.
- `feature_names_for_discriminator_ordered` (List[str]): Base features + TI names as generated by pandas-ta.
- `gan_epochs` (int): Total GAN training epochs.
- `gan_batch_size` (int).
- `generator_lr`, `generator_beta1` (float).
- `discriminator_lr`, `discriminator_beta1` (float).
- `discriminator_lstm_units` (int).
- `discriminator_dense_units` (int).
- `gan_save_interval` (int): Epoch interval for saving models.
- `gan_model_dir` (str).
- `gan_loss_plot_file` (str).

### Evaluator Plugin Parameters
- `evaluator_metrics` (List[str]): ["mmd","acf","wasserstein","kstest","discriminative_score","predictive_score","visual"].
- `evaluator_mmd_gamma` (Optional[float]).
- `evaluator_acf_nlags` (int).
- `evaluator_predictive_model_type` (str), `evaluator_predictive_epochs` (int), `evaluator_predictive_batch_size` (int).
- `evaluator_plot_max_features` (int), `evaluator_plot_max_lags_acf` (int).

### Optimizer Plugin Parameters
- `hyperparameter_optimization_mode` (bool).
- `run_hyperparameter_optimization` (bool).
- `population_size`, `n_generations` (int).
- `cxpb`, `mutpb` (float).
- `hyperparameter_bounds` (Dict[str,Tuple]).
- `optimizer_n_samples_per_eval` (int).
- `optimizer_start_datetime` (Optional[str]).

### General Execution & Output
- `random_seed` (int).
- `num_synthetic_samples_to_generate` (int).
- `start_datetime` (Optional[str]).
- `output_file` (str): Path for synthetic CSV.
- `metrics_file` (str).
- `save_config` (str).
- `save_log` (str).
- `quiet_mode` (bool).
- `datetime_col_name` (str).
- `target_column_order` (List[str]).
- `num_base_features_generated` (int).
- `preprocessor_plugin` (str).
- `gan_training_mode` (bool): Toggle between baseline and GAN-improved evaluation.
- `trainer` (str).

## Usage

```bash
# Train GAN:
sdg --trainer gan_trainer --gan_epochs 10000 --x_train_file path/to/train.csv --output_file synth.csv --metrics_file metrics.json

# Generate synthetic data (after training):
sdg --n_samples 1000 --max_steps_train 5000 --generator_sequential_model_file models/decoder.keras --discriminator_sequential_model_file models/disc.keras --output_file synth.csv

# Hyperparameter optimization:
sdg --run_hyperparameter_optimization True --population_size 20 --n_generations 10
```

All other parameters are available as CLI flags. Run `sdg --help` to list all arguments (no defaults shown).

## Directory Structure

```
timeseries-gan/
├── app/
│   ├── main.py               # Entry point: loads config, plugins, dispatches pipelines
│   ├── cli.py                # Defines all CLI flags (no defaults), mapping to config keys
│   ├── config.py             # DEFAULT_VALUES for every parameter
│   ├── plugin_loader.py      # Discovers and loads plugins via setuptools entry points
│   └── config_merger.py      # Merges CLI args, config file, remote config, DEFAULT_VALUES, plugin params
│
├── core/
│   ├── data_io.py            # CSV read/write, sliding-window slicing, datetime utilities
│   ├── datetime_utils.py     # Generate synthetic datetimes, extract cyclical date features
│   ├── models/               # Builders for Keras architectures
│   │   ├── z_generator_builder.py   # Builds LSTM (or other) model that generates latent sequences Z_seq
│   │   ├── vae_decoder_loader.py    # Loads pre-trained VAE decoder model for generation
│   │   └── discriminator_builder.py # Builds window-based Discriminator (Conv1D/LSTM)
│   ├── technical_indicators/ # Optional calculators for indicators if not in decoder output
│   │   ├── calculator.py         # pandas-ta wrapper functions
│   │   └── tf_layer_calculator.py# TensorFlowTALayer for on-the-fly TI calculation in GAN
│   └── pipelines/            # Orchestrates end-to-end workflows
│       ├── base_pipeline.py      # Abstract base class for pipelines
│       ├── training_pipeline.py  # Implements GAN training flow (Z-Gen + VAE-Dec vs. Disc)
│       └── generation_pipeline.py# Implements synthetic data generation flow
│
├── tsg_plugins/              # Plugin implementations (each as one Python file)
│   ├── feeder_plugin.py           # FeederPlugin: noise & conditional input provider
│   ├── generator_wrapper_plugin.py# GeneratorPlugin: wraps pre-trained VAE decoder
│   ├── z_generator_plugin.py      # ZGeneratorPlugin: trainable latent sequence generator
│   ├── discriminator_plugin.py    # DiscriminatorPlugin: trainable window-based discriminator
│   ├── trainer_plugin.py          # GANTrainerPlugin: orchestrates GAN training steps
│   ├── evaluator_plugin.py        # EvaluatorPlugin: computes metrics on synthetic vs. real data
│   └── optimizer_plugin.py        # OptimizerPlugin: genetic-algorithm hyperparameter tuner
│
├── docs/                     # Sphinx documentation source (reStructuredText or Markdown)
├── tests/                    # Unit and integration tests for core modules and plugins
└── setup.py                  # Defines package metadata, dependencies, and plugin entry points
```

**File Responsibilities**:

- **app/main.py**: Bootstraps application; reads CLI args; loads or fetches config; initializes plugins; invokes `TrainingPipeline` or `GenerationPipeline` based on `gan_training_mode`.
- **app/cli.py**: Uses `argparse` to define one argument per config key; flags have no defaults so values always load from `config.py` unless overridden.
- **app/config.py**: Central dictionary `DEFAULT_VALUES` containing defaults for all parameters in hierarchical order.
- **app/config_merger.py**: Combines sources in priority: CLI args > loaded config (file/remote) > `DEFAULT_VALUES` > plugin-specific defaults.
- **app/plugin_loader.py**: Scans entry points (feeder.plugins, generator.plugins, etc.) and imports classes by name.

- **core/data_io.py**: `load_csv()`, `write_csv()`, and functions to slice time-series windows.
- **core/datetime_utils.py**: `generate_preceding_datetimes()`, functions to compute cyclical features like sin/cos of day, hour.
- **core/models/z_generator_builder.py**: Constructs Keras model generating latent sequence `Z_seq` from noise/context.
- **core/models/vae_decoder_loader.py**: Loads pre-trained VAE decoder; ensures correct input/output signature.
- **core/models/discriminator_builder.py**: Builds Discriminator network combining Conv1D and/or LSTM layers.
- **core/technical_indicators/calculator.py**: Wraps `pandas-ta` for post-hoc TI calculation when needed.
- **core/technical_indicators/tf_layer_calculator.py**: Defines `TensorFlowTALayer` that computes TIs inside a Keras graph during GAN training.
- **core/pipelines/base_pipeline.py**: Abstracts common logic (config, plugin access, logging).
- **core/pipelines/training_pipeline.py**: Implements the adversarial loop: sample noise via Feeder, generate `Z_seq`, produce synthetic windows, compute discriminator loss, update Z-Gen, occasionally fine-tune VAE decoder, save models.
- **core/pipelines/generation_pipeline.py**: Given trained Z-Gen and VAE decoder, generates `n_samples` synthetic ticks, prepends `max_steps_train` real rows, writes output CSV.

- **tsg_plugins/feeder_plugin.py**: `FeederPlugin` class with `plugin_params` dictionary, methods `set_params()`, `generate_z_noise()` and `build_conditionals()`.
- **tsg_plugins/generator_wrapper_plugin.py**: `GeneratorPlugin` class wrapping VAE decoder; methods `load_model()`, `set_params()`, `generate_sequence(z_seq, cond, context)`, `post_process()` (optional TIs).
- **tsg_plugins/z_generator_plugin.py**: `ZGeneratorPlugin` class with `plugin_params` for latent_dim; methods `build_model()`, `set_params()`, `sample_z_seq()` during training and generation.
- **tsg_plugins/discriminator_plugin.py**: `DiscriminatorPlugin` class; methods `build_model()`, `set_params()`, `compute_loss()`.
- **tsg_plugins/trainer_plugin.py**: `GANTrainerPlugin` class orchestrating one epoch: calls feeder for noise/conditionals, z-generator to get `Z_seq`, generator_wrapper to produce synth, discriminator loss, optimizer steps, callbacks.
- **tsg_plugins/evaluator_plugin.py**: `EvaluatorPlugin` class computing metrics (MMD, ACF, Wasserstein, KS, discriminative and predictive scores, visual plots) after generation.
- **tsg_plugins/optimizer_plugin.py**: `OptimizerPlugin` class configuring DEAP; methods `set_params()`, `optimize()` that iteratively evaluate synthetic data quality.

This structure ensures each file is under 200 lines, each module has a single responsibility, and core application logic remains centralized under `app/`. The plugin system continues to merge parameters hierarchically using `config_merger.py` before invoking each plugin's own defaults.