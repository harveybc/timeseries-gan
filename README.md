# Synthetic Data Generator (SDG)

## Description

The **Synthetic Data Generator (SDG)** is a plugin-based framework for training and generating multi-feature time series data using a Sequential Conditional Variational Autoencoder–Generative Adversarial Network (SC-VAE-GAN). It supports:

- Training the SC-VAE-GAN on real data (e.g., EUR/USD hourly price dataset).
- Generating arbitrary-length synthetic sequences, including base market features (OHLC, log-returns, external fundamentals) and a full suite of technical indicators.
- Hyperparameter optimization via genetic algorithms.
- Plugin extensibility for feeder, generator, trainer, evaluator, optimizer, and preprocessors.

All configuration defaults reside in `app/config.py`. Command-line arguments accept every configuration key without default values (defaults loaded from `config.py`).

## Architecture & Modules

The project follows a **highly modular architecture** with **extreme separation of concerns**, where the main data processing pipeline has been refactored into focused modules under 200 lines each. The system is organized into the following components:

```
app/                   # Core orchestrator, config, and modular pipeline
├── main.py           # High-level orchestration (CLI parsing, plugin loading, config merging)
├── data_processor.py # Main pipeline orchestrator with operation mode dispatching
├── config.py         # Default configuration values
├── cli.py            # CLI argument parsing
├── plugin_loader.py  # Plugin loading and initialization
├── config_merger.py  # Configuration merging logic
├── pipeline/         # Operation mode pipelines (focused modules)
│   ├── train_pipeline.py     # GAN training workflow (~180 lines)
│   ├── optimize_pipeline.py  # Hyperparameter optimization workflow (~190 lines)
│   └── generate_pipeline.py  # Data generation and evaluation workflow (~190 lines)
├── data_generation/  # Synthetic and real data processing modules
│   ├── synthetic_generator.py   # Synthetic data generation logic (~200 lines)
│   └── real_data_processor.py  # Real data segment processing (~130 lines)
├── evaluation/       # Evaluation and metrics modules
│   └── metrics_evaluator.py    # Comprehensive evaluation metrics (~140 lines)
└── utils/            # Utility modules
    ├── latent_shape_inference.py # Latent shape inference (~190 lines)
    └── output_manager.py         # Output file management (~180 lines)

core/                  # Shared utilities, model builders (legacy structure)
tsg_plugins/           # Plugin implementations
docs/                  # Sphinx documentation sources  
tests/                 # Unit and integration tests
```

### app/ - Refactored Modular Pipeline

#### Core Orchestration
- **main.py**: High-level orchestration that handles CLI argument parsing, configuration loading/merging, plugin initialization, and dispatching to `run_pipeline()`. Maintains the exact plugin loading and configuration merging approach.
- **data_processor.py**: Main pipeline orchestrator (~170 lines) that handles operation mode dispatching ("train", "optimize", "generate"), latent shape inference, and delegates to specialized pipeline modules.
- **config.py**: Defines `DEFAULT_VALUES` for every CLI parameter.
- **cli.py**: Argparse definitions for all config keys (no defaults).
- **plugin_loader.py**: Dynamic loading based on setuptools entry points.
- **config_merger.py**: Merges defaults, file-based configs, remote configs, and CLI overrides.

#### Pipeline Modules (app/pipeline/)
Each pipeline module follows single responsibility principle and handles one specific operation mode:

- **train_pipeline.py** (~180 lines): Handles GAN training workflow including:
  - Training configuration validation and data availability checks
  - Training data loading and preprocessing
  - GAN training execution using trainer plugin
  - Post-training tasks (model saving, logging)

- **optimize_pipeline.py** (~190 lines): Handles hyperparameter optimization workflow including:
  - Optimization setup validation and plugin availability checks
  - Genetic algorithm parameter configuration
  - Optimization execution with fitness evaluation
  - Results handling and persistence

- **generate_pipeline.py** (~190 lines): Handles data generation and evaluation workflow including:
  - Generation configuration validation
  - Synthetic data generation using feeder and generator plugins
  - Real data segment processing and integration
  - Data combination and output management
  - Optional evaluation using evaluator plugin

#### Data Processing Modules (app/data_generation/)
- **synthetic_generator.py** (~200 lines): Encapsulates synthetic data generation logic including:
  - Target datetime sequence generation
  - Noise generation and conditioning using feeder plugin
  - Synthetic data generation using generator plugin
  - Initial window preparation for conditional generation
  - Data format conversion and validation

- **real_data_processor.py** (~130 lines): Handles real data processing including:
  - Real data segment loading and validation
  - Data preprocessing and format standardization
  - Integration preparation for combination with synthetic data

#### Evaluation Module (app/evaluation/)
- **metrics_evaluator.py** (~140 lines): Comprehensive evaluation metrics computation including:
  - Statistical metrics (MMD, Wasserstein distance, KS tests)
  - Temporal analysis (ACF, autocorrelation functions)
  - Machine learning based evaluation (discriminative score using RandomForest)
  - Predictive score evaluation
  - Results persistence and reporting
  - **Note**: Uses sklearn only for evaluation metrics, NOT for generator/discriminator models (which remain pure Keras with Conv1D, attention, LSTM, dense layers)

#### Utility Modules (app/utils/)
- **latent_shape_inference.py** (~190 lines): Focused utility for latent shape compatibility including:
  - Automatic latent shape detection from generator models
  - Plugin configuration updates for shape compatibility
  - Generator-feeder latent space coordination
  - Error handling and fallback mechanisms

- **output_manager.py** (~180 lines): Output file management and data operations including:
  - Output directory management and path resolution
  - Data combination (synthetic + real data segments)
  - File saving with proper formatting and validation
  - Output path updates based on evaluation stages

### tsg_plugins/

Each plugin follows the same mandatory interface structure with extreme separation of concerns. Plugins are organized as follows:

```
tsg_plugins/
├── gan_trainer_plugin/           # GAN Training Plugin (FULLY MODULARIZED - 385 lines main)
│   ├── __init__.py              # Package initialization  
│   ├── gan_trainer_plugin.py    # Main plugin interface (385 lines - COMPLETED)
│   ├── training_coordinator.py  # Core GAN training orchestration (~200 lines)
│   ├── model_builder.py         # Discriminator and GAN model construction (~180 lines)
│   ├── data_generator.py        # Training data generation and batching (~190 lines)
│   ├── model_persistence.py     # Model saving and loading operations (~280 lines)
│   ├── training_metrics.py      # Training progress tracking and visualization (~180 lines)
│   ├── technical_indicators.py  # TensorFlow technical indicator layer (~200 lines)
│   ├── parameter_manager.py     # Parameter extraction and validation (~190 lines)
│   ├── directory_manager.py     # Output directory management (~200 lines)
│   └── plugin_interface.py      # Plugin interaction management (~200 lines)
├── generator_plugin/            # Generator Plugin (PARTIALLY MODULARIZED)
│   ├── __init__.py              # Package initialization
│   ├── generator_plugin.py      # Main generator plugin (PENDING SIZE REDUCTION)
│   ├── model_loader.py          # Model loading and initialization
│   ├── normalization_handler.py # Data normalization and denormalization (~200 lines)
│   └── feature_processor.py     # Feature processing and technical indicators
├── feeder_plugin.py             # Feeder plugin (589 lines - PENDING REFACTORING)
├── optimizer_plugin.py          # Optimizer plugin (457 lines - PENDING REFACTORING)
├── evaluator_plugin.py          # Evaluator plugin
└── preprocessor_plugin.py       # Preprocessor plugin
```

#### Plugin Interface Requirements

All plugins must implement the following mandatory methods:
- `plugin_params`: Class-level parameter dictionary with defaults
- `__init__(config, *args)`: Initialize with configuration merging
- `set_params(**kwargs)`: Update parameters dynamically
- `get_debug_info()`: Return debug information dictionary
- `add_debug_info(key, value)`: Add debug information key-value pairs

#### Plugin Descriptions

- **gan_trainer_plugin**: **[FULLY MODULARIZED - 385 lines]** Coordinates GAN training with extreme separation of concerns across 10 specialized modules:
  - `gan_trainer_plugin.py` (385 lines): Main plugin interface with mandatory methods
  - `training_coordinator.py` (~200 lines): Core GAN training orchestration and epoch management
  - `model_builder.py` (~180 lines): Discriminator and GAN model construction with Keras layers
  - `data_generator.py` (~190 lines): Training data generation, batching, and real/fake data handling
  - `model_persistence.py` (~280 lines): Comprehensive model saving/loading with epoch templates
  - `training_metrics.py` (~180 lines): Training progress tracking, loss plotting, and visualization
  - `technical_indicators.py` (~200 lines): TensorFlow technical indicator calculations and layers
  - `parameter_manager.py` (~190 lines): Parameter extraction, validation, and configuration management
  - `directory_manager.py` (~200 lines): Output directory creation and file path management
  - `plugin_interface.py` (~200 lines): Generator/feeder plugin interaction and model extraction
- **generator_plugin**: **[PARTIALLY MODULARIZED]** Generator model wrapper with initial modular structure:
  - `generator_plugin.py`: Main generator plugin interface (size reduction pending)
  - `model_loader.py`: Model loading and initialization from pre-trained VAE decoder
  - `normalization_handler.py` (~200 lines): Data normalization/denormalization with min-max scaling
  - `feature_processor.py`: Feature processing and technical indicator calculations
- **feeder**: Supplies initial noise and conditional inputs (date features, autoregressive contexts)
- **evaluator**: Computes distributional, temporal, correlation, stylized facts, predictive, and visual metrics
- **optimizer**: Genetic algorithm hyperparameter tuner using DEAP
- **preprocessor**: (Optional) initial data transforms for upstream VAE training

#### Refactoring Status

- ✅ **GAN Trainer Plugin**: **FULLY MODULARIZED** into 10 focused modules (385 lines main plugin)
  - **Completed modules**: training_coordinator.py, model_builder.py, data_generator.py, model_persistence.py, training_metrics.py, technical_indicators.py, parameter_manager.py, directory_manager.py, plugin_interface.py
  - **Status**: All mandatory plugin methods preserved, comprehensive integration testing completed
- 🔄 **Generator Plugin**: **PARTIALLY MODULARIZED** - Initial structure created
  - **Completed modules**: normalization_handler.py (~200 lines), model_loader.py, feature_processor.py
  - **Status**: Main plugin file size reduction pending
- 🔄 **Feeder Plugin**: 589 lines - Next priority for modularization  
- 🔄 **Optimizer Plugin**: 457 lines - Scheduled for modularization

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

## Modular Architecture Benefits

### Extreme Separation of Concerns
The recent refactoring has transformed the architecture from a monolithic approach to a highly modular design with extreme separation of concerns:

**Before**: Single monolithic `data_processor.py` file with 1400+ lines handling all operations.

**After**: Modular architecture with 9 focused modules, each under 200 lines:

1. **`data_processor.py`** (~170 lines) - Clean orchestrator with operation mode dispatching
2. **`train_pipeline.py`** (~180 lines) - GAN training workflow  
3. **`optimize_pipeline.py`** (~190 lines) - Hyperparameter optimization workflow
4. **`generate_pipeline.py`** (~190 lines) - Data generation and evaluation workflow
5. **`synthetic_generator.py`** (~200 lines) - Synthetic data generation logic
6. **`real_data_processor.py`** (~130 lines) - Real data segment processing
7. **`metrics_evaluator.py`** (~140 lines) - Comprehensive evaluation metrics
8. **`latent_shape_inference.py`** (~190 lines) - Latent shape compatibility
9. **`output_manager.py`** (~180 lines) - Output file management

#### GAN Trainer Plugin Modular Architecture

The GAN Trainer Plugin has been completely refactored from a single 1320-line file into 10 specialized modules (385 lines main plugin), each with focused responsibilities:

**Core Plugin Interface** (385 lines):
- `gan_trainer_plugin.py`: Main plugin interface maintaining mandatory methods (plugin_params, __init__, set_params, get_debug_info, add_debug_info)

**Training Components**:
- `training_coordinator.py` (~200 lines): Orchestrates GAN training loops, epoch management, and training callbacks
- `data_generator.py` (~190 lines): Handles real/fake data generation, batching, and conditional input preparation
- `training_metrics.py` (~180 lines): Tracks training progress, manages loss history, and generates training visualizations

**Model Management**:
- `model_builder.py` (~180 lines): Constructs discriminator and GAN models using Keras Conv1D, LSTM, and attention layers
- `model_persistence.py` (~280 lines): Comprehensive model saving/loading with epoch-based templates and final model persistence
- `technical_indicators.py` (~200 lines): TensorFlow-based technical indicator calculations and custom layers

**Configuration and Infrastructure**:
- `parameter_manager.py` (~190 lines): Parameter extraction, validation, and configuration management across plugins
- `directory_manager.py` (~200 lines): Output directory creation, file path management, and result organization
- `plugin_interface.py` (~200 lines): Manages interactions with generator/feeder plugins and model extraction

**Integration Benefits**:
- **Maintainability**: Each module under 200 lines with single responsibility
- **Testability**: Individual module testing with focused unit tests  
- **Extensibility**: Easy to add new functionality without affecting other modules
- **Debugging**: Clear separation makes issue isolation straightforward
- **Plugin Compatibility**: 100% backward compatibility maintained with existing plugin interfaces

#### Generator Plugin Modular Architecture (Partial)

The Generator Plugin has begun modularization with initial specialized modules created:

**Completed Modules**:
- `normalization_handler.py` (~200 lines): Comprehensive data normalization and denormalization operations including:
  - Min-max normalization with parameter loading from JSON files
  - OHLC data handling with price relationship preservation
  - Log returns to price conversion
  - Feature-specific normalization statistics
- `model_loader.py`: Model loading and initialization from pre-trained VAE decoder models
- `feature_processor.py`: Feature processing and technical indicator calculations

**Pending**: Main `generator_plugin.py` size reduction to bring it under 400 lines by moving remaining functionality to specialized modules.

### Key Benefits

**Maintainability**: Each module has a single, well-defined responsibility making the codebase easier to understand, debug, and modify.

**Testability**: Smaller, focused modules are easier to unit test and validate independently.

**Extensibility**: New functionality can be added by creating new focused modules without affecting existing code.

**Operation Mode Dispatching**: Clear separation of "train", "optimize", and "generate" modes with dedicated pipeline modules.

**Plugin Compatibility**: The refactoring maintains 100% backward compatibility with existing plugins while improving the internal architecture.

**Code Reusability**: Focused modules can be reused across different operation modes and contexts.

**Reduced Complexity**: Breaking down complex workflows into smaller, manageable pieces reduces cognitive load for developers.

## Directory Structure

```
timeseries-gan/
├── app/                      # Refactored modular pipeline with extreme separation of concerns
│   ├── main.py              # Entry point: high-level orchestration (CLI, config, plugins)
│   ├── data_processor.py    # Main pipeline orchestrator with operation mode dispatching (~170 lines)
│   ├── cli.py               # CLI argument definitions (no defaults)
│   ├── config.py            # DEFAULT_VALUES for every parameter
│   ├── plugin_loader.py     # Plugin discovery and loading via setuptools entry points  
│   ├── config_merger.py     # Hierarchical configuration merging
│   │
│   ├── pipeline/            # Operation mode pipelines (focused modules under 200 lines)
│   │   ├── __init__.py
│   │   ├── train_pipeline.py     # GAN training workflow (~180 lines)
│   │   ├── optimize_pipeline.py  # Hyperparameter optimization workflow (~190 lines)
│   │   └── generate_pipeline.py  # Data generation and evaluation workflow (~190 lines)
│   │
│   ├── data_generation/     # Data processing modules (focused on single responsibilities)
│   │   ├── __init__.py
│   │   ├── synthetic_generator.py   # Synthetic data generation logic (~200 lines)
│   │   └── real_data_processor.py  # Real data segment processing (~130 lines)
│   │
│   ├── evaluation/          # Evaluation and metrics modules
│   │   ├── __init__.py
│   │   └── metrics_evaluator.py    # Comprehensive evaluation metrics (~140 lines)
│   │
│   └── utils/               # Utility modules (focused functionality)
│       ├── __init__.py
│       ├── latent_shape_inference.py # Latent shape inference (~190 lines)
│       └── output_manager.py         # Output file management (~180 lines)
│
├── core/                     # Shared utilities, model builders (legacy structure)
│   ├── data_io.py           # CSV read/write, sliding-window slicing, datetime utilities
│   ├── datetime_utils.py    # Generate synthetic datetimes, extract cyclical date features
│   ├── models/              # Builders for Keras architectures
│   │   ├── z_generator_builder.py   # Builds LSTM model for latent sequences Z_seq
│   │   ├── vae_decoder_loader.py    # Loads pre-trained VAE decoder model
│   │   └── discriminator_builder.py # Builds window-based Discriminator (Conv1D/LSTM)
│   ├── technical_indicators/ # Optional calculators for indicators
│   │   ├── calculator.py         # pandas-ta wrapper functions
│   │   └── tf_layer_calculator.py# TensorFlowTALayer for on-the-fly TI calculation
│   └── pipelines/           # Legacy pipeline structure (being phased out)
│       ├── base_pipeline.py      # Abstract base class for pipelines
│       ├── training_pipeline.py  # Legacy GAN training flow
│       └── generation_pipeline.py# Legacy synthetic data generation flow
│
├── tsg_plugins/             # Plugin implementations (each as focused modules)
│   ├── feeder_plugin.py           # FeederPlugin: noise & conditional input provider
│   ├── generator_wrapper_plugin.py# GeneratorPlugin: wraps pre-trained VAE decoder
│   ├── z_generator_plugin.py      # ZGeneratorPlugin: trainable latent sequence generator
│   ├── discriminator_plugin.py    # DiscriminatorPlugin: trainable window-based discriminator
│   ├── trainer_plugin.py          # GANTrainerPlugin: orchestrates GAN training steps
│   ├── evaluator_plugin.py        # EvaluatorPlugin: computes metrics on synthetic vs. real data
│   └── optimizer_plugin.py        # OptimizerPlugin: genetic-algorithm hyperparameter tuner
│
├── docs/                    # Sphinx documentation source
├── tests/                   # Unit and integration tests for core modules and plugins
└── setup.py                # Package metadata, dependencies, and plugin entry points
```

### Key Architectural Improvements

**Extreme Separation of Concerns**: The original monolithic `data_processor.py` (1400+ lines) has been completely refactored into focused modules:

1. **Operation Mode Dispatching**: `data_processor.py` now serves as a clean orchestrator (~170 lines) that dispatches to specialized pipeline modules based on operation mode ("train", "optimize", "generate").

2. **Focused Pipeline Modules**: Each operation mode has its own dedicated pipeline module under `app/pipeline/`:
   - `TrainPipeline`: Handles complete GAN training workflow
   - `OptimizePipeline`: Manages hyperparameter optimization using genetic algorithms  
   - `GeneratePipeline`: Coordinates synthetic data generation and evaluation

3. **Specialized Data Processing**: Data generation logic is separated into focused modules under `app/data_generation/`:
   - `SyntheticDataGenerator`: Handles synthetic data generation workflow
   - `RealDataProcessor`: Manages real data segment processing

4. **Dedicated Evaluation**: Evaluation logic is isolated in `app/evaluation/metrics_evaluator.py` with comprehensive metrics computation.

5. **Utility Modules**: Common utilities are organized under `app/utils/`:
   - `latent_shape_inference.py`: Automatic compatibility between generator and feeder plugins
   - `output_manager.py`: File management and data combination operations

**Plugin Compatibility Maintained**: The refactoring preserves the exact plugin loading and configuration merging approach, ensuring full backward compatibility with existing plugins.

**Single Responsibility Principle**: Each module now has a clear, focused responsibility and is kept under 200 lines for maintainability.