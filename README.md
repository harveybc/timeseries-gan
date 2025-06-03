# Synthetic Data Generator (SDG)

A plugin-based framework for training and generating multi-feature time series data using a Sequential Conditional Variational Autoencoder–Generative Adversarial Network (SC-VAE-GAN).

## Operation Modes

- **Train Mode**: Train the SC-VAE-GAN on real data
- **Generate Mode**: Generate synthetic sequences with OHLC features, technical indicators, and date features  
- **Optimize Mode**: Hyperparameter optimization via genetic algorithms

## Installation

```bash
git clone https://github.com/harveybc/timeseries-gan.git
cd timeseries-gan
pip install -r requirements.txt
python -m build
pip install .
```

## Quick Start

```bash
# Train GAN (using default EUR/USD hourly data)
sdg --trainer gan_trainer --gan_epochs 1000

# Generate synthetic data after training
sdg --n_samples 1000 --output_file synthetic_data.csv

# Hyperparameter optimization
sdg --run_hyperparameter_optimization True --population_size 10 --n_generations 5
```

For detailed usage, configuration, and system documentation, see [REFERENCE.md](REFERENCE.md).

## Directory Structure

```
timeseries-gan/
├── app/                          # Core application modules
│   ├── config.py                 # Default configuration parameters
│   ├── cli.py                    # Command-line interface
│   ├── main.py                   # Entry point and orchestration
│   ├── data_processor.py         # Main pipeline orchestrator (~170 lines)
│   ├── plugin_loader.py          # Plugin discovery and loading
│   ├── config_merger.py          # Configuration merging
│   ├── pipeline/                 # Pipeline modules (under 200 lines each)
│   │   ├── train_pipeline.py     # GAN training workflow
│   │   ├── optimize_pipeline.py  # Hyperparameter optimization
│   │   └── generate_pipeline.py  # Data generation and evaluation
│   ├── data_generation/          # Data processing modules
│   │   ├── synthetic_generator.py # Synthetic data generation
│   │   └── real_data_processor.py # Real data processing
│   ├── evaluation/               # Evaluation metrics
│   │   └── metrics_evaluator.py  # Comprehensive evaluation
│   └── utils/                    # Utility modules
│       ├── latent_shape_inference.py # Latent shape compatibility
│       └── output_manager.py     # Output file management
├── tsg_plugins/                  # Plugin implementations
│   ├── feeder_plugin/            # FULLY MODULARIZED (394 lines main)
│   │   ├── feeder_plugin.py      # Main plugin interface and orchestration
│   │   ├── encoder_handler.py    # Keras model loading and encoding (201 lines)
│   │   ├── data_preprocessor.py  # Data normalization and cleaning (329 lines)
│   │   └── condition_manager.py  # Condition extraction and processing (394 lines)
│   ├── generator_plugin/         # FULLY MODULARIZED (360 lines main)
│   │   ├── generator_plugin.py   # Main plugin interface
│   │   ├── normalization_handler.py # Data normalization
│   │   ├── model_loader.py       # Model loading
│   │   ├── feature_processor.py  # Feature processing
│   │   ├── technical_indicator_calculator.py # TI calculations
│   │   ├── data_generator.py     # Data generation
│   │   ├── sequence_builder.py   # Sequence building
│   │   ├── feature_validator.py  # Feature validation
│   │   ├── initial_data_handler.py # Initial data handling
│   │   └── pandas_ta_compat.py   # Compatibility layer
│   ├── evaluator_plugin.py       # Model evaluation (368 lines)
│   ├── optimizer_plugin.py       # Hyperparameter optimization (299 lines)
│   └── trainer_plugin.py         # GAN training (629 lines)
├── examples/                     # Sample data and trained models
│   ├── data/                     # EUR/USD datasets by processing phase
│   ├── results/                  # Pre-trained models and results
│   └── scripts/                  # Example scripts
└── tests/                        # Test suites
    ├── unit_tests/               # Unit tests for individual modules
    └── integration_tests/        # Integration tests for plugin interactions
```

## Features

- **57 Features**: OHLC prices, 15 technical indicators, date features, fundamental data
- **Extreme Modularity**: Feeder and generator plugins fully modularized with components under 400 lines each
- **Keras Integration**: Full migration to Keras/TensorFlow from PyTorch for better compatibility
- **Pre-trained Models**: Ready-to-use encoder/decoder models for EUR/USD hourly data
- **Flexible Architecture**: Plugin-based system for easy extension and customization
- **Comprehensive Testing**: Unit and integration tests for all modular components