# SDG System Reference Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Sequential Conditional VAE-GAN Architecture](#sequential-conditional-vae-gan-architecture)
3. [Operation Modes](#operation-modes)
4. [Configuration Parameters](#configuration-parameters)
5. [File Structure and Integration](#file-structure-and-integration)
6. [Detailed Code Documentation](#detailed-code-documentation)

---

## System Overview

The **Synthetic Data Generator (SDG)** is a sophisticated plugin-based framework that implements a Sequential Conditional Variational Autoencoder–Generative Adversarial Network (SC-VAE-GAN) for generating high-quality multi-feature time series data. The system is designed with extreme modularity and separation of concerns, making it highly maintainable, testable, and extensible.

### Key Features

- **51-Feature Time Series Generation**: Generates OHLC prices, 20 technical indicators, cyclical date features, and fundamental market data
- **Three Operation Modes**: Train, Generate, and Optimize with dedicated pipeline modules
- **Pre-trained Models**: Ready-to-use encoder/decoder models trained on EUR/USD hourly data
- **Extreme Modularity**: Largest module under 630 lines, with generator plugin fully modularized into 10 modules under 420 lines each
- **Plugin Architecture**: Extensible design supporting custom feeders, generators, evaluators, and optimizers

---

## Sequential Conditional VAE-GAN Architecture

### Architecture Overview

The system combines three key machine learning components in a sequential pipeline:

1.  **Variational Autoencoder (VAE)**: A pre-trained VAE decoder is a core component of the GAN's generator.
2.  **Conditional Generation**: Date/time features and previous step's synthetic output for temporal conditioning.
3.  **Generative Adversarial Network (GAN)**: Adversarial training for improved synthetic data quality, where the generator is a composite model.

### Training Process Enhancements

The GAN training process has been significantly enhanced with the following features:

1.  **Model Summaries**: Upon starting the training, Keras model summaries for both the generator and discriminator are printed to the logs. This allows for immediate verification of the model architectures.
2.  **Detailed Epoch Logging**: During training, progress for each epoch is logged, including:
    *   Current epoch number and total epochs.
    *   Generator loss (G_loss), Discriminator loss (D_loss), and its components (D_loss_real, D_loss_fake).
    *   Current learning rates for both generator and discriminator optimizers.
    *   Epoch duration.
    *   `ReduceLROnPlateau` specific information like patience counters (`wait`) and cooldown counters, if applicable.
3.  **Dynamic Learning Rate Adjustment**: The `ReduceLROnPlateau` Keras callback is integrated for both the generator and discriminator optimizers.
    *   It monitors `g_loss` for the generator and `d_loss` for the discriminator.
    *   Configuration parameters for `ReduceLROnPlateau` (e.g., `reduce_lr_factor`, `reduce_lr_patience`, `reduce_lr_min_delta`, `reduce_lr_cooldown`, `reduce_lr_min_lr`) can be set in `app/config.py` and are managed by the `GANTrainerPlugin`.
    *   The `TrainingCoordinator` manually calls the `on_epoch_end` method of these callbacks after each training epoch.
4.  **Early Stopping (Planned)**:
    *   Configuration parameters for `EarlyStopping` (e.g., `enable_early_stopping`, `es_patience`, `es_min_delta`, `es_monitor_metric`) are available in `GANTrainerPlugin`.
    *   **Status**: The callback is configured in `GANTrainerPlugin` but its integration into the `TrainingCoordinator`'s training loop (checking `model.stop_training`) is pending implementation.
5.  **Model Saving**:
    *   **Current Behavior**: The `TrainingCoordinator` saves model checkpoints (generator, discriminator, and the combined GAN model used for generator training) once after the entire training loop is completed. These are saved in the `models_dir` specified in the configuration, with filenames like `generator_epoch_{total_epochs}.keras`.
    *   **Intended Final Paths**: The `app/config.py` specifies `save_generator_sequential_model_file` and `save_discriminator_sequential_model_file`. The alignment of the final saving step to use these exact configured paths is pending. The composite GAN model (generator + non-trainable discriminator) is also saved with an epoch-based name.

### Regularization Strategy

A specific regularization strategy has been implemented to improve generalization and training stability:

### Generator Regularization

*   **L2 Regularization**: Applied to the `kernel_regularizer` (and `recurrent_regularizer` for LSTMs) of specific layer types within the generator:
    *   `Dense` layers
    *   `Conv1D` layers
    *   `LSTM` layers (including those wrapped by `Bidirectional` layers)
*   The L2 regularization factor (`generator_l2_reg`) can be configured in `app/config.py`. L2 regularization is only active if `use_generator_l2_reg` is set to `true` in the configuration.
*   **No Dropout or BatchNormalization**: Dropout and BatchNormalization layers are explicitly *not* used in the newly constructed parts of the generator models (e.g., the BiLSTM Z-Generator) to ensure the L2 strategy is the primary regularization technique. For the VAE-based generator, L2 is applied to its decoder's trainable layers, while its original architecture regarding Dropout/BatchNormalization is preserved unless those layers are among Dense, Conv1D, or LSTM.

### Discriminator Regularization

*   **No Regularization**: The discriminator is designed *without* any explicit regularization techniques:
    *   No `L2 regularization`.
    *   No `Dropout` layers.
    *   No `BatchNormalization` layers.
*   **No Label Smoothing**: Label smoothing is not employed. The discriminator is trained with hard labels (0 for fake, 1 for real).

This targeted regularization approach aims to prevent the discriminator from becoming too powerful too quickly and helps the generator learn more effectively.

### Feature Engineering and Consistency

A critical aspect of the SC-VAE-GAN's training stability is ensuring feature consistency between the synthetic data produced by the generator and the real data used to train the discriminator. Both data streams must present the same number and order of features to the discriminator.

To address this, the `GeneratorPlugin` incorporates a dedicated method, `prepare_features_for_discriminator`. This method processes raw real data batches before they are fed to the discriminator during training. Its responsibilities include:

1.  **Calculating Technical Indicators**: It applies the same technical indicator calculations (e.g., RSI, MACD, Bollinger Bands) to the real data as those defined for the synthetic data generation process.
2.  **Generating Cyclical Date/Time Features**: It computes cyclical representations of date and time components (e.g., hour of day, day of week) from the real data's timestamps.
3.  **Feature Alignment and Ordering**: It ensures that the processed real data contains all 51 features specified in the system configuration (e.g., `full_feature_names_ordered`). Any features missing after the previous steps are padded (e.g., with 0.0 if necessary), and all features are arranged in the precise order expected by the discriminator.

By applying this consistent feature engineering pipeline to the real data, the system guarantees that the discriminator receives input with the expected shape (e.g., `(batch_size, sequence_length, 51)`), resolving potential feature mismatch errors and contributing to a more stable and effective adversarial training process. This preparation step is handled within the `TrainingCoordinator` by calling the `generator_plugin.prepare_features_for_discriminator()` method on real data batches.

### Autoencoder Model Analysis

The pre-trained autoencoder models (located in `examples/results/phase_4_3/`) implement a sophisticated encoder-decoder architecture:

#### Encoder Model (`phase_4_3_cnn_small_encoder_model.keras`)
- **Input**: Multi-feature time series windows (144 timesteps × 57 features)
- **Architecture**: Conv1D layers with attention mechanisms and LSTM components
- **Output**: Latent representations with shape `[18, 32]` (sequence_length=18, latent_dim=32)
- **Purpose**: Compress real market data into meaningful latent representations

#### Decoder Model (`phase_4_3_cnn_small_decoder_model.keras`)
- **Input**: Designed to accept three inputs:
    1.  Latent sequence `z_seq` (e.g., shape `(batch_size, 18, 32)`), typically generated by an internal Z-generator.
    2.  Context vector `h_context` (e.g., shape `(batch_size, 64)`).
    3.  Conditional features (e.g., shape `(batch_size, 10)`).
- **Architecture**: Details of the decoder's internal layers (e.g., LSTMs, Dense layers).
- **Output**: Reconstructed features (e.g., 23 base features of shape `(batch_size, 23)`).
- **Integration in GAN Generator**:
    *   The `GeneratorPlugin` loads this pre-trained VAE decoder from the path specified in `config["generator_vae_decoder_model_path_param"]`.
    *   The VAE decoder's weights are set to `trainable=True` to allow fine-tuning during the adversarial GAN training process.
    *   It forms a core part of the composite GAN generator, which also includes an internal BiLSTM Z-generator (to produce `z_seq` from noise) and handles the various inputs.

---

## Operation Modes

The SDG system supports multiple operation modes, configurable via the `operation_mode` parameter in `app/config.py`. Each mode utilizes a specific pipeline to perform its tasks.

### Train Mode
- **Purpose**: To train the GAN components (Generator and Discriminator).
- **Process**: Involves loading real training data, building/compiling models (potentially using parts of pre-trained models like a VAE decoder for the generator), and running the adversarial training loop.
- **Key Configuration**: `operation_mode: "train"`. Relies on data paths like `x_train_file`, training parameters like `gan_epochs`, `learning_rate`, etc.

### Generate Mode

**Purpose**: To generate a specified number of synthetic time series samples using pre-trained Generator and Discriminator models and prepend them to an existing dataset.

**Activation**:
This mode is activated when the following conditions in `app/config.py` are met:
1.  `operation_mode: "generate"`
2.  `load_generator_sequential_model_file` is not `None` and points to a valid pre-trained full Generator Keras model file (`.keras`).
3.  `load_discriminator_sequential_model_file` is not `None` and points to a valid pre-trained full Discriminator Keras model file (`.keras`).

**Process Overview**:
1.  **Model Loading**: The system loads the full, pre-trained Generator and Discriminator models from the paths specified in `load_generator_sequential_model_file` and `load_discriminator_sequential_model_file` respectively.
2.  **Base Data Loading**: The initial segment of the real dataset is loaded from `x_train_file`, specifically the first `max_steps_train` rows.
3.  **Target `DATE_TIME` Calculation**:
    *   The `DATE_TIME` of the first record in the loaded segment of `x_train_file` is identified.
    *   The `dataset_periodicity` (e.g., "1h") is used to determine time intervals.
4.  **Synthetic Data Generation (`n_samples`)**:
    *   The system generates `n_samples` new synthetic data records.
    *   **`DATE_TIME` Generation**:
        *   The `DATE_TIME` for each synthetic sample is generated sequentially backwards from the target start time. The `DATE_TIME` of the *last* (most recent) synthetic sample generated will be exactly one `dataset_periodicity` unit before the `DATE_TIME` of the first record from `x_train_file`.
        *   Crucially, generated `DATE_TIME` values **must only fall on weekdays** (Monday to Friday). Any calculated timestamp that falls on a Saturday or Sunday is skipped, and the process continues by adjusting to the previous weekday to ensure a continuous series of weekday data before prepending.
    *   **Feature Generation**: For each valid weekday `DATE_TIME`:
        *   Appropriate conditional inputs (e.g., cyclical date/time features derived from the generated `DATE_TIME`, noise vectors) are prepared. This leverages logic similar to that in the `FeederPlugin`.
        *   The loaded Generator model is used to predict the primary features of the synthetic sample.
        *   The full set of 51 features is assembled for each sample, combining generated features with derived ones (like date features and any TIs not directly produced by the generator).
5.  **Data Prepending**: The newly generated `n_samples` of synthetic data (now with chronologically earlier, weekday-only `DATE_TIME` values) are prepended to the initially loaded segment from `x_train_file`.
6.  **Output**: The combined dataset (synthetic data + original `x_train_file` segment) is saved to the path specified by `generated_data_file` in the `output_dir`.

**Note**: While the Discriminator model is loaded, its primary role in "generate" mode is typically for potential validation or more advanced generation techniques. The core generation loop relies on the Generator.

### Optimize Mode
- **Purpose**: For hyperparameter tuning of the GAN training process or model architectures.
- **Process**: (Details depend on the specific `OptimizerPlugin` implementation) Typically involves defining a search space, iteratively training with different hyperparameter sets, evaluating, and selecting the best set.
- **Key Configuration**: `operation_mode: "optimize"`.

---

## Configuration Parameters

The system's behavior is highly configurable through parameters defined in the `app/config.py` file. This section outlines the key configuration parameters available for tuning the system's operation.

### General Parameters
- `operation_mode`: Sets the current operation mode of the SDG system. Possible values:
    - `"train"`: Engage the training pipeline for the GAN.
    - `"generate"`: Activate the data generation pipeline using pre-trained models.
    - `"optimize"`: Enable the optimization mode for hyperparameter tuning.
- `random_seed`: An integer value to ensure reproducibility across different runs. It seeds the random number generators used in TensorFlow and Python.

### Train Mode Parameters
- `x_train_file`: File path for the input training data.
- `gan_epochs`: Number of epochs to train the GAN.
- `learning_rate`: Learning rate for the optimizer.
- `batch_size`: Number of samples per gradient update.
- `model_save_path`: Directory where the trained model will be saved.

### Generate Mode Parameters
- `load_generator_sequential_model_file`: File path to the pre-trained full Generator model.
- `load_discriminator_sequential_model_file`: File path to the pre-trained full Discriminator model.
- `n_samples`: Number of synthetic samples to generate.
- `x_train_file`: File path for the input training data, used to determine the prepending point for generated data.
- `max_steps_train`: Maximum number of rows to read from `x_train_file` for determining the initial data segment.
- `dataset_periodicity`: Defines the time interval of the data (e.g., hourly, daily) to correctly space the generated `DATE_TIME` values.

### Optimize Mode Parameters
- `operation_mode`: Must be set to `"optimize"` to activate the optimization pipeline.
- Additional parameters specific to the chosen optimization plugin (e.g., `OptimizerPlugin`) may be available, depending on the implementation.

### Plugin-Specific Parameters
- Each plugin (e.g., `GeneratorPlugin`, `DiscriminatorPlugin`, `OptimizerPlugin`) may have its own set of configuration parameters. Refer to the respective plugin documentation for detailed parameter descriptions.

---

## File Structure and Integration

The SDG system is organized into a modular file structure, promoting separation of concerns and ease of maintenance. This section provides an overview of the key components and their interactions.

### Core Components
- `app/`: Contains the main application code, including:
    - `config.py`: Configuration file for setting parameters and paths.
    - `main.py`: Entry point for running the SDG system.
    - `trainer.py`: Contains the `TrainingCoordinator` class responsible for managing the training process.
    - `generator_plugin.py`: Implements the `GeneratorPlugin` class for synthetic data generation.
    - `discriminator_plugin.py`: Implements the `DiscriminatorPlugin` class for model discrimination.
    - `optimizer_plugin.py`: Implements the `OptimizerPlugin` class for hyperparameter optimization.
- `data/`: Directory for storing input data files (e.g., training data, external data sources).
- `models/`: Directory for saving and loading model checkpoints.
- `results/`: Directory for storing output files, including generated data and model evaluation results.
- `examples/`: Contains example scripts and notebooks for demonstrating system usage.

### Integration and Execution
- The system is designed to be executed as a Python package. The main entry point is the `app/main.py` file.
- Configuration parameters are set in the `app/config.py` file. This includes defining the operation mode (`operation_mode`) and specifying file paths for data and models.
- Depending on the operation mode, different components are activated:
    - **Train Mode**: The `TrainingCoordinator` is initialized, and the `train` method is called to start the GAN training process.
    - **Generate Mode**: The `GeneratorPlugin` is used to load pre-trained models and generate synthetic data, which is then saved to an output file.
    - **Optimize Mode**: The `OptimizerPlugin` is activated to perform hyperparameter tuning based on the defined search space and objective function.

---

## Detailed Code Documentation

The following sections provide detailed documentation of the key classes and functions in the SDG system codebase. This includes descriptions of their responsibilities, usage, and interactions with other components.

### app/config.py

#### Configuration Parameters

- `operation_mode`: (str) The current operation mode of the SDG system. Possible values:
    - `"train"`: Engage the training pipeline for the GAN.
    - `"generate"`: Activate the data generation pipeline using pre-trained models.
    - `"optimize"`: Enable the optimization mode for hyperparameter tuning.
- `random_seed`: (int) An integer value to ensure reproducibility across different runs. It seeds the random number generators used in TensorFlow and Python.

#### Train Mode Parameters

- `x_train_file`: (str) File path for the input training data.
- `gan_epochs`: (int) Number of epochs to train the GAN.
- `learning_rate`: (float) Learning rate for the optimizer.
- `batch_size`: (int) Number of samples per gradient update.
- `model_save_path`: (str) Directory where the trained model will be saved.

#### Generate Mode Parameters

- `load_generator_sequential_model_file`: (str) File path to the pre-trained full Generator model.
- `load_discriminator_sequential_model_file`: (str) File path to the pre-trained full Discriminator model.
- `n_samples`: (int) Number of synthetic samples to generate.
- `x_train_file`: (str) File path for the input training data, used to determine the prepending point for generated data.
- `max_steps_train`: (int) Maximum number of rows to read from `x_train_file` for determining the initial data segment.
- `dataset_periodicity`: (str) Defines the time interval of the data (e.g., hourly, daily) to correctly space the generated `DATE_TIME` values.

#### Optimize Mode Parameters

- `operation_mode`: Must be set to `"optimize"` to activate the optimization pipeline.
- Additional parameters specific to the chosen optimization plugin (e.g., `OptimizerPlugin`) may be available, depending on the implementation.

#### Plugin-Specific Parameters

- Each plugin (e.g., `GeneratorPlugin`, `DiscriminatorPlugin`, `OptimizerPlugin`) may have its own set of configuration parameters. Refer to the respective plugin documentation for detailed parameter descriptions.

### app/main.py

#### Main Entry Point

- The `app/main.py` file is the main entry point for running the SDG system. It initializes the application, loads the configuration, and starts the appropriate pipeline based on the `operation_mode`.

#### Key Functions

- `run_training()`: Initializes the `TrainingCoordinator` and starts the training process in the specified mode (e.g., train, resume).
- `run_generation()`: Executes the generation pipeline, loading pre-trained models and generating synthetic data.
- `run_optimization()`: Activates the optimization pipeline for hyperparameter tuning.

### app/trainer.py

#### TrainingCoordinator Class

- The `TrainingCoordinator` class is responsible for managing the training process of the GAN. It coordinates between the generator and discriminator, handles data loading and preprocessing, and manages the training loop.

#### Key Methods

- `train()`: The main training loop for the GAN. It iteratively trains the generator and discriminator, applies regularization, and updates the models.
- `load_data()`: Loads the training data from the specified file and prepares it for training.
- `save_model()`: Saves the trained model checkpoints (generator, discriminator, and composite GAN model) to the specified directory.

### app/generator_plugin.py

#### GeneratorPlugin Class

- The `GeneratorPlugin` class implements the functionality for synthetic data generation. It loads the pre-trained generator model, prepares the input data, and generates synthetic samples.

#### Key Methods

- `load_model()`: Loads the pre-trained generator model from the specified file.
- `generate()`: Generates synthetic data samples using the loaded generator model and the provided input features.

### app/discriminator_plugin.py

#### DiscriminatorPlugin Class

- The `DiscriminatorPlugin` class implements the functionality for the discriminator model. It loads the pre-trained discriminator model and provides methods for evaluating real and synthetic data.

#### Key Methods

- `load_model()`: Loads the pre-trained discriminator model from the specified file.
- `evaluate()`: Evaluates the provided data (real or synthetic) using the loaded discriminator model.

### app/optimizer_plugin.py

#### OptimizerPlugin Class

- The `OptimizerPlugin` class implements the functionality for hyperparameter optimization. It defines the search space, objective function, and optimization algorithm.

#### Key Methods

- `optimize()`: Performs the hyperparameter optimization, iteratively training and evaluating the model with different hyperparameter settings.

