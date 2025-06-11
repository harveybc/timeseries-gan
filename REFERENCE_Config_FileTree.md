# REFERENCE_Config_FileTree.md

## Configuration Parameters (`app/config.py`)

This section details the configuration parameters used in the `timeseries-gan` project, as defined in `app/config.py`. These parameters control various aspects of the data generation, model training, and evaluation processes.

### Plugin Selection
-   `"feeder"`: `"default_feeder"` (Plugin for providing input data/noise to the generator)
-   `"generator"`: `"default_generator"` (Plugin for the GAN's generator model)
-   `"discriminator"`: `"default_discriminator"` (Plugin for the GAN's discriminator model)
-   `"evaluator"`: `"default_evaluator"` (Plugin for evaluating generated data)
-   `"optimizer"`: `"default_optimizer"` (Plugin for hyperparameter optimization, if used)
-   `"trainer"`: `"gan_trainer"` (Plugin for managing the GAN training process)
-   `"operation_mode"`: `"train"` (Default operation mode: "train", "generate", or "optimize")
-   `"use_generator_l2_reg"`: `True` (Enable L2 regularization for the Generator by default)

### Data Configuration
-   `"x_train_file"`: `"examples/data/phase_3/normalized_d4.csv"` (Primary data source for training)
-   `"y_train_file"`: `"examples/data/phase_3/normalized_d4.csv"` (Target labels for training, often same as x_train_file for GANs)
-   `"x_validation_file"`: `"examples/data/phase_3/normalized_d5.csv"` (Validation data)
-   `"y_validation_file"`: `"examples/data/phase_3/normalized_d5.csv"` (Validation labels)
-   `"x_test_file"`: `"examples/data/phase_3/normalized_d6.csv"` (Test data)
-   `"y_test_file"`: `"examples/data/phase_3/normalized_d6.csv"` (Test labels)
-   `"target_column"`: `"CLOSE"` (Target column name in the dataset)
-   `"predicted_horizons"`: `[24, 48, 72, 96, 120, 144]` (Prediction horizons for forecasting tasks, if applicable)
-   `"dataset_periodicity"`: `"1h"` (Frequency of the time series data, e.g., "1h" for hourly)

### Generation and Model Parameters
-   `"n_samples"`: `12600` (Number of synthetic samples to generate)
-   `"max_steps_train"`: `25200` (Maximum number of real data rows for initial conditioning or training length)
-   `"latent_shape"`: `[18, 32]` (Shape of the latent space vector: [sequence_length, latent_dim])
-   `"batch_size"`: `32` (General batch size for data processing)
-   `"seq_len"`: `144` (Sequence length for model inputs/outputs, corresponds to `generator_decoder_input_window_size`)
-   `"gan_epochs"`: `100` (Number of epochs for GAN training)
-   `"gan_batch_size"`: `32` (Batch size specifically for GAN training)
-   `"noise_dim"`: `100` (Dimension of the noise vector input to the generator)
-   `"conditional_features_dim"`: `10` (Dimension of the conditional features vector for the generator)
-   `"context_vector_dim"`: `64` (Dimension of the context vector used for sequential generation)

### Feeder Plugin Parameters (`FeederPlugin`)
-   `"feeder_sampling_method"`: `"standard_normal"` (Method for sampling noise, e.g., "standard_normal")
-   `"feeder_encoder_sampling_technique"`: `"direct"` (Technique for using encoder output, e.g., "direct", "kde")
-   `"encoder_model_path"`: `"examples/results/phase_4_3/phase_4_3_cnn_small_encoder_model.keras"` (Path to the pre-trained encoder model)
-   `"feeder_feature_columns_for_encoder"`: `[]` (List of feature columns from real data to feed into the encoder)
-   `"feeder_real_data_file_has_header"`: `True` (Boolean indicating if the real data CSV has a header row)
-   `"feeder_datetime_col_in_real_data"`: `"DATE_TIME"` (Name of the datetime column in the real data)
-   `"feeder_date_features_for_conditioning"`: `["day_of_month", "hour_of_day", "day_of_week"]` (Date/time components to use for conditional input)
-   `"feeder_fundamental_features_for_conditioning"`: `["S&P500_Close", "vix_close"]` (Fundamental features used for conditioning)
-   `"feeder_max_day_of_month"`: `31`
-   `"feeder_max_hour_of_day"`: `23`
-   `"feeder_max_day_of_week"`: `6`
-   `"feeder_context_vector_dim"`: `64` (Dimension of the context vector provided by the feeder, should match `context_vector_dim`)
-   `"feeder_context_vector_strategy"`: `"random"` (Strategy for generating the initial context vector, e.g., "random", "zero")
-   `"feeder_copula_kde_bw_method"`: `None` (Bandwidth method for KDE if using copula-based sampling)
-   `"feeder_noise_dim"`: `32` (Noise dimension for the feeder plugin, potentially different from GAN's main `noise_dim` if feeder generates structured noise for VAE's Z-space)

### Generator Plugin Parameters (`GeneratorPlugin`)
-   `"generator_sequential_model_file"`: `"examples/results/phase_4_3/phase_4_3_cnn_small_decoder_model.keras"` (Path to the pre-trained VAE decoder model, used as part of the composite generator)
-   `"discriminator_sequential_model_file"`: `"examples/results/phase_4_3/phase_4_3_discriminator_model.keras"` (Path to a pre-trained discriminator model, if available/used)
-   `"save_generator_sequential_model_file"`: `"examples/results/phase_4_3/phase_4_3_cnn_small_decoder_model.keras"` (Path to save the trained generator model)
-   `"save_discriminator_sequential_model_file"`: `"examples/results/phase_4_3/phase_4_3_discriminator_model.keras"` (Path to save the trained discriminator model)
-   `"generator_decoder_input_window_size"`: `144` (Input window size for the VAE decoder component)
-   `"generator_full_feature_names_ordered"`: List of 51 feature names in the exact order expected for the final generated output. Includes OHLC, technical indicators, cyclical date/time features, fundamental data, and placeholders.
    *   Example: `["DATE_TIME", "OPEN", ..., "Market_Volatility_Idx"]`
-   `"generator_decoder_output_feature_names"`: List of 23 features that the VAE decoder part of the generator is expected to output. These are typically core features from which technical indicators are derived.
    *   Example: `["OPEN", "LOW", ..., "CLOSE_30m_tick_8"]`
-   `"generator_ohlc_feature_names"`: `["OPEN", "HIGH", "LOW", "CLOSE"]` (Names of OHLC features)
-   `"generator_ti_feature_names"`: List of 15 technical indicator names.
    *   Example: `["RSI", "MACD", ..., "ROC"]`
-   `"generator_l2_reg"`: `0.01` (L2 regularization factor for generator layers if `use_generator_l2_reg` is `True`)

### Discriminator Plugin Parameters (`DiscriminatorPlugin`)
-   `"discriminator_full_feature_names_ordered"`: List of 51 feature names in the exact order the discriminator expects as input. This should match `generator_full_feature_names_ordered`.
    *   Example: `["DATE_TIME", "OPEN", ..., "Market_Volatility_Idx"]`
-   `"discriminator_ohlc_feature_names"`: `["OPEN", "HIGH", "LOW", "CLOSE"]`
-   `"discriminator_ti_feature_names"`: List of 15 technical indicator names.

### Training Parameters (General & Optimizer)
-   `"learning_rate"`: `0.0002` (Default learning rate, can be overridden by specific G/D LR)
-   `"beta1"`: `0.5` (Adam optimizer beta1)
-   `"beta2"`: `0.999` (Adam optimizer beta2)
-   `"epsilon"`: `1e-8` (Adam optimizer epsilon)
-   `"amsgrad"`: `False` (Adam optimizer amsgrad)
-   `"weight_decay"`: `0.0` (Optimizer weight decay)
-   `"clip_gradients"`: `True` (Enable gradient clipping)
-   `"max_grad_norm"`: `10.0` (Maximum gradient norm for clipping)
-   `"loss"`: `"mse"` (Default loss function, e.g., "mse", "binary_crossentropy")

### ReduceLROnPlateau Parameters (Learning Rate Scheduling)
-   `"enable_reduce_lr_on_plateau"`: `True` (Enable ReduceLROnPlateau callback for G and D)
-   `"lr_monitor_metric_g"`: `"g_loss"` (Metric to monitor for generator's LR scheduler)
-   `"lr_monitor_metric_d"`: `"d_loss"` (Metric to monitor for discriminator's LR scheduler)
-   `"lr_reduction_factor"`: `0.1` (Factor by which LR is reduced, new_lr = lr * factor)
-   `"lr_patience"`: `10` (Epochs with no improvement before LR is reduced)
-   `"lr_min_delta"`: `0.0001` (Minimum change in monitored quantity to qualify as an improvement)
-   `"min_lr_g"`: `1e-7` (Minimum learning rate for the generator)
-   `"min_lr_d"`: `1e-7` (Minimum learning rate for the discriminator)
    *Note: Specific learning rates for generator and discriminator, like `generator_lr` and `discriminator_lr`, are typically set directly in `GANTrainerPlugin` or passed via CLI, overriding the general `learning_rate`.*

### Logging and Output
-   `"log_interval"`: `100` (Logging frequency, e.g., every 100 batches/steps)
-   `"save_model_interval"`: `5000` (Model saving frequency during training)
-   `"output_dir"`: `"examples/results/phase_4_3/"` (Directory for saving results like generated data, models)
-   `"log_dir"`: `"examples/logs/phase_4_3/"` (Directory for saving log files)
-   `"tensorboard_log_dir"`: `"examples/logs/phase_4_3/tensorboard/"` (Directory for TensorBoard logs)
-   `"wandb_project"`: `"timeseries-gan"` (Weights & Biases project name)
-   `"wandb_entity"`: `"your_wandb_entity"` (Weights & Biases entity - **USER SHOULD CHANGE THIS**)
-   `"wandb_run_name"`: `"run-1"` (Weights & Biases run name)

### Miscellaneous
-   `"seed"`: `42` (Random seed for reproducibility)
-   `"num_workers"`: `4` (Number of worker threads for data loading)
-   `"pin_memory"`: `True` (For CUDA memory pinning)
-   `"persistent_workers"`: `True` (Keep data loader workers alive)
-   `"shuffle"`: `True` (Shuffle data during training)
-   `"drop_last"`: `True` (Drop the last incomplete batch)
-   `"prefetch_factor"`: `2` (Data prefetching factor for data loaders)
-   `"batch_size_eval"`: `64` (Batch size for evaluation)
-   `"seq_len_eval"`: `288` (Sequence length for evaluation)
-   `"max_steps_eval"`: `1000` (Maximum steps for evaluation)
-   `"eval_metric"`: `"mse"` (Primary metric for evaluation)
-   `"early_stopping_patience"`: `10` (Patience for early stopping, if enabled)
-   `"checkpointing"`: `True` (Enable model checkpointing)
-   `"resume_from_checkpoint"`: `False` (Resume training from a checkpoint)
-   `"fp16"`: `False` (Use 16-bit floating point precision)
-   `"tf32"`: `True` (Allow TensorFlow 32-bit operations on Ampere GPUs)
-   `"autocast"`: `True` (Enable automatic mixed precision)

---

## Project File Structure (`timeseries-gan/`)

This provides an overview of the main directories and key files within the `timeseries-gan` project.

```
timeseries-gan/
├── app/
│   ├── __init__.py
│   ├── config.py               # Default configuration parameters
│   ├── main.py                 # Main application entry point
│   ├── plugin_loader.py        # Loads plugins
│   └── utils.py                # Utility functions
│
├── examples/
│   ├── data/                   # Example datasets (e.g., CSV files)
│   │   └── phase_3/
│   │       ├── normalized_d4.csv
│   │       └── ...
│   ├── logs/                   # Directory for log files and TensorBoard data
│   │   └── phase_4_3/
│   │       └── tensorboard/
│   └── results/                # Directory for output files (models, generated data)
│       └── phase_4_3/
│           ├── phase_4_3_cnn_small_decoder_model.keras
│           ├── phase_4_3_cnn_small_encoder_model.keras
│           └── ...
│
├── tsg_plugins/                # Timeseries GAN specific plugins
│   ├── __init__.py
│   ├── default_feeder_plugin/
│   │   ├── __init__.py
│   │   └── default_feeder_plugin.py
│   ├── discriminator_plugin/
│   │   ├── __init__.py
│   │   └── discriminator_plugin.py
│   ├── gan_trainer_plugin/
│   │   ├── __init__.py
│   │   ├── gan_trainer_plugin.py # Core GAN training logic
│   │   └── training_coordinator.py # Manages GAN training steps
│   ├── generator_plugin/
│   │   ├── __init__.py
│   │   └── generator_plugin.py   # Composite generator model
│   └── ... (other potential plugins like evaluators)
│
├── tests/                      # Unit and integration tests
│   ├── __init__.py
│   └── ...
│
├── .gitignore
├── LICENSE
├── pyproject.toml              # Project metadata and build configuration
├── README.md                   # Main project README
├── REFERENCE.md                # Main reference documentation (this file)
├── requirements.txt            # Python package dependencies
├── sdg.py                      # Script to run the SDG application (symlink or wrapper for app.main)
├── setup.py                    # Python package setup script
└── ... (other project files like Dockerfile, CI configurations, etc.)
```

**Key Components:**

*   **`app/`**: Core application logic, including the main entry point (`main.py`) and configuration (`config.py`).
*   **`tsg_plugins/`**: Contains all custom plugins for the timeseries GAN, such as:
    *   `default_feeder_plugin/`: Handles data input, noise generation, and conditioning.
    *   `generator_plugin/`: Defines the composite generator model (BiLSTM Z-gen + VAE Decoder).
    *   `discriminator_plugin/`: Defines the discriminator model.
    *   `gan_trainer_plugin/`: Orchestrates the GAN training process, including the `TrainingCoordinator`.
*   **`examples/`**: Provides example data, pre-trained models, and a place for output results and logs.
*   **`tests/`**: Houses automated tests for the project.
*   **Root Files**: Includes project setup (`setup.py`, `pyproject.toml`), dependencies (`requirements.txt`), and primary documentation (`README.md`, `REFERENCE.md`).

This structure promotes modularity and separation of concerns, making the project easier to understand, maintain, and extend.
