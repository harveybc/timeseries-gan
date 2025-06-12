"""
config.py for SDG (Timeseries GAN Synthetic Data Generator)

This module defines the default values for every command-line parameter
supported by the sdg application. These defaults are used when no value
is provided via CLI, config file, or remote config.
"""

DEFAULT_VALUES = {
    # Plugin selection
    "feeder": "default_feeder",
    "generator": "default_generator",
    "discriminator": "default_discriminator", # Ensure this line exists
    "evaluator": "default_evaluator",
    "optimizer": "default_optimizer",
    "trainer": "gan_trainer", # Ensure this line exists
    "operation_mode": "generate", # Added: Default operation mode, possible values: "train", "generate", "optimize"
    "use_generator_l2_reg": True, # Added: Enable L2 regularization for Generator by default
    "use_evaluation": False, # Added: Enable/disable evaluation phase in generate mode

    # Data for evaluation and base for generation
    # "real_data_file": "examples/data/phase_3/normalized_d4.csv", # REMOVED - Redundant, use x_train_file
    "x_train_file": "examples/data/phase_3/normalized_d4.csv", # Primary data source
    "y_train_file": "examples/data/phase_3/normalized_d4.csv",
    "x_validation_file": "examples/data/phase_3/normalized_d5.csv",
    "y_validation_file": "examples/data/phase_3/normalized_d5.csv",
    "x_test_file": "examples/data/phase_3/normalized_d6.csv",
    "y_test_file": "examples/data/phase_3/normalized_d6.csv",
    "target_column": "CLOSE", 
    "predicted_horizons": [24,48,72,96,120,144],

    "dataset_periodicity": "1h", 

     # Generation parameters
    "n_samples": 12600,
    "max_steps_train": 25200,
    "latent_shape": [18, 32], 
    "batch_size": 32, 
    "seq_len": 144, # ADDED: Corresponds to generator_decoder_input_window_size or expected output sequence length
    "gan_epochs": 2000,  # Added gan_epochs for training
    "gan_batch_size": 32,  # Added gan_batch_size for training
    
    # --- Parameters for FeederPlugin ---
    "feeder_sampling_method": "standard_normal", 
    "feeder_encoder_sampling_technique": "direct", 
    "encoder_model_path": "examples/results/phase_4_3/phase_4_3_cnn_small_encoder_model.keras", 
    "decoder_model_path": "examples/results/phase_4_3/phase_4_3_cnn_small_decoder_model.keras", 
    "feeder_feature_columns_for_encoder": [], 
    "feeder_real_data_file_has_header": True,
    "feeder_datetime_col_in_real_data": "DATE_TIME",
    "feeder_date_features_for_conditioning": ["month", "day_of_year", "day_of_month", "hour_of_day", "day_of_week"], # MODIFIED: 5 date parts for 10 cyclical features
    "feeder_fundamental_features_for_conditioning": [], # MODIFIED: Empty, as generator expects 10 cyclical date features
    "feeder_max_month": 12, # ADDED
    "feeder_max_day_of_year": 366, # ADDED (explicitly, though ConditionManager might default)
    "feeder_max_day_of_month": 31,
    "feeder_max_hour_of_day": 23,
    "feeder_max_day_of_week": 6,
    # "feeder_max_day_of_year": 366, # REMOVED
    "feeder_context_vector_dim": 64, 
    "feeder_context_vector_strategy": "random",
    "feeder_copula_kde_bw_method": None,
    "feeder_noise_dim": 32,  # Added: Noise dimension for feeder plugin

    # --- GAN Training Parameters ---
    "noise_dim": 100,  # Added: Noise dimension for generator input
    "conditional_features_dim": 10,  # CORRECTED: Changed back to 10 to match VAE decoder expectations
    "context_vector_dim": 64,  # Added: Context vector dimension (same as feeder_context_vector_dim)

    # --- Parameters for GeneratorPlugin ---
    # Path to a pre-trained full generator model (if available, otherwise one is built)
    "generator_model_path": None, # Explicitly set to None for building new
    # Path to the VAE DECODER model, used when building a new generator
    "generator_vae_decoder_model_path_param": "examples/results/phase_4_3/phase_4_3_cnn_small_decoder_model.keras",
    # Path to a pre-trained discriminator model (if available, otherwise one is built)
    "discriminator_model_path": None, # Explicitly set to None for building new

    # Default save paths for newly trained models
    "save_generator_sequential_model_file": "examples/results/phase_4_3/phase_4_3_generator_model.keras",
    "save_discriminator_sequential_model_file": "examples/results/phase_4_3/phase_4_3_discriminator_model.keras",
    
    # Default load paths for  trained models for generation
    "load_generator_sequential_model_file": "examples/results/phase_4_3/phase_4_3_generator_model.keras", # MODIFIED: Point to the full generator model
    "load_discriminator_sequential_model_file": "examples/results/phase_4_3/phase_4_3_discriminator_model.keras",

    # Output filename for the generation mode
    "generated_data_file": "examples/results/phase_4_3/normalized_d4_25200_synthetic_12600_prepended_o.csv", # User's value retained

    "generator_decoder_input_window_size": 144,
    
    # 23-Feature Base Architecture Configuration
    "generator_base_feature_names_ordered": [
        "OPEN", "HIGH", "LOW", "CLOSE", 
        "vix_close", "S&P500_Close",
        "BC-BO",
        "CLOSE_15m_tick_1", "CLOSE_15m_tick_2", "CLOSE_15m_tick_3", "CLOSE_15m_tick_4",
        "CLOSE_15m_tick_5", "CLOSE_15m_tick_6", "CLOSE_15m_tick_7", "CLOSE_15m_tick_8",
        "CLOSE_30m_tick_1", "CLOSE_30m_tick_2", "CLOSE_30m_tick_3", "CLOSE_30m_tick_4",
        "CLOSE_30m_tick_5", "CLOSE_30m_tick_6", "CLOSE_30m_tick_7", "CLOSE_30m_tick_8"
    ], 
    "generator_decoder_output_feature_names": [
        # Based on cvae_target_feature_names from REFERENCE.md - exact 23 features
        "OPEN", "HIGH", "LOW", "CLOSE", 
        "vix_close", "S&P500_Close",
        "BC-BO",
        "CLOSE_15m_tick_1", "CLOSE_15m_tick_2", "CLOSE_15m_tick_3", "CLOSE_15m_tick_4",
        "CLOSE_15m_tick_5", "CLOSE_15m_tick_6", "CLOSE_15m_tick_7", "CLOSE_15m_tick_8",
        "CLOSE_30m_tick_1", "CLOSE_30m_tick_2", "CLOSE_30m_tick_3", "CLOSE_30m_tick_4",
        "CLOSE_30m_tick_5", "CLOSE_30m_tick_6", "CLOSE_30m_tick_7", "CLOSE_30m_tick_8"
    ], 
    "generator_ohlc_feature_names": ["OPEN", "HIGH", "LOW", "CLOSE"],
    
    # Post-processing features (calculated after generation, not during GAN training)
    "generator_post_processing_ti_feature_names": [
        "RSI", "MACD", "MACD_Histogram", "MACD_Signal", "EMA",
        "Stochastic_%K", "Stochastic_%D", "ADX", "DI+", "DI-", "ATR", "CCI", "WilliamsR", "Momentum", "ROC"
    ],
    "generator_post_processing_datetime_feature_names": [
        "day_of_month_sin", "day_of_month_cos",
        "hour_of_day_sin", "hour_of_day_cos",
        "day_of_week_sin", "day_of_week_cos",
        "day_of_year_sin", "day_of_year_cos"
    ],
    
    # Configuration for 44-feature architecture  
    "base_features_count": 23,
    "post_processing_features_enabled": True,  # Changed to True to generate full feature sets for compatibility
    "base_features_output_dim": 23,
    "discriminator_input_dim": 23,  # Training: Discriminator expects 23 base features only
    
    # Full feature names ordered to match training data structure (45 features total)
    # Order: DATE_TIME, Technical Indicators (15), OHLC (4), Market Data & Derived Features (26)
    "generator_full_feature_names_ordered": [
        # Technical Indicators first (15 features) - matches training data order
        "RSI", "MACD", "MACD_Histogram", "MACD_Signal", "EMA",
        "Stochastic_%K", "Stochastic_%D", "ADX", "DI+", "DI-", 
        "ATR", "CCI", "WilliamsR", "Momentum", "ROC",
        # OHLC second (4 features) - matches training data order  
        "OPEN", "HIGH", "LOW", "CLOSE",
        # Derived bid/ask spreads (4 features) - matches training data order
        "BC-BO", "BH-BL", "BH-BO", "BO-BL", 
        # External market data (2 features) - matches training data order
        "S&P500_Close", "vix_close",
        # Sub-periodicity data (16 features) - matches training data order
        "CLOSE_15m_tick_1", "CLOSE_15m_tick_2", "CLOSE_15m_tick_3", "CLOSE_15m_tick_4",
        "CLOSE_15m_tick_5", "CLOSE_15m_tick_6", "CLOSE_15m_tick_7", "CLOSE_15m_tick_8",
        "CLOSE_30m_tick_1", "CLOSE_30m_tick_2", "CLOSE_30m_tick_3", "CLOSE_30m_tick_4", 
        "CLOSE_30m_tick_5", "CLOSE_30m_tick_6", "CLOSE_30m_tick_7", "CLOSE_30m_tick_8",
        # Raw date features (3 features) - matches training data order (NOT cyclical sin/cos)
        "day_of_month", "hour_of_day", "day_of_week"
        # Total: 15 + 4 + 4 + 2 + 16 + 3 = 44 features (+ DATE_TIME = 45 columns)
    ],
    
    # Discriminator configuration for 23-feature architecture
    "discriminator_base_feature_names_ordered": [
        "OPEN", "HIGH", "LOW", "CLOSE", 
        "vix_close", "S&P500_Close",
        "BC-BO",
        "CLOSE_15m_tick_1", "CLOSE_15m_tick_2", "CLOSE_15m_tick_3", "CLOSE_15m_tick_4",
        "CLOSE_15m_tick_5", "CLOSE_15m_tick_6", "CLOSE_15m_tick_7", "CLOSE_15m_tick_8",
        "CLOSE_30m_tick_1", "CLOSE_30m_tick_2", "CLOSE_30m_tick_3", "CLOSE_30m_tick_4",
        "CLOSE_30m_tick_5", "CLOSE_30m_tick_6", "CLOSE_30m_tick_7", "CLOSE_30m_tick_8"
    ],
    "discriminator_ohlc_feature_names": ["OPEN", "HIGH", "LOW", "CLOSE"],

    # Training parameters
    "learning_rate": 0.001,
    "beta1": 0.5,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "amsgrad": False,
    "weight_decay": 0.0,
    "clip_gradients": True,
    "max_grad_norm": 10.0,
    "loss": "mse", # Mean Squared Error

    # Logging and output 
    "log_interval": 100,
    "save_model_interval": 5000,
    "output_dir": "examples/results/phase_4_3/",
    "log_dir": "examples/logs/phase_4_3/",
    "tensorboard_log_dir": "examples/logs/phase_4_3/tensorboard/",
    "wandb_project": "timeseries-gan",
    "wandb_entity": "your_wandb_entity", # CHANGE THIS to your Weights & Biases entity
    "wandb_run_name": "run-1",

    # Miscellaneous
    "seed": 42,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "shuffle": True,
    "drop_last": True,
    "prefetch_factor": 2,
    "batch_size_eval": 64,
    "seq_len_eval": 288,
    "max_steps_eval": 1000,
    "eval_metric": "mse",
    "early_stopping_patience": 120,
    "checkpointing": True,
    "resume_from_checkpoint": False,
    "fp16": False,
    "tf32": True,
    "autocast": True,
    

    # Added: L2 Regularization for Generator
    "l2_regularization": 1e-5,

    # Added: Discriminator Architecture Parameters - Updated for 44-feature architecture
    "discriminator_conv_filters": [32, 16, 8],       # Decreasing filter sizes for proper dimensionality reduction
    "discriminator_conv_kernel_size": 5,             # Larger kernel for better feature extraction
    "discriminator_conv_strides": [2, 2, 2],         # Stride=2 for downsampling at each layer
    "discriminator_lstm_units": 32,                  # Reduced LSTM units 
    "discriminator_dense_units": [16, 8],            # Decreasing dense layer sizes
    "discriminator_dropout_rate": 0.3,
    "num_features": 23,                              # Training: Use 23 base features for GAN training

    # Added: MMD (Maximum Mean Discrepancy) Loss Parameters
    "enable_mmd_loss": True,          # Enable/disable MMD loss
    "mmd_lambda_g": 0.0001,            # Weight for MMD loss in generator
    "mmd_lambda_d": 0.00001,           # Weight for MMD loss in discriminator (optional)
    "mmd_gamma": None,               # RBF kernel bandwidth (None = auto)
    "mmd_sample_size": 128,          # Number of samples for MMD computation (performance)

    # Added: ReduceLROnPlateau parameters
    "enable_reduce_lr_on_plateau": True,
    "lr_monitor_metric_g": "g_loss", # Metric for generator's LR scheduler
    "lr_monitor_metric_d": "d_loss", # Metric for discriminator's LR scheduler
    "lr_reduction_factor": 0.5,   # Factor by which LR is reduced
    "lr_patience": 30,            # Epochs with no improvement before LR is reduced
    "lr_min_delta": 1e-7,       # Minimum change to qualify as an improvement
    "min_lr_g": 1e-7,             # Minimum LR for generator
    "min_lr_d": 1e-7,             # Minimum LR for discriminator
}
