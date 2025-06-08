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
    
    # --- Parameters for FeederPlugin ---
    "feeder_sampling_method": "standard_normal", 
    "feeder_encoder_sampling_technique": "direct", 
    "encoder_model_path": "examples/results/phase_4_3/phase_4_3_cnn_small_encoder_model.keras", 
    "feeder_feature_columns_for_encoder": [], 
    "feeder_real_data_file_has_header": True,
    "feeder_datetime_col_in_real_data": "DATE_TIME",
    "feeder_date_features_for_conditioning": ["day_of_month", "hour_of_day", "day_of_week"], # MODIFIED
    "feeder_fundamental_features_for_conditioning": ["S&P500_Close", "vix_close"],
    "feeder_max_day_of_month": 31,
    "feeder_max_hour_of_day": 23,
    "feeder_max_day_of_week": 6,
    # "feeder_max_day_of_year": 366, # REMOVED
    "feeder_context_vector_dim": 64, 
    "feeder_context_vector_strategy": "random",
    "feeder_copula_kde_bw_method": None,

    # --- Parameters for GeneratorPlugin ---
    "generator_sequential_model_file": "examples/results/phase_4_3/phase_4_3_cnn_small_decoder_model.keras",
    "discriminator_sequential_model_file": "examples/results/phase_4_3/phase_4_3_discriminator_model.keras",
    "save_generator_sequential_model_file": "examples/results/phase_4_3/phase_4_3_cnn_small_decoder_model.keras",
    "save_discriminator_sequential_model_file": "examples/results/phase_4_3/phase_4_3_discriminator_model.keras",
    
    "generator_decoder_input_window_size": 144, 
    "generator_full_feature_names_ordered": [
        "DATE_TIME", 
        "OPEN", "HIGH", "LOW", "CLOSE", 
        "RSI", "MACD", "MACD_Histogram", "MACD_Signal", "EMA",
        "Stochastic_%K", "Stochastic_%D", "ADX", "DI+", "DI-", "ATR", "CCI", "WilliamsR", "Momentum", "ROC",
        "day_of_month_sin", "day_of_month_cos",
        "hour_of_day_sin", "hour_of_day_cos",
        "day_of_week_sin", "day_of_week_cos",
        "day_of_year_sin", "day_of_year_cos", # Added back for 51 features
        "S&P500_Close", "vix_close",
        "BC-BO", "BH-BL",
        "CLOSE_15m_tick_1", "CLOSE_15m_tick_2", "CLOSE_15m_tick_3", "CLOSE_15m_tick_4",
        "CLOSE_15m_tick_5", "CLOSE_15m_tick_6", "CLOSE_15m_tick_7", "CLOSE_15m_tick_8",
        "CLOSE_30m_tick_1", "CLOSE_30m_tick_2", "CLOSE_30m_tick_3", "CLOSE_30m_tick_4",
        "CLOSE_30m_tick_5", "CLOSE_30m_tick_6", "CLOSE_30m_tick_7", "CLOSE_30m_tick_8"
        # Removed raw date parts: "day_of_month", "hour_of_day", "day_of_week" 
    ], 
    "generator_decoder_output_feature_names": [
        # Based on cvae_target_feature_names from REFERENCE.md - exact 23 features
        "OPEN", "LOW", "HIGH", "vix_close", 
        "BC-BO", "BH-BL", 
        "S&P500_Close",
        "CLOSE_15m_tick_1", "CLOSE_15m_tick_2", "CLOSE_15m_tick_3", "CLOSE_15m_tick_4",
        "CLOSE_15m_tick_5", "CLOSE_15m_tick_6", "CLOSE_15m_tick_7", "CLOSE_15m_tick_8",
        "CLOSE_30m_tick_1", "CLOSE_30m_tick_2", "CLOSE_30m_tick_3", "CLOSE_30m_tick_4",
        "CLOSE_30m_tick_5", "CLOSE_30m_tick_6", "CLOSE_30m_tick_7", "CLOSE_30m_tick_8"
    ], 
    "generator_ohlc_feature_names": ["OPEN", "HIGH", "LOW", "CLOSE"],
    "generator_ti_feature_names": [
        "RSI", "MACD", "MACD_Histogram", "MACD_Signal", "EMA",
        "Stochastic_%K", "Stochastic_%D", "ADX", "DI+", "DI-", "ATR", "CCI", "WilliamsR", "Momentum", "ROC"
    ],
    "discriminator_full_feature_names_ordered": [
        "DATE_TIME", 
        "OPEN", "HIGH", "LOW", "CLOSE", 
        "RSI", "MACD", "MACD_Histogram", "MACD_Signal", "EMA",
        "Stochastic_%K", "Stochastic_%D", "ADX", "DI+", "DI-", "ATR", "CCI", "WilliamsR", "Momentum", "ROC",
        "day_of_month_sin", "day_of_month_cos",
        "hour_of_day_sin", "hour_of_day_cos",
        "day_of_week_sin", "day_of_week_cos",
        "day_of_year_sin", "day_of_year_cos", # Added back for 51 features
        "S&P500_Close", "vix_close",
        "BC-BO", "BH-BL",
        "CLOSE_15m_tick_1", "CLOSE_15m_tick_2", "CLOSE_15m_tick_3", "CLOSE_15m_tick_4",
        "CLOSE_15m_tick_5", "CLOSE_15m_tick_6", "CLOSE_15m_tick_7", "CLOSE_15m_tick_8",
        "CLOSE_30m_tick_1", "CLOSE_30m_tick_2", "CLOSE_30m_tick_3", "CLOSE_30m_tick_4",
        "CLOSE_30m_tick_5", "CLOSE_30m_tick_6", "CLOSE_30m_tick_7", "CLOSE_30m_tick_8"
    ],
    "discriminator_ohlc_feature_names": ["OPEN", "HIGH", "LOW", "CLOSE"],
    "discriminator_ti_feature_names": [
        "RSI", "MACD", "MACD_Histogram", "MACD_Signal", "EMA",
        "Stochastic_%K", "Stochastic_%D", "ADX", "DI+", "DI-", "ATR", "CCI", "WilliamsR", "Momentum", "ROC"
    ],

    # Training parameters
    "learning_rate": 0.0002,
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
    "early_stopping_patience": 10,
    "checkpointing": True,
    "resume_from_checkpoint": False,
    "fp16": False,
    "tf32": True,
    "autocast": True
}
