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
    "encoder_model_file": "examples/results/phase_4_3/phase_4_3_cnn_small_encoder_model.keras", 
    "feeder_feature_columns_for_encoder": [], 
    "feeder_real_data_file_has_header": True,
    "feeder_datetime_col_in_real_data": "DATE_TIME",
    "feeder_date_features_for_conditioning": ["day_of_month", "hour_of_day", "day_of_week", "day_of_year"], # ADDED "day_of_year"
    "feeder_fundamental_features_for_conditioning": ["S&P500_Close", "vix_close"],
    "feeder_max_day_of_month": 31,
    "feeder_max_hour_of_day": 23,
    "feeder_max_day_of_week": 6,
    "feeder_max_day_of_year": 366, # ADDED
    "feeder_context_vector_dim": 64, # CHANGED from 16 to 64 to match Generator/main config
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
        "OPEN", "HIGH", "LOW", "CLOSE", # CLOSE is derived
        "RSI", "MACD", "MACD_Histogram", "MACD_Signal", "EMA",
        "Stochastic_%K", "Stochastic_%D", "ADX", "DI+", "DI-", "ATR", "CCI", "WilliamsR", "Momentum", "ROC",
        "day_of_month_sin", "day_of_month_cos",
        "hour_of_day_sin", "hour_of_day_cos",
        "day_of_week_sin", "day_of_week_cos",
        "day_of_year_sin", "day_of_year_cos",
        "S&P500_Close", "vix_close",
        "BC-BO", "BH-BL",
        # Add all CLOSE tick features that are used by VAE decoder
        "CLOSE_15m_tick_1", "CLOSE_15m_tick_2", "CLOSE_15m_tick_3", "CLOSE_15m_tick_4",
        "CLOSE_15m_tick_5", "CLOSE_15m_tick_6", "CLOSE_15m_tick_7", "CLOSE_15m_tick_8",
        "CLOSE_30m_tick_1", "CLOSE_30m_tick_2", "CLOSE_30m_tick_3", "CLOSE_30m_tick_4",
        "CLOSE_30m_tick_5", "CLOSE_30m_tick_6", "CLOSE_30m_tick_7", "CLOSE_30m_tick_8",
        "day_of_month", "hour_of_day", "day_of_week",
        # Added 7 features based on REFERENCE.md to reach 57 numeric features (excluding DATE_TIME)
        "log_return", 
        "stl_trend", "stl_seasonal", "stl_resid", 
        "wav_approx_L2", "wav_detail_L1", "wav_detail_L2"
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
        "Stochastic_%K", "Stochastic_%D", "ADX", "DI+", "DI-", 
        "ATR", "CCI", "WilliamsR", "Momentum", "ROC",
        # Adding 5 more TIs to reach 20
        "SMA_20", # Simple Moving Average 20 period
        "BB_UPPER", # Bollinger Bands Upper
        "BB_MIDDLE", # Bollinger Bands Middle
        "BB_LOWER", # Bollinger Bands Lower
        "OBV" # On-Balance Volume
    ],
    # Ensure cvae_target_feature_names has 23 features as per REFERENCE.md
    "cvae_target_feature_names": [ # Should be 23 features
        "OPEN", "LOW", "HIGH", "vix_close", "BC-BO", "BH-BL", "S&P500_Close",
        "CLOSE_15m_tick_1", "CLOSE_15m_tick_2", "CLOSE_15m_tick_3", "CLOSE_15m_tick_4",
        "CLOSE_15m_tick_5", "CLOSE_15m_tick_6", "CLOSE_15m_tick_7", "CLOSE_15m_tick_8",
        "CLOSE_30m_tick_1", "CLOSE_30m_tick_2", "CLOSE_30m_tick_3", "CLOSE_30m_tick_4",
        "CLOSE_30m_tick_5", "CLOSE_30m_tick_6", "CLOSE_30m_tick_7", "CLOSE_30m_tick_8"
    ],
    # ...existing code...
    # Ensure feeder_date_features_for_conditioning has 4 base parts for 8 cyclical features
    "feeder_date_features_for_conditioning": ["day_of_month", "hour_of_day", "day_of_week", "day_of_year"],
    # ...existing code...
    "generator_full_feature_names_ordered": [
        "DATE_TIME",
        # 23 Base Features from cvae_target_feature_names
        "OPEN", "LOW", "HIGH", "vix_close", "BC-BO", "BH-BL", "S&P500_Close",
        "CLOSE_15m_tick_1", "CLOSE_15m_tick_2", "CLOSE_15m_tick_3", "CLOSE_15m_tick_4",
        "CLOSE_15m_tick_5", "CLOSE_15m_tick_6", "CLOSE_15m_tick_7", "CLOSE_15m_tick_8",
        "CLOSE_30m_tick_1", "CLOSE_30m_tick_2", "CLOSE_30m_tick_3", "CLOSE_30m_tick_4",
        "CLOSE_30m_tick_5", "CLOSE_30m_tick_6", "CLOSE_30m_tick_7", "CLOSE_30m_tick_8",
        # 8 Cyclical Date Features
        "day_of_month_sin", "day_of_month_cos", "hour_of_day_sin", "hour_of_day_cos",
        "day_of_week_sin", "day_of_week_cos", "day_of_year_sin", "day_of_year_cos",
        # 20 Technical Indicators
        "RSI", "MACD", "MACD_Histogram", "MACD_Signal", "EMA",
        "Stochastic_%K", "Stochastic_%D", "ADX", "DI+", "DI-",
        "ATR", "CCI", "WilliamsR", "Momentum", "ROC",
        "SMA_20", "BB_UPPER", "BB_MIDDLE", "BB_LOWER", "OBV"
    ],
    # ...existing code...
}
