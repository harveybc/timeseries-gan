"""
cli.py for SDG (Synthetic Data Generator)

This module defines the command-line interface for the sdg application,
including arguments for feeder, generator, evaluator, optimizer plugins,
generation and evaluation parameters, and remote configuration/logging.
"""

import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="SDG: Synthetic Data Generator with plugin-based architecture."
    )

    # Plugin selection
    parser.add_argument('--feeder', type=str, help='Feeder plugin name')
    parser.add_argument('--generator', type=str, help='Generator plugin name')
    parser.add_argument('--evaluator', type=str, help='Evaluator plugin name')
    parser.add_argument('--optimizer', type=str, help='Optimizer plugin name')
    parser.add_argument('--preprocessor_plugin', type=str, help='Preprocessor plugin name')
    parser.add_argument('--trainer', type=str, help='Trainer plugin name')

    # Data file paths
    parser.add_argument('--x_train_file', type=str, help='Path to training input CSV')
    parser.add_argument('--y_train_file', type=str, help='Path to training target CSV')
    parser.add_argument('--x_validation_file', type=str, help='Path to validation input CSV')
    parser.add_argument('--y_validation_file', type=str, help='Path to validation target CSV')
    parser.add_argument('--x_test_file', type=str, help='Path to test input CSV')
    parser.add_argument('--y_test_file', type=str, help='Path to test target CSV')
    parser.add_argument('--output_file', type=str, help='Path for synthetic output CSV')
    parser.add_argument('--metrics_file', type=str, help='Path for evaluation metrics JSON')
    parser.add_argument('--save_config', type=str, help='Path to save merged configuration JSON')
    parser.add_argument('--save_log', type=str, help='Path to save debug log JSON')

    # Generation parameters
    parser.add_argument('--n_samples', type=int, help='Number of synthetic samples to generate')
    parser.add_argument('--max_steps_train', type=int, help='Number of real rows to prepend')
    parser.add_argument('--seq_len', type=int, help='Sequence length for VAE decoder')
    parser.add_argument('--batch_size', type=int, help='Batch size for training/generation')
    parser.add_argument('--latent_shape', type=json.loads,
                        help='JSON list for latent_shape e.g. "[seq_len, latent_dim]"')

    # Feeder plugin parameters
    parser.add_argument('--feeder_sampling_method', type=str)
    parser.add_argument('--feeder_encoder_sampling_technique', type=str)
    parser.add_argument('--encoder_model_file', type=str)
    parser.add_argument('--feeder_feature_columns_for_encoder', type=json.loads)
    parser.add_argument('--feeder_real_data_file_has_header', type=json.loads)
    parser.add_argument('--feeder_datetime_col_in_real_data', type=str)
    parser.add_argument('--feeder_date_features_for_conditioning', type=json.loads)
    parser.add_argument('--feeder_fundamental_features_for_conditioning', type=json.loads)
    parser.add_argument('--feeder_max_day_of_month', type=int)
    parser.add_argument('--feeder_max_hour_of_day', type=int)
    parser.add_argument('--feeder_max_day_of_week', type=int)
    parser.add_argument('--feeder_max_day_of_year', type=int)
    parser.add_argument('--feeder_context_vector_dim', type=int)
    parser.add_argument('--feeder_context_vector_strategy', type=str)
    parser.add_argument('--feeder_copula_kde_bw_method', type=json.loads)

    # Generator (decoder) plugin parameters
    parser.add_argument('--generator_sequential_model_file', type=str)
    parser.add_argument('--discriminator_sequential_model_file', type=str)
    parser.add_argument('--save_generator_sequential_model_file', type=str)
    parser.add_argument('--save_discriminator_sequential_model_file', type=str)
    parser.add_argument('--generator_decoder_input_window_size', type=int)
    parser.add_argument('--generator_full_feature_names_ordered', type=json.loads)
    parser.add_argument('--generator_decoder_output_feature_names', type=json.loads)
    parser.add_argument('--generator_ohlc_feature_names', type=json.loads)
    parser.add_argument('--generator_ti_feature_names', type=json.loads)
    parser.add_argument('--generator_date_conditional_feature_names', type=json.loads)
    parser.add_argument('--generator_feeder_conditional_feature_names', type=json.loads)
    parser.add_argument('--generator_ti_calculation_min_lookback', type=int)
    parser.add_argument('--generator_ti_params', type=json.loads)
    parser.add_argument('--generator_normalization_params_file', type=str)
    parser.add_argument('--generator_decoder_input_name_latent', type=str)
    parser.add_argument('--generator_decoder_input_name_window', type=str)
    parser.add_argument('--generator_decoder_input_name_conditions', type=str)
    parser.add_argument('--generator_decoder_input_name_context', type=str)
    parser.add_argument('--context_vector_dim', type=int)

    # Discriminator/trainer parameters
    parser.add_argument('--base_feature_names_ordered', type=json.loads)
    parser.add_argument('--feature_names_for_discriminator_ordered', type=json.loads)
    parser.add_argument('--gan_epochs', type=int)
    parser.add_argument('--gan_batch_size', type=int)
    parser.add_argument('--generator_lr', type=float)
    parser.add_argument('--generator_beta1', type=float)
    parser.add_argument('--discriminator_lr', type=float)
    parser.add_argument('--discriminator_beta1', type=float)
    parser.add_argument('--discriminator_lstm_units', type=int)
    parser.add_argument('--discriminator_dense_units', type=int)
    parser.add_argument('--gan_save_interval', type=int)
    parser.add_argument('--gan_model_dir', type=str)
    parser.add_argument('--gan_loss_plot_file', type=str)

    # Evaluator plugin parameters
    parser.add_argument('--evaluator_metrics', type=json.loads)
    parser.add_argument('--evaluator_mmd_gamma', type=json.loads)
    parser.add_argument('--evaluator_acf_nlags', type=int)
    parser.add_argument('--evaluator_predictive_model_type', type=str)
    parser.add_argument('--evaluator_predictive_epochs', type=int)
    parser.add_argument('--evaluator_predictive_batch_size', type=int)
    parser.add_argument('--evaluator_plot_max_features', type=int)
    parser.add_argument('--evaluator_plot_max_lags_acf', type=int)

    # Optimizer plugin parameters
    parser.add_argument('--hyperparameter_optimization_mode', type=json.loads)
    parser.add_argument('--run_hyperparameter_optimization', type=json.loads)
    parser.add_argument('--population_size', type=int)
    parser.add_argument('--n_generations', type=int)
    parser.add_argument('--cxpb', type=float)
    parser.add_argument('--mutpb', type=float)
    parser.add_argument('--hyperparameter_bounds', type=json.loads)
    parser.add_argument('--optimizer_n_samples_per_eval', type=int)
    parser.add_argument('--optimizer_start_datetime', type=str)

    # General execution parameters
    parser.add_argument('--random_seed', type=int)
    parser.add_argument('--num_synthetic_samples_to_generate', type=int)
    parser.add_argument('--start_datetime', type=str)
    parser.add_argument('--quiet_mode', action='store_true')
    parser.add_argument('--datetime_col_name', type=str)
    parser.add_argument('--target_column_order', type=json.loads)
    parser.add_argument('--num_base_features_generated', type=int)
    parser.add_argument('--gan_training_mode', type=json.loads)

    return parser.parse_known_args()
