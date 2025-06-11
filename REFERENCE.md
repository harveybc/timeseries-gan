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

