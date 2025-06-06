#!/usr/bin/env python3
"""
main.py

Entry point for TimeSeries-GAN: High-level orchestration that loads and merges 
configurations, initializes plugins, then dispatches to the unified pipeline.
"""

import sys
import traceback
from typing import Dict, Any

from app.cli import parse_args
from app.config import DEFAULT_VALUES
from app.config_handler import load_config, remote_load_config, save_config
from app.plugin_loader import load_plugin
from app.config_merger import merge_config, process_unknown_args
from app.data_processor import run_pipeline


def main():
    """
    High-level orchestration: Parse CLI args, load/merge configurations, 
    initialize plugins, then dispatch to unified pipeline.
    """
    print("Starting TimeSeries-GAN pipeline...")
    
    # Parse CLI arguments
    print("Parsing command line arguments...")
    args, unknown_args = parse_args()
    cli_args: Dict[str, Any] = vars(args)
    
    # Load configuration from file if specified
    print("Loading configuration...")
    config: Dict[str, Any] = DEFAULT_VALUES.copy()
    file_config: Dict[str, Any] = {}
    
    if hasattr(args, 'remote_load_config') and args.remote_load_config:
        try:
            file_config = remote_load_config(args.remote_load_config)
            print(f"Remote configuration loaded from {args.remote_load_config}")
        except Exception as e:
            print(f"Failed to load remote configuration: {e}")
    elif hasattr(args, 'load_config') and args.load_config:
        try:
            file_config = load_config(args.load_config)
            print(f"Configuration loaded from {args.load_config}")
        except Exception as e:
            print(f"Failed to load configuration file: {e}")
    
    # Process unknown arguments
    unknown_args_dict = process_unknown_args(unknown_args)
    
    # Initial configuration merge
    current_config = merge_config(config, {}, {}, file_config, cli_args, unknown_args_dict)
    
    # Load and initialize plugins
    print("Loading and initializing plugins...")
    plugins = load_and_initialize_plugins(current_config)
    
    # Merge plugin configurations
    print("Merging plugin configurations...")
    current_config = merge_plugin_configurations(current_config, plugins, file_config, cli_args, unknown_args_dict)
    
    # Set final parameters for all plugins
    print("Setting final parameters for all plugins...")
    for plugin in plugins.values():
        if plugin:
            plugin.set_params(**current_config)
    
    # Run the unified pipeline
    print("Dispatching to unified pipeline...")
    run_pipeline(current_config, **plugins)
    
    # Save final configuration if requested
    if hasattr(args, 'save_config') and args.save_config:
        try:
            save_config(current_config, args.save_config)
            print(f"Final configuration saved to {args.save_config}")
        except Exception as e:
            print(f"Failed to save local configuration: {e}")
    
    print("Pipeline execution completed.")


def load_and_initialize_plugins(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load and initialize all required plugins based on configuration.
    
    Args:
        config: Current configuration dictionary
        
    Returns:
        Dictionary containing initialized plugin instances
    """
    plugins = {
        'feeder_plugin': None,
        'generator_plugin': None,
        'discriminator_plugin': None,
        'evaluator_plugin': None,
        'optimizer_plugin': None,
        'preprocessor_plugin': None,
        'trainer_plugin': None
    }
    
    # Load Feeder Plugin
    plugin_name_feeder = config.get('feeder', 'default_feeder')
    print(f"Loading Feeder Plugin: {plugin_name_feeder}")
    try:
        feeder_class, _ = load_plugin('feeder.plugins', plugin_name_feeder)
        plugins['feeder_plugin'] = feeder_class(config)
        print(f"✓ Feeder Plugin '{plugin_name_feeder}' loaded successfully")
    except Exception as e:
        print(f"Failed to load Feeder Plugin '{plugin_name_feeder}': {e}")
        traceback.print_exc()
        # Continue without feeder for now

    # Load Generator Plugin
    plugin_name_generator = config.get('generator', 'default_generator')
    print(f"Loading Generator Plugin: {plugin_name_generator}")
    try:
        generator_class, _ = load_plugin('generator.plugins', plugin_name_generator)
        plugins['generator_plugin'] = generator_class(config)
        print(f"✓ Generator Plugin '{plugin_name_generator}' loaded successfully")
    except Exception as e:
        print(f"Failed to load Generator Plugin '{plugin_name_generator}': {e}")
        traceback.print_exc()
        # Try to load default implementation
        try:
            from tsg_plugins.generator_plugin.generator_plugin import GeneratorPlugin
            plugins['generator_plugin'] = GeneratorPlugin(config)
            print("✓ Default Generator Plugin loaded successfully")
        except Exception as e2:
            print(f"Failed to load default Generator Plugin: {e2}")

    # Load Discriminator Plugin (BEFORE Trainer)
    plugin_name_discriminator = config.get('discriminator', 'default_discriminator')
    print(f"Loading Discriminator Plugin: {plugin_name_discriminator}")
    try:
        discriminator_class, _ = load_plugin('discriminator.plugins', plugin_name_discriminator)
        plugins['discriminator_plugin'] = discriminator_class(config)
        print(f"✓ Discriminator Plugin '{plugin_name_discriminator}' loaded successfully")
    except Exception as e:
        print(f"Failed to load Discriminator Plugin '{plugin_name_discriminator}': {e}")
        traceback.print_exc()
        # Try to load default implementation
        try:
            from tsg_plugins.discriminator_plugin import DiscriminatorPlugin
            plugins['discriminator_plugin'] = DiscriminatorPlugin(config)
            print("✓ Default Discriminator Plugin loaded successfully")
        except Exception as e2:
            print(f"Failed to load default Discriminator Plugin: {e2}")

    # Load Trainer Plugin
    plugin_name_trainer = config.get('trainer', 'gan_trainer')
    print(f"Loading Trainer Plugin: {plugin_name_trainer}")
    try:
        trainer_class, _ = load_plugin('trainer.plugins', plugin_name_trainer)
        # Pass the generator and discriminator plugins to the trainer
        plugins['trainer_plugin'] = trainer_class(config, 
                                                  generator_plugin_instance=plugins['generator_plugin'],
                                                  discriminator_plugin_instance=plugins['discriminator_plugin'])
        print(f"✓ Trainer Plugin '{plugin_name_trainer}' loaded successfully")
    except Exception as e:
        print(f"Failed to load Trainer Plugin '{plugin_name_trainer}': {e}")
        traceback.print_exc()
        # Try to load default implementation
        try:
            from tsg_plugins.gan_trainer_plugin.gan_trainer_plugin import GANTrainerPlugin
            plugins['trainer_plugin'] = GANTrainerPlugin(config,
                                                         generator_plugin_instance=plugins['generator_plugin'],
                                                         discriminator_plugin_instance=plugins['discriminator_plugin'])
            print("✓ Default Trainer Plugin loaded successfully")
        except Exception as e2:
            print(f"Failed to load default Trainer Plugin: {e2}")

    # Load Evaluator Plugin (optional)
    plugin_name_evaluator = config.get('evaluator', 'default_evaluator')
    if plugin_name_evaluator:
        print(f"Loading Evaluator Plugin: {plugin_name_evaluator}")
        try:
            evaluator_class, _ = load_plugin('evaluator.plugins', plugin_name_evaluator)
            plugins['evaluator_plugin'] = evaluator_class(config)
            print(f"✓ Evaluator Plugin '{plugin_name_evaluator}' loaded successfully")
        except Exception as e:
            print(f"Failed to load Evaluator Plugin '{plugin_name_evaluator}': {e}")

    # Load Optimizer Plugin (optional)
    plugin_name_optimizer = config.get('optimizer', None)
    if plugin_name_optimizer:
        print(f"Loading Optimizer Plugin: {plugin_name_optimizer}")
        try:
            optimizer_class, _ = load_plugin('optimizer.plugins', plugin_name_optimizer)
            plugins['optimizer_plugin'] = optimizer_class(config)
            print(f"✓ Optimizer Plugin '{plugin_name_optimizer}' loaded successfully")
        except Exception as e:
            print(f"Failed to load Optimizer Plugin '{plugin_name_optimizer}': {e}")

    # Load Preprocessor Plugin (optional)
    plugin_name_preprocessor = config.get('preprocessor', None)
    if plugin_name_preprocessor:
        print(f"Loading Preprocessor Plugin: {plugin_name_preprocessor}")
        try:
            preprocessor_class, _ = load_plugin('preprocessor.plugins', plugin_name_preprocessor)
            plugins['preprocessor_plugin'] = preprocessor_class(config)
            print(f"✓ Preprocessor Plugin '{plugin_name_preprocessor}' loaded successfully")
        except Exception as e:
            print(f"Failed to load Preprocessor Plugin '{plugin_name_preprocessor}': {e}")

    return plugins


def merge_plugin_configurations(current_config: Dict[str, Any], plugins: Dict[str, Any], 
                               file_config: Dict[str, Any], cli_args: Dict[str, Any], 
                               unknown_args_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge plugin-specific configurations with the current configuration.
    
    Args:
        current_config: Current merged configuration
        plugins: Dictionary of loaded plugin instances
        file_config: Configuration from file
        cli_args: Command line arguments
        unknown_args_dict: Unknown command line arguments
        
    Returns:
        Updated configuration dictionary with plugin-specific settings
    """
    # For now, just return the current config
    # Plugin-specific configuration merging can be implemented here if needed
    return current_config


if __name__ == "__main__":
    main()
