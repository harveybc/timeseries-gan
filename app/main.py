#!/usr/bin/env python3
"""
main.py

Entry point for TimeSeries-GAN: High-level orchestration that loads and merges 
configurations, initializes plugins, then dispatches to the unified pipeline.
"""

import sys
import os
import traceback
from typing import Dict, Any
import logging # Added import

from app.cli import parse_args
from app.config import DEFAULT_VALUES
from app.config_handler import load_config, remote_load_config, save_config
from app.plugin_loader import load_plugin
from app.config_merger import merge_config, process_unknown_args
from app.data_processor import run_pipeline
from app.pipeline.generate_pipeline import GeneratePipeline # Added import for GeneratePipeline
# import tensorflow as tf # TensorFlow import moved to where it's used to avoid import errors if not strictly needed at module level

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def main():
    """
    High-level orchestration: Parse CLI args, load/merge configurations, 
    initialize plugins, then dispatch to unified pipeline.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
    for plugin_name, plugin_instance in plugins.items(): # Modified to iterate over items
        if plugin_instance:
            plugin_instance.set_params(**current_config) # Passing full config for now

    # OPERATION MODE DISPATCH
    operation_mode = current_config.get("operation_mode", "train") # Default to train
    
    if operation_mode == "generate":
        print(f"Operation Mode: {operation_mode}")
        if not current_config.get("load_generator_sequential_model_file") or \
           not current_config.get("load_discriminator_sequential_model_file"): # Discriminator path also checked as per initial reqs
            print("Error: For 'generate' mode, 'load_generator_sequential_model_file' and "
                  "'load_discriminator_sequential_model_file' must be specified in the config.")
            sys.exit(1)

        generator_model_path = current_config["load_generator_sequential_model_file"]
        discriminator_model_path = current_config["load_discriminator_sequential_model_file"]

        if not os.path.exists(generator_model_path):
            print(f"Error: Generator model file not found: {generator_model_path}")
            sys.exit(1)
        
        if not os.path.exists(discriminator_model_path):
            print(f"Error: Discriminator model file not found: {discriminator_model_path}")
            # sys.exit(1) # Commenting out exit for discriminator, as it might not be strictly needed for basic generation
            print(f"Warning: Discriminator model file {discriminator_model_path} not found. Proceeding with generation if discriminator is not essential.")


        print("Initializing Generate Pipeline directly...")
        feeder_plugin = plugins.get('feeder_plugin')
        generator_plugin_instance = plugins.get('generator_plugin')
        evaluator_plugin = plugins.get('evaluator_plugin') 

        if not feeder_plugin:
            print("Error: FeederPlugin is required for generate mode but was not loaded.")
            sys.exit(1)
        if not generator_plugin_instance:
            print("Error: GeneratorPlugin is required for generate mode but was not loaded.")
            sys.exit(1)
            
        # The GeneratePipeline will be responsible for loading the models using the paths from config.
        # We no longer load them directly in main.py to keep concerns separated.
        # GeneratorPlugin instance is passed, and GeneratePipeline can use its config to get model paths.

        generate_pipeline = GeneratePipeline(current_config, feeder_plugin, generator_plugin_instance, evaluator_plugin)
        print("Executing generate pipeline...")
        generate_pipeline.execute()
        print("Finished executing generate pipeline.")

    else:
        # Run the unified pipeline for other modes (train, optimize, etc.)
        print(f"Dispatching to unified pipeline for operation mode: {operation_mode}...")
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
    Ensures that plugins are instantiated in the correct order to resolve dependencies.
    
    Args:
        config: Current configuration dictionary
        
    Returns:
        Dictionary containing initialized plugin instances
    """
    plugins: Dict[str, Any] = {
        'feeder_plugin': None,
        'generator_plugin': None,
        'discriminator_plugin': None,
        'evaluator_plugin': None,
        'optimizer_plugin': None,
        'trainer_plugin': None
    }
    
    # Define plugin loading order based on dependencies
    # Core plugins first
    plugin_order = [
        ('feeder', 'feeder_plugin', []),
        ('generator', 'generator_plugin', []),
        ('discriminator', 'discriminator_plugin', []),
    ]
    
    # Plugins with dependencies (Trainer, Optimizer)
    # GANTrainerPlugin depends on Feeder, Generator, Discriminator
    # OptimizerPlugin depends on Feeder, Generator, Discriminator, GANTrainer
    dependent_plugins = [
        ('trainer', 'trainer_plugin', ['feeder_plugin', 'generator_plugin', 'discriminator_plugin']),
        ('optimizer', 'optimizer_plugin', ['feeder_plugin', 'generator_plugin', 'discriminator_plugin', 'trainer_plugin']),
        ('evaluator', 'evaluator_plugin', []), # Evaluator does not directly depend on generator_plugin in its constructor
    ]

    # Load and initialize core plugins
    for config_key, plugin_key, _ in plugin_order:
        plugin_name = config.get(config_key, f"default_{config_key.replace('_plugin', '')}")
        print(f"Loading {plugin_key.replace('_', ' ').title()}: {plugin_name}")
        try:
            plugin_class, _ = load_plugin(f"{config_key.split('_')[0]}.plugins", plugin_name)
            if plugin_class:
                # Core plugins are typically initialized with the main config
                plugins[plugin_key] = plugin_class(config=config)
                print(f"✓ {plugin_key.replace('_', ' ').title()} '{plugin_name}' loaded successfully")
            else:
                print(f"Failed to load {plugin_key.replace('_', ' ').title()} '{plugin_name}'. Class not found.")
        except Exception as e:
            print(f"Failed to load {plugin_key.replace('_', ' ').title()} '{plugin_name}'. Error: {e}")
            # Depending on how critical the plugin is, you might want to raise the exception
            # or allow the system to continue with a None for this plugin.
            # For now, we'll log and continue, but critical plugins like generator/discriminator/feeder
            # would likely cause issues downstream if None.

    # Load and initialize dependent plugins
    for config_key, plugin_key, dependencies in dependent_plugins:
        plugin_name = config.get(config_key, f"default_{config_key.replace('_plugin', '')}")
        print(f"Loading {plugin_key.replace('_', ' ').title()}: {plugin_name}")
        try:
            plugin_class, _ = load_plugin(f"{config_key.split('_')[0]}.plugins", plugin_name)
            if plugin_class:
                # Prepare constructor arguments for dependent plugins
                constructor_args = {'config': config}
                missing_deps = False
                for dep_key in dependencies:
                    if plugins.get(dep_key) is None:
                        print(f"Critical: Dependency '{dep_key}' for '{plugin_key}' is missing. Cannot initialize.")
                        missing_deps = True
                        break
                    constructor_args[dep_key] = plugins[dep_key]
                
                if not missing_deps:
                    plugins[plugin_key] = plugin_class(**constructor_args)
                    print(f"✓ {plugin_key.replace('_', ' ').title()} '{plugin_name}' loaded successfully")
                else:
                    print(f"Failed to initialize {plugin_key.replace('_', ' ').title()} '{plugin_name}' due to missing dependencies.")
            else:
                print(f"Failed to load {plugin_key.replace('_', ' ').title()} '{plugin_name}'. Class not found.")
        except Exception as e:
            print(f"Failed to load or initialize {plugin_key.replace('_', ' ').title()} '{plugin_name}'. Error: {e}")
            # Handle as above, potentially re-raising or logging.

    # Final check for critical plugins (optional, but good practice)
    critical_plugin_keys = ['feeder_plugin', 'generator_plugin', 'discriminator_plugin', 'trainer_plugin']
    for key in critical_plugin_keys:
        if plugins[key] is None:
            print(f"CRITICAL ERROR: Core plugin '{key}' failed to load or initialize. The application may not function correctly.")
            # Consider raising an exception here if these are absolutely essential
            # raise RuntimeError(f"Core plugin '{key}' could not be initialized.")

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
