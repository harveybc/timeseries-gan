#!/usr/bin/env python3
"""
optimize_pipeline.py

Hyperparameter optimization pipeline for TimeSeries-GAN.
Handles genetic algorithm-based optimization workflow for finding
optimal hyperparameters across feeder, generator, and evaluator plugins.

This module encapsulates optimization-specific logic following
single responsibility principle and extreme separation of concerns.

Author: TimeSeries-GAN Team
"""

import sys
import traceback
from typing import Dict, Any, Optional


class OptimizePipeline:
    """
    Pipeline for hyperparameter optimization using genetic algorithms.
    
    This pipeline coordinates the complete optimization workflow:
    - Configures optimization parameters and constraints
    - Executes genetic algorithm optimization across plugin parameters
    - Handles fitness evaluation using evaluator plugin
    - Manages optimization result persistence and reporting
    
    Attributes:
        config: Configuration dictionary containing optimization parameters
        optimizer_plugin: Plugin instance for genetic algorithm optimization
        feeder_plugin: Plugin instance for noise generation (optimization target)
        generator_plugin: Plugin instance for data generation (optimization target)
        evaluator_plugin: Plugin instance for fitness evaluation
    """
    
    def __init__(self, config: Dict[str, Any], optimizer_plugin, feeder_plugin,
                 generator_plugin, evaluator_plugin):
        """
        Initialize optimization pipeline with configuration and plugin instances.
        
        Args:
            config: Configuration dictionary containing optimization parameters
            optimizer_plugin: Plugin instance for genetic algorithm optimization
            feeder_plugin: Plugin instance for noise generation
            generator_plugin: Plugin instance for data generation
            evaluator_plugin: Plugin instance for fitness evaluation
        """
        self.config = config
        self.optimizer_plugin = optimizer_plugin
        self.feeder_plugin = feeder_plugin
        self.generator_plugin = generator_plugin
        self.evaluator_plugin = evaluator_plugin
    
    def execute(self) -> None:
        """
        Execute the complete hyperparameter optimization pipeline.
        
        Performs the following steps:
        1. Validate optimization configuration and plugin availability
        2. Configure optimization parameters and search space
        3. Execute genetic algorithm optimization
        4. Handle optimization results and reporting
        
        Raises:
            SystemExit: If required plugins are unavailable or optimization fails
        """
        print("Starting hyperparameter optimization pipeline...")
        
        try:
            # Validate optimization setup
            self._validate_optimization_setup()
            
            # Configure optimization parameters
            self._configure_optimization()
            
            # Execute optimization
            optimal_params = self._execute_optimization()
            
            # Handle optimization results
            self._handle_optimization_results(optimal_params)
            
            print("✔ Hyperparameter optimization completed successfully.")
            
        except Exception as e:
            print(f"❌ Hyperparameter optimization failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    def _validate_optimization_setup(self) -> None:
        """
        Validate optimization configuration and plugin availability.
        
        Checks that all required plugins are available and optimization
        parameters are properly configured.
        
        Raises:
            ValueError: If required plugins are missing or configuration is invalid
        """
        # Validate required plugins
        if not self.optimizer_plugin:
            raise ValueError("Optimizer plugin is required for optimization mode")
        
        if not self.feeder_plugin:
            raise ValueError("Feeder plugin is required for optimization")
        
        if not self.generator_plugin:
            raise ValueError("Generator plugin is required for optimization")
        
        if not self.evaluator_plugin:
            raise ValueError("Evaluator plugin is required for fitness evaluation")
        
        # Validate optimization configuration
        required_config_keys = [
            "optimization_population_size",
            "optimization_generations", 
            "optimization_mutation_rate",
            "optimization_crossover_rate"
        ]
        
        for key in required_config_keys:
            if key not in self.config:
                print(f"⚠ Warning: Optimization parameter '{key}' not configured, using plugin defaults")
        
        print("✓ Optimization setup validated")
    
    def _configure_optimization(self) -> None:
        """
        Configure optimization parameters and search space.
        
        Sets up the parameter search space, constraints, and optimization
        algorithm configuration based on plugin capabilities and user config.
        """
        print("Configuring optimization parameters...")
        
        # Extract optimization configuration
        population_size = self.config.get("optimization_population_size", 20)
        generations = self.config.get("optimization_generations", 10)
        mutation_rate = self.config.get("optimization_mutation_rate", 0.1)
        crossover_rate = self.config.get("optimization_crossover_rate", 0.7)
        
        print(f"Optimization configuration:")
        print(f"  Population size: {population_size}")
        print(f"  Generations: {generations}")
        print(f"  Mutation rate: {mutation_rate}")
        print(f"  Crossover rate: {crossover_rate}")
        
        # Configure optimizer plugin with parameters
        self.optimizer_plugin.set_params(
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate
        )
        
        print("✓ Optimization configuration completed")
    
    def _execute_optimization(self) -> Dict[str, Any]:
        """
        Execute genetic algorithm optimization.
        
        Runs the genetic algorithm to find optimal hyperparameters
        for the configured plugins and evaluation metrics.
        
        Returns:
            Dict[str, Any]: Dictionary containing optimal parameters found
            
        Raises:
            RuntimeError: If optimization execution fails
        """
        try:
            print("Executing genetic algorithm optimization...")
            
            # Execute optimization using optimizer plugin
            optimal_params = self.optimizer_plugin.optimize(
                feeder_plugin=self.feeder_plugin,
                generator_plugin=self.generator_plugin,
                evaluator_plugin=self.evaluator_plugin,
                config=self.config
            )
            
            print("✓ Genetic algorithm optimization completed")
            return optimal_params
            
        except Exception as e:
            raise RuntimeError(f"Optimization execution failed: {e}")
    
    def _handle_optimization_results(self, optimal_params: Dict[str, Any]) -> None:
        """
        Handle optimization results including persistence and reporting.
        
        Args:
            optimal_params: Dictionary containing optimal parameters found
        """
        try:
            print("Handling optimization results...")
            
            # Log optimal parameters
            print("Optimal parameters found:")
            for param_name, param_value in optimal_params.items():
                print(f"  {param_name}: {param_value}")
            
            # Save optimal parameters if output file is configured
            output_file = self.config.get("optimization_output_file")
            if output_file:
                import json
                import os
                
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump(optimal_params, f, indent=2)
                print(f"✓ Optimal parameters saved to: {output_file}")
            
            print("✓ Optimization results handled successfully")
            
        except Exception as e:
            print(f"⚠ Warning: Failed to handle optimization results: {e}")
            # Don't fail the pipeline for result handling issues
