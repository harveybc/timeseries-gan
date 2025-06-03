#!/usr/bin/env python3
"""
Model Persistence Module

This module handles model saving and loading operations,
providing focused functionality for model persistence management.
"""

import os
import json
import logging
import tensorflow as tf
from typing import Any, Dict, List, Optional
from datetime import datetime


class ModelPersistence:
    """Handles model saving and loading operations."""
    
    def __init__(self, params: Dict[str, Any], logger: logging.Logger):
        """Initialize model persistence manager."""
        self.params = params
        self.logger = logger
        
        # File naming templates
        self.generator_template = params.get("save_generator_epoch_template", "generator_epoch_{epoch}.keras")
        self.discriminator_template = params.get("save_discriminator_epoch_template", "discriminator_epoch_{epoch}.keras")
        self.gan_template = params.get("save_gan_epoch_template", "gan_epoch_{epoch}.keras")
        
        # Final model filenames
        self.final_generator_name = params.get("final_generator_model_filename", "generator_final.keras")
        self.final_discriminator_name = params.get("final_discriminator_model_filename", "discriminator_final.keras")
        self.final_gan_name = params.get("final_gan_model_filename", "gan_final.keras")
        
        self.logger.info("ModelPersistence initialized")
    
    def save_models(self, generator: Optional[tf.keras.Model] = None,
                   discriminator: Optional[tf.keras.Model] = None,
                   gan_model: Optional[tf.keras.Model] = None,
                   models_dir: str = "", path_prefix: str = "",
                   epoch: Optional[int] = None) -> Dict[str, str]:
        """
        Save GAN models to disk.
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            gan_model: Combined GAN model
            models_dir: Directory to save models
            path_prefix: Prefix for model files
            epoch: Current epoch (for checkpoint naming)
        
        Returns:
            Dictionary with saved file paths
        """
        saved_paths = {}
        
        try:
            # Ensure models directory exists
            os.makedirs(models_dir, exist_ok=True)
            
            # Save generator
            if generator is not None:
                if epoch is not None:
                    filename = self.generator_template.format(epoch=epoch)
                else:
                    filename = path_prefix + self.final_generator_name
                
                generator_path = os.path.join(models_dir, filename)
                generator.save(generator_path)
                saved_paths['generator'] = generator_path
                self.logger.info(f"Generator saved to: {generator_path}")
            
            # Save discriminator
            if discriminator is not None:
                if epoch is not None:
                    filename = self.discriminator_template.format(epoch=epoch)
                else:
                    filename = path_prefix + self.final_discriminator_name
                
                discriminator_path = os.path.join(models_dir, filename)
                discriminator.save(discriminator_path)
                saved_paths['discriminator'] = discriminator_path
                self.logger.info(f"Discriminator saved to: {discriminator_path}")
            
            # Save GAN model
            if gan_model is not None:
                if epoch is not None:
                    filename = self.gan_template.format(epoch=epoch)
                else:
                    filename = path_prefix + self.final_gan_name
                
                gan_path = os.path.join(models_dir, filename)
                gan_model.save(gan_path)
                saved_paths['gan_model'] = gan_path
                self.logger.info(f"GAN model saved to: {gan_path}")
            
            # Save metadata
            metadata_path = self._save_metadata(models_dir, path_prefix, epoch, saved_paths)
            saved_paths['metadata'] = metadata_path
            
            return saved_paths
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
            raise
    
    def load_models(self, models_dir: str, path_prefix: str = "",
                   epoch: Optional[int] = None) -> Dict[str, Optional[tf.keras.Model]]:
        """
        Load GAN models from disk.
        
        Args:
            models_dir: Directory containing saved models
            path_prefix: Prefix for model files
            epoch: Specific epoch to load (for checkpoints)
        
        Returns:
            Dictionary with loaded models
        """
        loaded_models = {
            'generator': None,
            'discriminator': None,
            'gan_model': None
        }
        
        try:
            # Load generator
            generator_path = self._get_model_path(
                models_dir, path_prefix, self.generator_template, 
                self.final_generator_name, epoch
            )
            if generator_path and os.path.exists(generator_path):
                loaded_models['generator'] = tf.keras.models.load_model(generator_path)
                self.logger.info(f"Generator loaded from: {generator_path}")
            
            # Load discriminator
            discriminator_path = self._get_model_path(
                models_dir, path_prefix, self.discriminator_template,
                self.final_discriminator_name, epoch
            )
            if discriminator_path and os.path.exists(discriminator_path):
                loaded_models['discriminator'] = tf.keras.models.load_model(discriminator_path)
                self.logger.info(f"Discriminator loaded from: {discriminator_path}")
            
            # Load GAN model
            gan_path = self._get_model_path(
                models_dir, path_prefix, self.gan_template,
                self.final_gan_name, epoch
            )
            if gan_path and os.path.exists(gan_path):
                loaded_models['gan_model'] = tf.keras.models.load_model(gan_path)
                self.logger.info(f"GAN model loaded from: {gan_path}")
            
            return loaded_models
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise
    
    def _get_model_path(self, models_dir: str, path_prefix: str, 
                       template: str, final_name: str, epoch: Optional[int]) -> Optional[str]:
        """Get the full path for a model file."""
        if epoch is not None:
            filename = template.format(epoch=epoch)
        else:
            filename = path_prefix + final_name
        
        return os.path.join(models_dir, filename)
    
    def _save_metadata(self, models_dir: str, path_prefix: str, 
                      epoch: Optional[int], saved_paths: Dict[str, str]) -> str:
        """Save metadata about the saved models."""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'path_prefix': path_prefix,
            'saved_files': saved_paths,
            'model_parameters': {
                'seq_len': self.params.get('seq_len'),
                'latent_dim': self.params.get('latent_dim'),
                'batch_size': self.params.get('gan_batch_size'),
                'generator_lr': self.params.get('generator_lr'),
                'discriminator_lr': self.params.get('discriminator_lr')
            }
        }
        
        metadata_filename = f"{path_prefix}model_metadata.json"
        if epoch is not None:
            metadata_filename = f"model_metadata_epoch_{epoch}.json"
        
        metadata_path = os.path.join(models_dir, metadata_filename)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Metadata saved to: {metadata_path}")
        return metadata_path
    
    def load_metadata(self, models_dir: str, path_prefix: str = "",
                     epoch: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Load metadata about saved models."""
        try:
            metadata_filename = f"{path_prefix}model_metadata.json"
            if epoch is not None:
                metadata_filename = f"model_metadata_epoch_{epoch}.json"
            
            metadata_path = os.path.join(models_dir, metadata_filename)
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.logger.info(f"Metadata loaded from: {metadata_path}")
                return metadata
            else:
                self.logger.warning(f"Metadata file not found: {metadata_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading metadata: {e}")
            return None
    
    def list_available_checkpoints(self, models_dir: str) -> Dict[str, List[int]]:
        """
        List available model checkpoints.
        
        Args:
            models_dir: Directory containing saved models
        
        Returns:
            Dictionary with lists of available epochs for each model type
        """
        checkpoints = {
            'generator': [],
            'discriminator': [],
            'gan_model': []
        }
        
        try:
            if not os.path.exists(models_dir):
                return checkpoints
            
            # Scan directory for checkpoint files
            for filename in os.listdir(models_dir):
                # Check generator checkpoints
                if 'generator_epoch_' in filename and filename.endswith('.keras'):
                    try:
                        epoch = int(filename.replace('generator_epoch_', '').replace('.keras', ''))
                        checkpoints['generator'].append(epoch)
                    except ValueError:
                        pass
                
                # Check discriminator checkpoints
                elif 'discriminator_epoch_' in filename and filename.endswith('.keras'):
                    try:
                        epoch = int(filename.replace('discriminator_epoch_', '').replace('.keras', ''))
                        checkpoints['discriminator'].append(epoch)
                    except ValueError:
                        pass
                
                # Check GAN checkpoints
                elif 'gan_epoch_' in filename and filename.endswith('.keras'):
                    try:
                        epoch = int(filename.replace('gan_epoch_', '').replace('.keras', ''))
                        checkpoints['gan_model'].append(epoch)
                    except ValueError:
                        pass
            
            # Sort epochs
            for model_type in checkpoints:
                checkpoints[model_type].sort()
            
            self.logger.info(f"Found checkpoints: {checkpoints}")
            return checkpoints
            
        except Exception as e:
            self.logger.error(f"Error listing checkpoints: {e}")
            return checkpoints
    
    def cleanup_old_checkpoints(self, models_dir: str, keep_every_n: int = 5,
                               keep_latest_n: int = 3):
        """
        Clean up old checkpoint files to save disk space.
        
        Args:
            models_dir: Directory containing saved models
            keep_every_n: Keep every Nth checkpoint
            keep_latest_n: Number of latest checkpoints to always keep
        """
        try:
            checkpoints = self.list_available_checkpoints(models_dir)
            
            for model_type, epochs in checkpoints.items():
                if len(epochs) <= keep_latest_n:
                    continue  # Not enough checkpoints to clean
                
                # Determine which epochs to keep
                epochs_to_keep = set()
                
                # Keep latest N
                epochs_to_keep.update(epochs[-keep_latest_n:])
                
                # Keep every Nth
                for i, epoch in enumerate(epochs):
                    if i % keep_every_n == 0:
                        epochs_to_keep.add(epoch)
                
                # Remove checkpoints not in keep list
                for epoch in epochs:
                    if epoch not in epochs_to_keep:
                        # Build filename based on model type
                        if model_type == 'generator':
                            filename = self.generator_template.format(epoch=epoch)
                        elif model_type == 'discriminator':
                            filename = self.discriminator_template.format(epoch=epoch)
                        elif model_type == 'gan_model':
                            filename = self.gan_template.format(epoch=epoch)
                        
                        filepath = os.path.join(models_dir, filename)
                        if os.path.exists(filepath):
                            os.remove(filepath)
                            self.logger.info(f"Removed old checkpoint: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up checkpoints: {e}")
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information."""
        return {
            "generator_template": self.generator_template,
            "discriminator_template": self.discriminator_template,
            "gan_template": self.gan_template,
            "final_generator_name": self.final_generator_name,
            "final_discriminator_name": self.final_discriminator_name,
            "final_gan_name": self.final_gan_name
        }
