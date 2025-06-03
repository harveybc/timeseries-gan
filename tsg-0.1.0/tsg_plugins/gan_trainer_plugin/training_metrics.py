#!/usr/bin/env python3
"""
Training Metrics Module

This module handles training progress tracking and visualization,
providing focused functionality for metrics collection and analysis.
"""

import os
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime


class TrainingMetrics:
    """Handles training metrics collection and visualization."""
    
    def __init__(self, params: Dict[str, Any], logger: logging.Logger):
        """Initialize training metrics manager."""
        self.params = params
        self.logger = logger
        
        # Metrics storage
        self.metrics_history = {
            'epochs': [],
            'generator_losses': [],
            'discriminator_losses': [],
            'generator_accuracies': [],
            'discriminator_accuracies': [],
            'learning_rates_g': [],
            'learning_rates_d': [],
            'epoch_times': [],
            'timestamps': []
        }
        
        # Plotting configuration
        self.plot_dpi = params.get("loss_plot_dpi", 300)
        self.plot_template = params.get("loss_plot_epoch_template", "loss_plot_epoch_{epoch}.png")
        self.final_plot_name = params.get("final_loss_plot_filename", "loss_plot_final.png")
        
        self.logger.info("TrainingMetrics initialized")
    
    def record_epoch(self, epoch: int, generator_loss: float, discriminator_loss: float,
                    generator_accuracy: Optional[float] = None,
                    discriminator_accuracy: Optional[float] = None,
                    learning_rate_g: Optional[float] = None,
                    learning_rate_d: Optional[float] = None,
                    epoch_time: Optional[float] = None):
        """
        Record metrics for a training epoch.
        
        Args:
            epoch: Current epoch number
            generator_loss: Generator loss value
            discriminator_loss: Discriminator loss value
            generator_accuracy: Generator accuracy (optional)
            discriminator_accuracy: Discriminator accuracy (optional)
            learning_rate_g: Generator learning rate (optional)
            learning_rate_d: Discriminator learning rate (optional)
            epoch_time: Time taken for epoch in seconds (optional)
        """
        self.metrics_history['epochs'].append(epoch)
        self.metrics_history['generator_losses'].append(generator_loss)
        self.metrics_history['discriminator_losses'].append(discriminator_loss)
        self.metrics_history['generator_accuracies'].append(generator_accuracy or 0.0)
        self.metrics_history['discriminator_accuracies'].append(discriminator_accuracy or 0.0)
        self.metrics_history['learning_rates_g'].append(learning_rate_g or 0.0)
        self.metrics_history['learning_rates_d'].append(learning_rate_d or 0.0)
        self.metrics_history['epoch_times'].append(epoch_time or 0.0)
        self.metrics_history['timestamps'].append(datetime.now().isoformat())
        
        # Log every 100 epochs
        if epoch % 100 == 0:
            self.logger.info(f"Epoch {epoch} - G_loss: {generator_loss:.4f}, D_loss: {discriminator_loss:.4f}")
    
    def save_metrics(self, metrics_dir: str, filename: Optional[str] = None) -> str:
        """
        Save training metrics to JSON file.
        
        Args:
            metrics_dir: Directory to save metrics
            filename: Custom filename (optional)
        
        Returns:
            Path to saved metrics file
        """
        try:
            os.makedirs(metrics_dir, exist_ok=True)
            
            if filename is None:
                filename = self.params.get("training_metrics_filename", "training_metrics.json")
            
            metrics_path = os.path.join(metrics_dir, filename)
            
            # Prepare metrics data
            metrics_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'total_epochs': len(self.metrics_history['epochs']),
                    'parameters': {
                        'gan_batch_size': self.params.get('gan_batch_size'),
                        'generator_lr': self.params.get('generator_lr'),
                        'discriminator_lr': self.params.get('discriminator_lr'),
                        'seq_len': self.params.get('seq_len'),
                        'latent_dim': self.params.get('latent_dim')
                    }
                },
                'training_history': self.metrics_history,
                'summary_statistics': self._calculate_summary_stats()
            }
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            self.logger.info(f"Training metrics saved to: {metrics_path}")
            return metrics_path
            
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")
            raise
    
    def load_metrics(self, metrics_path: str) -> Dict[str, Any]:
        """
        Load training metrics from JSON file.
        
        Args:
            metrics_path: Path to metrics file
        
        Returns:
            Loaded metrics data
        """
        try:
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
            
            # Update internal metrics history
            if 'training_history' in metrics_data:
                self.metrics_history = metrics_data['training_history']
            
            self.logger.info(f"Training metrics loaded from: {metrics_path}")
            return metrics_data
            
        except Exception as e:
            self.logger.error(f"Error loading metrics: {e}")
            raise
    
    def plot_training_history(self, plots_dir: str, epoch: Optional[int] = None,
                             save_plot: bool = True) -> Optional[str]:
        """
        Plot training history.
        
        Args:
            plots_dir: Directory to save plots
            epoch: Current epoch (for checkpoint naming)
            save_plot: Whether to save the plot
        
        Returns:
            Path to saved plot if save_plot is True
        """
        try:
            if not self.metrics_history['epochs']:
                self.logger.warning("No training history to plot")
                return None
            
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('GAN Training History', fontsize=16)
            
            epochs = self.metrics_history['epochs']
            
            # Plot losses
            ax1.plot(epochs, self.metrics_history['generator_losses'], label='Generator Loss', color='blue')
            ax1.plot(epochs, self.metrics_history['discriminator_losses'], label='Discriminator Loss', color='red')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Losses')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot accuracies
            ax2.plot(epochs, self.metrics_history['generator_accuracies'], label='Generator Accuracy', color='blue')
            ax2.plot(epochs, self.metrics_history['discriminator_accuracies'], label='Discriminator Accuracy', color='red')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Training Accuracies')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot learning rates
            ax3.plot(epochs, self.metrics_history['learning_rates_g'], label='Generator LR', color='blue')
            ax3.plot(epochs, self.metrics_history['learning_rates_d'], label='Discriminator LR', color='red')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_title('Learning Rates')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')
            
            # Plot epoch times
            if any(t > 0 for t in self.metrics_history['epoch_times']):
                ax4.plot(epochs, self.metrics_history['epoch_times'], label='Epoch Time', color='green')
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Time (seconds)')
                ax4.set_title('Epoch Training Time')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No timing data available', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Epoch Training Time')
            
            plt.tight_layout()
            
            if save_plot:
                os.makedirs(plots_dir, exist_ok=True)
                
                if epoch is not None:
                    filename = self.plot_template.format(epoch=epoch)
                else:
                    filename = self.final_plot_name
                
                plot_path = os.path.join(plots_dir, filename)
                plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"Training plot saved to: {plot_path}")
                return plot_path
            else:
                plt.show()
                return None
                
        except Exception as e:
            self.logger.error(f"Error plotting training history: {e}")
            plt.close()
            raise
    
    def plot_loss_comparison(self, plots_dir: str, window_size: int = 100) -> str:
        """
        Plot smoothed loss comparison.
        
        Args:
            plots_dir: Directory to save plots
            window_size: Window size for smoothing
        
        Returns:
            Path to saved plot
        """
        try:
            if len(self.metrics_history['epochs']) < window_size:
                window_size = len(self.metrics_history['epochs'])
            
            # Calculate moving averages
            g_losses = np.array(self.metrics_history['generator_losses'])
            d_losses = np.array(self.metrics_history['discriminator_losses'])
            epochs = np.array(self.metrics_history['epochs'])
            
            g_losses_smooth = self._moving_average(g_losses, window_size)
            d_losses_smooth = self._moving_average(d_losses, window_size)
            epochs_smooth = epochs[window_size-1:]
            
            # Create plot
            plt.figure(figsize=(12, 6))
            plt.plot(epochs, g_losses, alpha=0.3, color='blue', label='Generator Loss (raw)')
            plt.plot(epochs, d_losses, alpha=0.3, color='red', label='Discriminator Loss (raw)')
            plt.plot(epochs_smooth, g_losses_smooth, color='blue', linewidth=2, label=f'Generator Loss (MA-{window_size})')
            plt.plot(epochs_smooth, d_losses_smooth, color='red', linewidth=2, label=f'Discriminator Loss (MA-{window_size})')
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('GAN Training Loss Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            os.makedirs(plots_dir, exist_ok=True)
            plot_path = os.path.join(plots_dir, "loss_comparison_smoothed.png")
            plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Loss comparison plot saved to: {plot_path}")
            return plot_path
            
        except Exception as e:
            self.logger.error(f"Error plotting loss comparison: {e}")
            plt.close()
            raise
    
    def _moving_average(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate moving average."""
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    def _calculate_summary_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics from training history."""
        if not self.metrics_history['epochs']:
            return {}
        
        g_losses = np.array(self.metrics_history['generator_losses'])
        d_losses = np.array(self.metrics_history['discriminator_losses'])
        epoch_times = np.array([t for t in self.metrics_history['epoch_times'] if t > 0])
        
        stats = {
            'final_generator_loss': float(g_losses[-1]) if len(g_losses) > 0 else None,
            'final_discriminator_loss': float(d_losses[-1]) if len(d_losses) > 0 else None,
            'min_generator_loss': float(np.min(g_losses)) if len(g_losses) > 0 else None,
            'min_discriminator_loss': float(np.min(d_losses)) if len(d_losses) > 0 else None,
            'mean_generator_loss': float(np.mean(g_losses)) if len(g_losses) > 0 else None,
            'mean_discriminator_loss': float(np.mean(d_losses)) if len(d_losses) > 0 else None,
            'std_generator_loss': float(np.std(g_losses)) if len(g_losses) > 0 else None,
            'std_discriminator_loss': float(np.std(d_losses)) if len(d_losses) > 0 else None,
            'average_epoch_time': float(np.mean(epoch_times)) if len(epoch_times) > 0 else None,
            'total_training_time': float(np.sum(epoch_times)) if len(epoch_times) > 0 else None,
            'total_epochs': len(self.metrics_history['epochs'])
        }
        
        return stats
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get the most recent training metrics."""
        if not self.metrics_history['epochs']:
            return {}
        
        return {
            'epoch': self.metrics_history['epochs'][-1],
            'generator_loss': self.metrics_history['generator_losses'][-1],
            'discriminator_loss': self.metrics_history['discriminator_losses'][-1],
            'generator_accuracy': self.metrics_history['generator_accuracies'][-1],
            'discriminator_accuracy': self.metrics_history['discriminator_accuracies'][-1],
            'timestamp': self.metrics_history['timestamps'][-1]
        }
    
    def reset_metrics(self):
        """Reset all metrics history."""
        for key in self.metrics_history:
            self.metrics_history[key] = []
        
        self.logger.info("Training metrics reset")
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information."""
        return {
            "total_epochs_recorded": len(self.metrics_history['epochs']),
            "latest_metrics": self.get_latest_metrics(),
            "summary_stats": self._calculate_summary_stats()
        }
