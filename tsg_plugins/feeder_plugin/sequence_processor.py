"""
Sequence Processor Module

Handles sequence processing operations for time series data.
Manages windowing, sequencing, and temporal structure preparation.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


class SequenceProcessor:
    """
    Handles sequence processing for time series data.
    
    Manages windowing operations, sequence preparation, and temporal
    structure validation for GAN training and generation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the sequence processor."""
        self.config = config
        
        # Sequence parameters
        self.sequence_length = config.get('sequence_length', 60)
        self.stride = config.get('sequence_stride', 1)
        self.overlap_ratio = config.get('sequence_overlap', 0.0)
        self.min_sequence_length = config.get('min_sequence_length', 10)
        
        # Padding and validation
        self.use_padding = config.get('use_sequence_padding', True)
        self.padding_method = config.get('padding_method', 'zero')
        self.validate_sequences = config.get('validate_sequences', True)
        
        # State tracking
        self.is_initialized = False
        self.processing_stats = {}
        
        logger.info("SequenceProcessor initialized")
    
    def initialize(self) -> bool:
        """Initialize the sequence processor."""
        try:
            # Validate parameters
            self._validate_parameters()
            
            # Setup processing statistics
            self._setup_processing_stats()
            
            self.is_initialized = True
            logger.info("SequenceProcessor initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SequenceProcessor: {e}")
            return False
    
    def process_sequences(self, data: np.ndarray, mode: str) -> np.ndarray:
        """
        Process input data into sequences.
        
        Args:
            data: Input time series data
            mode: Processing mode ('train', 'generate', 'evaluate')
            
        Returns:
            Processed sequences
        """
        try:
            if not self.is_initialized:
                raise ValueError("SequenceProcessor not initialized")
            
            # Validate input data
            if self.validate_sequences and not self.validate_input(data):
                raise ValueError("Input data validation failed")
            
            # Create sequences based on mode
            if mode == 'train':
                sequences = self._create_training_sequences(data)
            elif mode == 'generate':
                sequences = self._create_generation_sequences(data)
            elif mode == 'evaluate':
                sequences = self._create_evaluation_sequences(data)
            else:
                raise ValueError(f"Unknown processing mode: {mode}")
            
            # Apply post-processing
            sequences = self._post_process_sequences(sequences, mode)
            
            # Update statistics
            self._update_processing_stats(sequences, mode)
            
            logger.debug(f"Processed {len(sequences)} sequences for mode: {mode}")
            return sequences
            
        except Exception as e:
            logger.error(f"Error processing sequences: {e}")
            return np.array([])
    
    def _create_training_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create sequences for training mode."""
        sequences = []
        
        # Create overlapping windows for training
        for i in range(0, len(data) - self.sequence_length + 1, self.stride):
            sequence = data[i:i + self.sequence_length]
            sequences.append(sequence)
        
        return np.array(sequences) if sequences else np.array([])
    
    def _create_generation_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create sequences for generation mode."""
        # For generation, we might need seed sequences or conditioning
        if len(data) < self.sequence_length:
            if self.use_padding:
                # Pad the sequence
                padded_data = self._pad_sequence(data)
                return np.array([padded_data])
            else:
                logger.warning("Input data shorter than sequence length")
                return np.array([data])
        
        # Use the last sequence_length points as seed
        seed_sequence = data[-self.sequence_length:]
        return np.array([seed_sequence])
    
    def _create_evaluation_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create sequences for evaluation mode."""
        # Similar to training but potentially with different stride
        eval_stride = max(1, self.stride // 2)  # Denser sampling for evaluation
        sequences = []
        
        for i in range(0, len(data) - self.sequence_length + 1, eval_stride):
            sequence = data[i:i + self.sequence_length]
            sequences.append(sequence)
        
        return np.array(sequences) if sequences else np.array([])
    
    def _post_process_sequences(self, sequences: np.ndarray, mode: str) -> np.ndarray:
        """Apply post-processing to sequences."""
        if len(sequences) == 0:
            return sequences
        
        processed = sequences.copy()
        
        # Apply mode-specific post-processing
        if mode == 'train':
            processed = self._apply_training_postprocessing(processed)
        elif mode == 'generate':
            processed = self._apply_generation_postprocessing(processed)
        elif mode == 'evaluate':
            processed = self._apply_evaluation_postprocessing(processed)
        
        return processed
    
    def _apply_training_postprocessing(self, sequences: np.ndarray) -> np.ndarray:
        """Apply training-specific post-processing."""
        # Shuffle sequences for training
        np.random.shuffle(sequences)
        return sequences
    
    def _apply_generation_postprocessing(self, sequences: np.ndarray) -> np.ndarray:
        """Apply generation-specific post-processing."""
        # No shuffling for generation
        return sequences
    
    def _apply_evaluation_postprocessing(self, sequences: np.ndarray) -> np.ndarray:
        """Apply evaluation-specific post-processing."""
        # Keep original order for evaluation
        return sequences
    
    def _pad_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Pad sequence to required length."""
        if len(sequence) >= self.sequence_length:
            return sequence[:self.sequence_length]
        
        padding_length = self.sequence_length - len(sequence)
        
        if self.padding_method == 'zero':
            padding = np.zeros((padding_length, sequence.shape[1]), dtype=sequence.dtype)
        elif self.padding_method == 'repeat':
            padding = np.tile(sequence[-1:], (padding_length, 1))
        elif self.padding_method == 'mirror':
            if len(sequence) > 1:
                padding = np.flip(sequence[-padding_length:], axis=0)
            else:
                padding = np.tile(sequence, (padding_length, 1))
        else:
            # Default to zero padding
            padding = np.zeros((padding_length, sequence.shape[1]), dtype=sequence.dtype)
        
        return np.vstack([sequence, padding])
    
    def validate_input(self, data: np.ndarray) -> bool:
        """Validate input data format and dimensions."""
        try:
            # Check basic requirements
            if data is None or len(data) == 0:
                logger.error("Empty or None input data")
                return False
            
            # Check dimensions
            if data.ndim < 2:
                logger.error(f"Invalid data dimensions: {data.ndim}, expected at least 2")
                return False
            
            # Check minimum length
            if len(data) < self.min_sequence_length:
                logger.warning(f"Data length {len(data)} below minimum {self.min_sequence_length}")
                if not self.use_padding:
                    return False
            
            # Check for invalid values
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                logger.error("Invalid values (NaN or Inf) detected in input data")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating input: {e}")
            return False
    
    def create_sliding_windows(self, data: np.ndarray, window_size: int, step_size: int = 1) -> np.ndarray:
        """Create sliding windows from data."""
        try:
            if len(data) < window_size:
                logger.warning("Data length less than window size")
                return np.array([])
            
            windows = []
            for i in range(0, len(data) - window_size + 1, step_size):
                window = data[i:i + window_size]
                windows.append(window)
            
            return np.array(windows)
            
        except Exception as e:
            logger.error(f"Error creating sliding windows: {e}")
            return np.array([])
    
    def reshape_for_model(self, sequences: np.ndarray, target_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """Reshape sequences for model input."""
        try:
            if target_shape is None:
                # Default reshape: (batch, sequence, features)
                return sequences
            
            # Attempt to reshape to target shape
            return sequences.reshape(target_shape)
            
        except Exception as e:
            logger.error(f"Error reshaping sequences: {e}")
            return sequences
    
    def _validate_parameters(self):
        """Validate sequence processing parameters."""
        if self.sequence_length <= 0:
            raise ValueError(f"Invalid sequence length: {self.sequence_length}")
        
        if self.stride <= 0:
            raise ValueError(f"Invalid stride: {self.stride}")
        
        if not 0 <= self.overlap_ratio <= 1:
            raise ValueError(f"Invalid overlap ratio: {self.overlap_ratio}")
        
        valid_padding = ['zero', 'repeat', 'mirror']
        if self.padding_method not in valid_padding:
            logger.warning(f"Unknown padding method: {self.padding_method}, using 'zero'")
            self.padding_method = 'zero'
    
    def _setup_processing_stats(self):
        """Setup processing statistics tracking."""
        self.processing_stats = {
            'total_sequences': 0,
            'mode_counts': {},
            'avg_sequence_length': 0.0,
            'padding_applied': 0
        }
    
    def _update_processing_stats(self, sequences: np.ndarray, mode: str):
        """Update processing statistics."""
        if len(sequences) == 0:
            return
        
        batch_size = len(sequences)
        self.processing_stats['total_sequences'] += batch_size
        
        # Track mode usage
        self.processing_stats['mode_counts'][mode] = (
            self.processing_stats['mode_counts'].get(mode, 0) + batch_size
        )
        
        # Track average sequence length
        avg_length = np.mean([len(seq) for seq in sequences])
        self.processing_stats['avg_sequence_length'] = avg_length
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return self.processing_stats.copy()
    
    def is_ready(self) -> bool:
        """Check if the processor is ready for use."""
        return self.is_initialized
    
    def cleanup(self):
        """Cleanup processor resources."""
        self.processing_stats.clear()
        self.is_initialized = False
        logger.info("SequenceProcessor cleanup completed")
