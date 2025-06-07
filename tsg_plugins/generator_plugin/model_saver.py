\
import logging
import os
from typing import Optional
from tensorflow.keras.models import Model

class ModelSaver:
    \"\"\"Handles saving Keras models.\"\"\"

    def __init__(self, logger: Optional[logging.Logger] = None):
        \"\"\"
        Initialize the ModelSaver.

        Args:
            logger: Optional logger instance.
        \"\"\"
        self.logger = logger if logger else logging.getLogger(__name__)

    def save_model_to_path(self, model: Model, file_path: str, overwrite: bool = True) -> bool:
        \"\"\"
        Save a Keras model to the specified file path.

        Args:
            model: The Keras model to save.
            file_path: The path (including filename and .keras extension) where the model should be saved.
            overwrite: Whether to overwrite the file if it already exists.

        Returns:
            bool: True if saving was successful, False otherwise.
        \"\"\"
        if not model:
            self.logger.error("ModelSaver: No model provided to save.")
            return False
        
        if not file_path:
            self.logger.error("ModelSaver: No file_path provided to save the model.")
            return False

        try:
            # Ensure the directory exists
            dir_name = os.path.dirname(file_path)
            if dir_name: # Ensure dir_name is not empty (e.g. saving in current dir)
                os.makedirs(dir_name, exist_ok=True)
            
            self.logger.info(f"Saving model to {file_path} (overwrite={overwrite})...")
            model.save(file_path, overwrite=overwrite)
            self.logger.info(f"Model successfully saved to {file_path}.")
            return True
        except Exception as e:
            self.logger.error(f"ModelSaver: Failed to save model to {file_path}. Error: {e}", exc_info=True)
            return False

    def save_weights_to_path(self, model: Model, file_path: str, overwrite: bool = True) -> bool:
        \"\"\"
        Save model weights to the specified file path.

        Args:
            model: The Keras model whose weights are to be saved.
            file_path: The path (including filename, typically .h5 or .weights.h5) where weights should be saved.
            overwrite: Whether to overwrite the file if it already exists.

        Returns:
            bool: True if saving was successful, False otherwise.
        \"\"\"
        if not model:
            self.logger.error("ModelSaver: No model provided to save weights.")
            return False
        
        if not file_path:
            self.logger.error("ModelSaver: No file_path provided to save the model weights.")
            return False

        try:
            # Ensure the directory exists
            dir_name = os.path.dirname(file_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            
            self.logger.info(f"Saving model weights to {file_path} (overwrite={overwrite})...")
            model.save_weights(file_path, overwrite=overwrite)
            self.logger.info(f"Model weights successfully saved to {file_path}.")
            return True
        except Exception as e:
            self.logger.error(f"ModelSaver: Failed to save model weights to {file_path}. Error: {e}", exc_info=True)
            return False
