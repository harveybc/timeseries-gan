import logging
import sys

def setup_logger(name, level=logging.INFO):
    """
    Sets up a logger.
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False # To avoid duplicate logs if root logger is also configured
    return logger

# Example of a default logger for the application
# You can import this logger in other modules
# from app.utils.logging_utils import app_logger
# app_logger.info("This is a test log message.")

# app_logger = setup_logger('app')

def get_logger(name):
    """
    Retrieves a logger instance. If not already configured, it sets up a basic one.
    """
    logger = logging.getLogger(name)
    if not logger.handlers: # Check if logger already has handlers
        # Basic configuration if not already set up by a more specific call
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        logger.propagate = False
    return logger

if __name__ == '__main__':
    # Example usage:
    logger1 = get_logger('MyModule1')
    logger1.info("This is an info message from MyModule1.")
    logger1.warning("This is a warning message from MyModule1.")

    logger2 = setup_logger('MyModule2', level=logging.DEBUG)
    logger2.debug("This is a debug message from MyModule2.")
    logger2.info("This is an info message from MyModule2.")
