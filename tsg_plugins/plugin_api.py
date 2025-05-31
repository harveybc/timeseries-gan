from abc import ABC, abstractmethod

class FeederPlugin(ABC):
    @abstractmethod
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def feed(self, data_path):
        """Loads and preprocesses data from the given path."""
        pass

class GeneratorPlugin(ABC):
    @abstractmethod
    def __init__(self, config):
        self.config = config
        self.model = None

    @abstractmethod
    def load_model(self, model_path=None):
        """Loads the generator model."""
        pass

    @abstractmethod
    def generate(self, num_samples, initial_data=None):
        """Generates synthetic data."""
        pass

class TrainerPlugin(ABC):
    @abstractmethod
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def train(self, x_train_file):
        """Trains a model using data from x_train_file."""
        pass

    @abstractmethod
    def save_model(self):
        """Saves the trained model."""
        pass

    @abstractmethod
    def load_model(self):
        """Loads a pre-trained model."""
        pass

    @abstractmethod
    def get_generator(self):
        """Returns the generator part of the trained model, if applicable."""
        pass
