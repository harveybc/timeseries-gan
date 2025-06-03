import pytest
import numpy as np

from tsg_plugins.optimizer_plugin import OptimizerPlugin


class DummyFeeder:
    """Mock feeder plugin generating latent vectors."""
    def __init__(self, config=None):
        self.params = config or {}

    def set_params(self, **kwargs):
        self.params.update(kwargs)

    def generate(self, n_samples: int, latent_dim: int):
        # Return random latent vectors
        return np.random.randn(n_samples, latent_dim)


class DummyGenerator:
    """Mock generator plugin creating synthetic data."""
    def __init__(self, config=None):
        self.params = config or {}

    def set_params(self, **kwargs):
        self.params.update(kwargs)

    def generate(self, feeder_outputs_sequence, sequence_length_T, initial_full_feature_window):
        # Return a batch: list of one array shaped (sequence_length_T, features)
        feature_count = initial_full_feature_window.shape[1]
        synthetic = np.random.randn(sequence_length_T, feature_count)
        return [synthetic]


class DummyEvaluator:
    """Mock evaluator returning a simple MSE metric."""
    def __init__(self, config=None):
        self.params = config or {}

    def set_params(self, **kwargs):
        self.params.update(kwargs)

    def evaluate(self, synthetic_data):
        # Compute MSE against zero
        mse = np.mean(synthetic_data ** 2)
        return {'mse': mse}


@pytest.fixture
def sample_config():
    return {
        'population_size': 10,
        'n_generations': 5,
        'cxpb': 0.5,
        'mutpb': 0.2,
        'hyperparameter_bounds': {
            'latent_dim': (4, 8),
            'batch_size': (2, 4),
            'learning_rate': (1e-3, 1e-2)
        },
        'optimizer_n_samples_per_eval': 20,
        'random_seed': 42
    }


def test_validate_and_optimize(sample_config):
    """Test full optimization workflow with dummy plugins."""
    # Initialize optimizer plugin
    plugin = OptimizerPlugin(sample_config)
    assert plugin.validate_config()

    # Run optimization
    best_params = plugin.optimize(
        feeder_plugin=DummyFeeder(sample_config),
        generator_plugin=DummyGenerator(sample_config),
        evaluator_plugin=DummyEvaluator(sample_config),
        config=sample_config
    )

    # Best parameters should match keys
    expected_keys = set(sample_config['hyperparameter_bounds'].keys())
    assert set(best_params.keys()) == expected_keys

    # Parameter values within bounds
    for key, (min_val, max_val) in sample_config['hyperparameter_bounds'].items():
        val = best_params[key]
        assert min_val <= val <= max_val


def test_optimization_results_structure(sample_config):
    """Test that optimization results store the expected fields."""
    plugin = OptimizerPlugin(sample_config)
    plugin.optimize(
        feeder_plugin=DummyFeeder(sample_config),
        generator_plugin=DummyGenerator(sample_config),
        evaluator_plugin=DummyEvaluator(sample_config),
        config=sample_config
    )

    results = plugin.get_optimization_results()
    assert 'best_parameters' in results
    assert 'evaluation_stats' in results
    assert 'optimization_info' in results


def test_reset_and_cleanup(sample_config):
    """Test reset and cleanup functionality."""
    plugin = OptimizerPlugin(sample_config)
    plugin.optimize(
        feeder_plugin=DummyFeeder(sample_config),
        generator_plugin=DummyGenerator(sample_config),
        evaluator_plugin=DummyEvaluator(sample_config),
        config=sample_config
    )
    assert plugin.is_initialized

    plugin.reset()
    assert not plugin.is_initialized
    assert plugin.get_optimization_results() == {'status': 'not_run'}

    plugin.cleanup()
    # cleanup should not raise and should preserve reset state
    assert not plugin.is_initialized
    assert plugin.get_optimization_results() == {'status': 'not_run'}
