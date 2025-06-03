"""
setup.py for tsg (Timeseries GAN Synthetic Data Generator)

This module packages the `tsg` project using setuptools. It defines:

- Metadata (name, version, author, etc.)
- Entry points for the CLI `tsg` command
- Plugin entry points for feeder, generator, evaluator, optimizer
- Installation requirements and extras for development

"""

from setuptools import setup, find_packages
from pathlib import Path

# Project directory
here = Path(__file__).parent.resolve()

# Read long description from README.md
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="tsg",
    version="0.1.0",
    description=(
        "Synthetic Data Generator with plugin-based architecture for latent sampling, "
        "generation, evaluation, and optimization."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/harveybc/timeseries-gan",
    author="Harvey Bastidas",
    author_email="your.email@example.com",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Data Generation",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="synthetic-data generator plugin tsg",

    packages=find_packages(exclude=["tests", "docs"]),
    python_requires=">=3.8, <4",
    install_requires=[
        "numpy",        # Required for numerical operations
        "pandas",       # Required for data manipulation
        "tensorflow",   # Required for GAN trainer plugin
        "matplotlib",   # Required for plotting and visualization
        "scikit-learn", # Required for evaluation metrics
        "h5py",         # Required for model persistence
        "deap",         # Required for hyperparameter optimization
    ],
    extras_require={
        "dev": [
            "build",             # Build utilities
            "pytest",           # Testing framework
            "sphinx",           # Documentation generator
            "sphinx-rtd-theme"  # ReadTheDocs theme for Sphinx
        ]
    },

    entry_points={
        "console_scripts": [
            "tsg=app.main:main",
        ],
        "feeder.plugins": [
            "default_feeder=tsg_plugins.feeder_plugin:FeederPlugin",
        ],
        "generator.plugins": [
            "vae_generator = tsg_plugins.generator_plugin:GeneratorPlugin",
            "default_generator = tsg_plugins.generator_plugin:GeneratorPlugin"
        ],
        "evaluator.plugins": [ # Assuming you have an evaluator entry, based on logs
            "default_evaluator = tsg_plugins.evaluator_plugin:EvaluatorPlugin", # Ensure this points to your evaluator
        ],
        "optimizer.plugins": [
            # Corrected entry: points to the OptimizerPlugin class in tsg_plugins
            "default_optimizer = tsg_plugins.optimizer_plugin:OptimizerPlugin",
            # If you had an old entry like "default_optimizer = optimizer_plugins.default_optimizer:Plugin", remove or comment it out.
        ],
        "trainer.plugins": [
            "gan_trainer = tsg_plugins.gan_trainer_plugin.gan_trainer_plugin:GANTrainerPlugin",
        ],
    },

    include_package_data=True,
    project_urls={
        "Source": "https://github.com/harveybc/timeseries-gan",
        "Tracker": "https://github.com/harveybc/timeseries-gan/issues",
    },
)
