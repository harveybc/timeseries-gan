from setuptools import setup, find_packages

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='timeseries-gan',
    version='0.1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'timeseries_gan=app.main:main'
        ],
        'timeseries_gan.optimizers': [
            'default=app.plugins.optimizer_default:Plugin'
            # Add other optimizer plugins here
        ],
        'timeseries_gan.generators': [
            'default=app.plugins.generator_default:Plugin'
            # Add other generator plugins here
        ],
        'timeseries_gan.discriminators': [
            'default=app.plugins.discriminator_default:Plugin'
            # Add other discriminator plugins here
        ]
    },
    install_requires=[
    ],
    author='Harvey Bastidas',
    author_email='your.email@example.com',
    description='Generative adversarial network for timeseries.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/harveybc/timeseries-gan',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='synthetic data, generative machine learning, plugin architecture',
    python_requires='>=3.6',
    include_package_data=True,  # Include non-code files specified in MANIFEST.in
    zip_safe=False,  # Set to False if your package needs to access data files
)
