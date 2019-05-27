# Imports ==========

from setuptools import setup, find_packages

# Setup ==========

setup(
    name='genetic_algorithm',
    version='0.1',
    description='Gentic algorithm for ML hyperparameter tuning of scikit-learn pipelines',
    url='https://github.com/elicutler/genetic_algorithm',
    author='Eli Cutler',
    author_email='cutler.eli@gmail.com',
    license='MIT',
    packages=find_packages(),
    zip_safe=False
)