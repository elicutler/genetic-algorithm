# Imports ==========

from setuptools import setup, find_packages

# Setup ==========

setup(
    name='genetic-algorithm',
    version='0.1',
    description='Gentic algorithm for ML hyperparameter tuning of scikit-learn pipelines',
    url='https://github.com/elicutler/genetic-algorithm',
    author='Eli Cutler',
    author_email='cutler.eli@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'sklearn', 'xgboost'],
    zip_safe=False
)
