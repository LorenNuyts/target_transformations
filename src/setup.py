from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy as np

setup(
    name='target_transformation',
    version='1.0',
    packages=find_packages(),
    include_package_data=True,
    package_data={},
    setup_requires=[
        'setuptools',
        'numpy>=1.26.4'
    ],
    install_requires=[
        'six>=1.16.0',
        'scipy>=1.13.0',
        'scikit-learn>=1.4.2',
        'matplotlib==3.8.4',
        'pandas>=1.2.2',
    ],
)
