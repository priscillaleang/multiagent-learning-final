from setuptools import setup, find_packages

setup(
    name="multiagent-learning",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "open-spiel>=1.2",
        "tensorflow>=2.12.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
    ],
)
