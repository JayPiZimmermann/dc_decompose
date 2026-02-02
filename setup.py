from setuptools import setup, find_packages

setup(
    name="dc_decompose",
    version="0.1.0",
    description="DC Decomposition for Neural Networks using PyTorch Hooks",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
    ],
)
