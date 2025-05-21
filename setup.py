from setuptools import setup, find_packages

setup(
    name="rdoc_package",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="RDOC Package for behavioral task analysis",
    python_requires=">=3.8",
) 