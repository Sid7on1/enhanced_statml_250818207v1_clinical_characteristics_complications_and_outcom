import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from typing import List, Dict, Any

# Define constants
PROJECT_NAME = "enhanced_stat.ML_2508.18207v1_Clinical_characteristics_complications_and_outcom"
PROJECT_VERSION = "1.0.0"
PROJECT_AUTHOR = "Your Name"
PROJECT_EMAIL = "your@email.com"
PROJECT_LICENSE = "MIT"
PROJECT_URL = "https://github.com/your-username/your-repo-name"

# Define dependencies
DEPENDENCIES: List[str] = [
    "torch",
    "numpy",
    "pandas",
    # Add other dependencies here
]

# Define development dependencies
DEV_DEPENDENCIES: List[str] = [
    "pytest",
    "flake8",
    # Add other development dependencies here
]

# Define test dependencies
TEST_DEPENDENCIES: List[str] = [
    "pytest",
    # Add other test dependencies here
]

# Define documentation dependencies
DOC_DEPENDENCIES: List[str] = [
    "sphinx",
    # Add other documentation dependencies here
]

# Define the package data
PACKAGE_DATA: Dict[str, List[str]] = {
    "": ["*.txt", "*.md", "*.json"],
    "enhanced_stat.ML_2508.18207v1_Clinical_characteristics_complications_and_outcom": ["data/*"],
}

# Define the package directories
PACKAGE_DIRS: List[str] = [
    "enhanced_stat.ML_2508.18207v1_Clinical_characteristics_complications_and_outcom",
]

# Define the entry points
ENTRY_POINTS: Dict[str, List[str]] = {
    "console_scripts": [
        "enhanced_stat.ML_2508.18207v1_Clinical_characteristics_complications_and_outcom=enhanced_stat.ML_2508.18207v1_Clinical_characteristics_complications_and_outcom.main:main",
    ],
}

# Define the classifiers
CLASSIFIERS: List[str] = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
]

# Define the project long description
with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

# Define the setup function
def setup_package():
    setup(
        name=PROJECT_NAME,
        version=PROJECT_VERSION,
        author=PROJECT_AUTHOR,
        author_email=PROJECT_EMAIL,
        license=PROJECT_LICENSE,
        url=PROJECT_URL,
        description="Enhanced AI project based on stat.ML_2508.18207v1_Clinical-characteristics-complications-and-outcom with content analysis.",
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        packages=find_packages(),
        package_data=PACKAGE_DATA,
        package_dir={"": "src"},
        install_requires=DEPENDENCIES,
        extras_require={
            "dev": DEV_DEPENDENCIES,
            "test": TEST_DEPENDENCIES,
            "doc": DOC_DEPENDENCIES,
        },
        entry_points=ENTRY_POINTS,
        classifiers=CLASSIFIERS,
        python_requires=">=3.8",
    )

# Run the setup function
if __name__ == "__main__":
    setup_package()