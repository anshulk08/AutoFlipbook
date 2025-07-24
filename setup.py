"""
Setup script for Brain Tumor Flipbook Pipeline
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="brain-tumor-flipbooks",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Automated pipeline for generating digital flipbooks from longitudinal brain tumor MRI scans",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/brain-tumor-flipbooks",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/brain-tumor-flipbooks/issues",
        "Documentation": "https://github.com/yourusername/brain-tumor-flipbooks/wiki",
        "Source Code": "https://github.com/yourusername/brain-tumor-flipbooks",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "brain-flipbooks=brain_flipbook_pipeline.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "brain_flipbook_pipeline": [
            "examples/*.py",
            "examples/data/*",
            "templates/*.html",
        ],
    },
    keywords=[
        "neuroimaging",
        "brain tumors",
        "MRI",
        "medical imaging",
        "image registration",
        "longitudinal analysis",
        "flipbooks",
        "visualization",
        "FSL",
        "FLIRT",
    ],
    zip_safe=False,
)