#!/usr/bin/env python3
"""
Setup script for Vingi Personal Cognitive Load Optimization Framework
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith("#") and not line.startswith("sqlite3")
        ]
else:
    requirements = [
        "numpy>=1.24.0",
        "click>=8.1.0",
        "PyYAML>=6.0.0",
        "python-dateutil>=2.8.0",
    ]

setup(
    name="vingi",
    version="2.0.0-beta",
    author="Vingi Development Team",
    author_email="dev@vingi.ai",
    description="Personal Cognitive Load Optimization Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vingi-ai/vingi",
    project_urls={
        "Documentation": "https://vingi.ai/docs",
        "Source": "https://github.com/vingi-ai/vingi",
        "Bug Reports": "https://github.com/vingi-ai/vingi/issues",
    },
    
    # Package configuration
    packages=find_packages(where="src/python"),
    package_dir={"": "src/python"},
    
    # Dependencies
    install_requires=requirements,
    python_requires=">=3.11",
    
    # Entry points for CLI
    entry_points={
        "console_scripts": [
            "vingi=vingi.cli:cli",
        ],
    },
    
    # Package metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Scheduling",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Monitoring",
    ],
    
    # Additional metadata
    keywords="cognitive load optimization productivity AI assistant personal automation",
    license="MIT",
    
    # Include additional files
    include_package_data=True,
    package_data={
        "vingi": ["py.typed"],
    },
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "full": [
            "transformers>=4.21.0",
            "sentence-transformers>=2.2.0",
            "scikit-learn>=1.3.0",
            "scipy>=1.10.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
        ],
    },
    
    # Minimum requirements check
    zip_safe=False,
) 