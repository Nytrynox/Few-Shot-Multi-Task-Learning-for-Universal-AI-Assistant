from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="universal-ai-assistant",
    version="0.1.0",
    author="Vishnu Project Team",
    author_email="team@vishnu-project.org",
    description="Few-Shot Multi-Task Learning for Universal AI Assistant",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vishnu-project/universal-ai-assistant",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
            "pre-commit>=3.3.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "plotly>=5.14.0",
        ],
        "audio": [
            "torchaudio>=2.0.0",
            "librosa>=0.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "universal-assistant=src.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
)
