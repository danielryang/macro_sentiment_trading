"""
Setup configuration for macro sentiment trading pipeline
"""
from setuptools import setup, find_packages

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt, excluding comments and optional dependencies."""
    requirements = []
    try:
        with open('requirements.txt', 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments, empty lines, and optional dependencies
                if (line and 
                    not line.startswith('#') and 
                    not line.startswith('openai') and 
                    not line.startswith('anthropic') and
                    not line.startswith('pytest') and
                    not line.startswith('black') and
                    not line.startswith('flake8') and
                    not line.startswith('mypy')):
                    requirements.append(line)
    except FileNotFoundError:
        # Fallback to minimal requirements if file not found
        requirements = [
            "pandas>=1.5.0",
            "numpy>=1.21.0",
            "torch>=1.9.0",
            "transformers>=4.20.0",
            "scikit-learn>=1.1.0",
            "xgboost>=1.6.0",
            "yfinance>=0.1.70",
            "gdelt>=0.1.14",
            "shap>=0.41.0",
            "requests>=2.25.0",
            "beautifulsoup4>=4.10.0",
            "python-dotenv>=0.19.0",
            "tqdm>=4.60.0",
        ]
    return requirements

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="macro_sentiment_trading",
    version="1.0.0",
    description="Production-ready macro sentiment trading pipeline using GDELT, FinBERT, and ML models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Macro Sentiment Trading Project",
    author_email="",
    url="https://github.com/danielryang/macro_sentiment_trading",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "llm": [
            "openai>=1.0.0",
            "anthropic>=0.3.0",
        ],
        "bigquery": [
            "google-cloud-bigquery>=3.0.0",
            "google-cloud-bigquery-storage>=2.0.0",
            "db-dtypes>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "macro-sentiment-trading=src.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords="trading, sentiment-analysis, machine-learning, finance, nlp, finbert, gdelt, algorithmic-trading, quantitative-finance",
    project_urls={
        "Bug Reports": "https://github.com/danielryang/macro_sentiment_trading/issues",
        "Documentation": "https://github.com/danielryang/macro_sentiment_trading#readme",
        "Source Code": "https://github.com/danielryang/macro_sentiment_trading",
    },
)