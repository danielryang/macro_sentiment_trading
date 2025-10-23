# Contributing to Macro Sentiment Trading

Thank you for your interest in contributing to this project! This document provides guidelines for contributions.

## Code of Conduct

- Be respectful and constructive
- Focus on improving the codebase
- Help others learn and grow

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/danielryang/macro_sentiment_trading/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version)
   - Relevant logs from `logs/cli.log`

### Suggesting Enhancements

1. Check existing issues and pull requests
2. Create an issue describing:
   - The enhancement and its benefits
   - Possible implementation approach
   - Any potential drawbacks

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**:
   - Follow the code style (see below)
   - Add tests for new functionality
   - Update documentation as needed

4. **Test your changes**:
   ```bash
   # Run test suite
   python -m pytest tests/
   
   # Test CLI commands
   python cli/main.py status
   python cli/main.py get-signals --assets EURUSD
   ```

5. **Commit your changes**:
   ```bash
   git commit -m "Add feature: brief description"
   ```
   
   Use clear, descriptive commit messages:
   - `fix: Bug description`
   - `feat: New feature description`
   - `docs: Documentation update`
   - `test: Test addition/modification`
   - `refactor: Code refactoring`

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**:
   - Provide clear description of changes
   - Reference any related issues
   - Explain testing performed

## Code Style

### Python Style Guide

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use [Black](https://github.com/psf/black) for formatting (line length: 100)
- Use type hints where appropriate
- Write docstrings for functions and classes

```python
def calculate_sentiment(headlines: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Calculate sentiment scores for news headlines.
    
    Args:
        headlines: List of news headline strings
        batch_size: Number of headlines to process in each batch
        
    Returns:
        Array of sentiment scores (-1 to 1)
    """
    pass
```

### Code Organization

- Keep functions focused and under 50 lines when possible
- Use meaningful variable names
- Add comments for complex logic
- Avoid premature optimization

### Testing

- Write unit tests for new features
- Maintain test coverage for critical paths
- Use descriptive test names

```python
def test_sentiment_analyzer_handles_empty_input():
    """Test that sentiment analyzer gracefully handles empty headline list."""
    pass
```

## Development Setup

1. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pytest black flake8 mypy
   ```

2. **Install pre-commit hooks** (optional):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

3. **Run tests before committing**:
   ```bash
   # Unit tests
   pytest tests/
   
   # Style checks
   black src/ cli/ tests/
   flake8 src/ cli/ tests/
   
   # Type checks
   mypy src/ cli/
   ```

## Project Structure

```
macro_sentiment_trading/
├── src/               # Core pipeline modules
├── cli/               # Command-line interface
├── tests/             # Test suite
├── notebooks/         # Jupyter notebooks for analysis
├── data/              # Data storage (not in repo)
├── results/           # Model outputs (models in Git LFS)
└── docs/              # Additional documentation
```

## Areas for Contribution

### High Priority

- Additional asset support (new currency pairs, commodities)
- Performance optimizations for large datasets
- Additional technical indicators
- Improved caching strategies
- Documentation improvements

### Medium Priority

- Web interface for signal visualization
- Real-time data streaming
- Additional ML models (neural networks, ensembles)
- Backtesting enhancements
- Alert systems

### Lower Priority

- Mobile app integration
- Cloud deployment scripts
- Kubernetes configuration
- Additional visualization types

## Questions?

- Open an issue for questions: https://github.com/danielryang/macro_sentiment_trading/issues
- Check existing documentation:
  - README.md - Usage guide
  - CLAUDE.md - Development notes
  - TRADING_QUICKSTART.md - Quick start

## License

By contributing, you agree that your contributions will be licensed under the MIT License.


