# Changelog

All notable changes to the Macro Sentiment Trading project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-XX

### Initial Production Release

#### Added
- **Core Features**
  - Complete CLI interface with 14 commands for pipeline operations
  - Real-time trading signal generation from pre-trained models
  - Multi-timeframe backtesting (1D, 1W, 1M, 1Q, 1Y)
  - Support for 35+ financial assets (FX, crypto, equities, commodities)
  - Production-grade performance metrics (40+ institutional metrics)
  - SHAP-based model interpretability and feature analysis
  
- **Data Pipeline**
  - GDELT news data collection (free API and BigQuery)
  - Intelligent caching system with cross-method compatibility
  - FinBERT sentiment analysis with batch processing
  - Yahoo Finance market data integration
  - Robust error handling and automatic recovery

- **Machine Learning**
  - XGBoost and Logistic Regression ensemble models
  - Expanding window backtesting (prevents look-ahead bias)
  - 70+ engineered features (sentiment, technical, temporal)
  - Hyperparameter optimization with Optuna
  - TA-Lib integration (158 technical indicators)

- **Visualizations**
  - Alpha analytics dashboards
  - Performance tearsheets
  - Sentiment analysis plots
  - SHAP feature importance visualizations

- **Documentation**
  - Comprehensive README with usage examples
  - CLI command reference
  - Trading quickstart guide
  - Development guidelines (CLAUDE.md)
  - Contributing guidelines
  - MIT License

#### Technical Details
- Python 3.8+ support
- Git LFS for model storage (113MB pre-trained models)
- Comprehensive .gitignore for clean repository
- setuptools packaging for pip installation

### Research Foundation
- Based on arXiv:2505.16136v1 "Macro Sentiment Trading"
- FinBERT implementation from arXiv:1908.10063
- Enhanced with multi-timeframe analysis and institutional metrics

### Performance
- Realistic Sharpe ratios: 0.0-0.68 (production-grade)
- Multi-asset universe: Major FX, crypto, equities, commodities
- Transaction cost integration: 2bp (FX), 5bp (futures)
- BigQuery cache support: 10-year historical data (146K+ events)

### Known Limitations
- Minimum 6-month date range required for reliable training
- Single target class errors on very small datasets
- BigQuery requires Google Cloud credentials (optional)

---

## Future Releases (Planned)

### [1.1.0] - TBD
- Neural network models (LSTM, Transformer)
- Real-time data streaming
- Web dashboard for signal monitoring
- Additional technical indicators
- Enhanced alert systems

### [1.2.0] - TBD
- Multi-model ensemble strategies
- Portfolio optimization
- Risk management tools
- Cloud deployment templates
- API endpoints for programmatic access

---

## Version History

- **1.0.0** (2025-01-XX): Initial production release
- **0.1.0** (Development): Internal development version

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute to this project.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.







