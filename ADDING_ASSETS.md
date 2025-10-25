# How to Add New Assets

The system now supports **60+ tradeable assets** across 8 categories with a modular, easy-to-expand configuration.

## Quick Start

To add new assets, simply edit `src/market_processor.py` lines 49-155.

### Step-by-Step:

1. **Open** `src/market_processor.py`
2. **Find** the appropriate category dictionary (e.g., `major_fx`, `crypto`, `us_indices`)
3. **Add** your asset in the format: `'ASSET_NAME': 'YAHOO_TICKER'`
4. **Save** and run!

## Example: Adding a New Cryptocurrency

```python
# In src/market_processor.py, find the crypto section:
crypto = {
    'BTCUSD': 'BTC-USD',   # Bitcoin
    'ETHUSD': 'ETH-USD',   # Ethereum
    # Add your new crypto here:
    'AVAXUSD': 'AVAX-USD', # Avalanche  ← NEW!
}
```

## Example: Adding a New Stock

```python
# In src/market_processor.py, find the mega_cap_stocks section:
mega_cap_stocks = {
    'AAPL': 'AAPL',        # Apple
    'MSFT': 'MSFT',        # Microsoft
    # Add your new stock here:
    'NFLX': 'NFLX',        # Netflix  ← NEW!
}
```

## Supported Ticker Formats

The system automatically handles ALL Yahoo Finance ticker formats:

| Format | Example | Asset Type |
|--------|---------|------------|
| `XXX=X` | `EURUSD=X` | Major FX pairs |
| `XXX-USD` | `BTC-USD` | Cryptocurrencies |
| `XX=F` | `GC=F` | Futures (commodities, treasuries) |
| Plain | `AAPL`, `SPY` | Stocks and ETFs |

## Finding Yahoo Finance Tickers

1. Go to https://finance.yahoo.com/
2. Search for your asset
3. The ticker is in the URL: `finance.yahoo.com/quote/TICKER`

**Examples:**
- Bitcoin: `BTC-USD`
- Apple Stock: `AAPL`
- Gold Futures: `GC=F`
- EUR/USD: `EURUSD=X`

## Current Asset Inventory

### 7 Major FX Pairs
- EURUSD, USDJPY, GBPUSD, AUDUSD, USDCHF, USDCAD, NZDUSD

### 6 Cross FX Pairs
- EURGBP, EURJPY, GBPJPY, EURCHF, AUDJPY, EURAUD

### 6 Emerging Market FX
- USDMXN, USDBRL, USDZAR, USDTRY, USDINR, USDCNY

### 10 Cryptocurrencies
- BTCUSD, ETHUSD, BNBUSD, XRPUSD, ADAUSD, SOLUSD, DOTUSD, DOGEUSD, MATICUSD, LINKUSD

### 10 US Equity Indices (ETFs)
- SPY, QQQ, DIA, IWM, VTI, VOO, IVV, VEA, VWO, EFA

### 10 Mega Cap Stocks
- AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, BRK-B, JPM, V

### 8 Commodities (Futures)
- GOLD, SILVER, CRUDE, BRENT, NATGAS, COPPER, CORN, WHEAT

### 3 Fixed Income (Treasury Futures)
- TNOTE, TBOND, TFIVE

## CLI Usage

Train models for specific assets:
```bash
python cli/main.py run-pipeline --assets BTCUSD AAPL GOLD --start-date 2024-01-01 --end-date 2024-12-31
```

Generate signals for all assets:
```bash
python cli/main.py get-signals  # Uses all 60 assets
```

Generate signals for specific categories:
```bash
python cli/main.py get-signals --assets EURUSD GBPUSD USDJPY  # Just FX
python cli/main.py get-signals --assets BTCUSD ETHUSD SOLUSD  # Just crypto
```

## Notes

- **All ticker formats tested**: FX (=X), Crypto (-USD), Futures (=F), and plain tickers
- **OHLCV filtering works**: All assets correctly filter raw price/volume columns
- **No code changes needed**: Just add to the dictionary!
- **Category tracking**: Assets are automatically organized by category

## Validation

All assets have been validated with:
- ✅ Data download from Yahoo Finance
- ✅ Technical indicator computation (158 TA-Lib indicators)
- ✅ Feature engineering (674 features per asset)
- ✅ OHLCV filtering (0 raw price columns in features)

**Total: 60 assets ready for trading signals!** 🚀
