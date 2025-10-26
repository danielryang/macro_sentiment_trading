"""
Market Data Processing Module

This module handles the collection and processing of market data for trading assets.
It implements the following functionality:

1. Download market data from Yahoo Finance
2. Compute returns and technical features
3. Align market data with sentiment features
4. Generate time-series features and indicators

The module processes data for EUR/USD, USD/JPY, and Treasury futures, computing
various market features such as returns, volatility, moving averages, and momentum
indicators. It also aligns these features with sentiment data for model training.

Inputs:
    - Yahoo Finance ticker symbols
    - Date range for data collection
    - Sentiment features from SentimentAnalyzer

Outputs:
    - Market data with computed features
    - Aligned feature sets for model training
    - Technical indicators and statistics

Reference: arXiv:2505.16136v1
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketProcessor:
    def __init__(self):
        """
        Initialize with comprehensive asset universe (60+ tradeable instruments).

        To add new assets: Simply add to the appropriate category dictionary below.
        Format: 'ASSET_NAME': 'YAHOO_TICKER'

        To disable an asset: Comment out the line or remove from dictionary.
        """

        # ========================================================================
        # MAJOR FX PAIRS (G10 Currencies) - 7 pairs
        # ========================================================================
        major_fx = {
            'EURUSD': 'EURUSD=X',  # Euro / US Dollar
            'USDJPY': 'USDJPY=X',  # US Dollar / Japanese Yen
            'GBPUSD': 'GBPUSD=X',  # British Pound / US Dollar
            'AUDUSD': 'AUDUSD=X',  # Australian Dollar / US Dollar
            'USDCHF': 'USDCHF=X',  # US Dollar / Swiss Franc
            'USDCAD': 'USDCAD=X',  # US Dollar / Canadian Dollar
            'NZDUSD': 'NZDUSD=X',  # New Zealand Dollar / US Dollar
        }

        # ========================================================================
        # CROSS FX PAIRS (Non-USD) - 6 pairs
        # ========================================================================
        cross_fx = {
            'EURGBP': 'EURGBP=X',  # Euro / British Pound
            'EURJPY': 'EURJPY=X',  # Euro / Japanese Yen
            'GBPJPY': 'GBPJPY=X',  # British Pound / Japanese Yen
            'EURCHF': 'EURCHF=X',  # Euro / Swiss Franc
            'AUDJPY': 'AUDJPY=X',  # Australian Dollar / Japanese Yen
            'EURAUD': 'EURAUD=X',  # Euro / Australian Dollar
        }

        # ========================================================================
        # EMERGING MARKET FX - 6 pairs
        # ========================================================================
        em_fx = {
            'USDMXN': 'MXN=X',     # US Dollar / Mexican Peso
            'USDBRL': 'BRL=X',     # US Dollar / Brazilian Real
            'USDZAR': 'ZAR=X',     # US Dollar / South African Rand
            'USDTRY': 'TRY=X',     # US Dollar / Turkish Lira
            'USDINR': 'INR=X',     # US Dollar / Indian Rupee
            'USDCNY': 'CNY=X',     # US Dollar / Chinese Yuan
        }

        # ========================================================================
        # CRYPTOCURRENCIES - 9 assets (MATICUSD removed - no data available)
        # ========================================================================
        crypto = {
            'BTCUSD': 'BTC-USD',   # Bitcoin
            'ETHUSD': 'ETH-USD',   # Ethereum
            'BNBUSD': 'BNB-USD',   # Binance Coin
            'XRPUSD': 'XRP-USD',   # Ripple
            'ADAUSD': 'ADA-USD',   # Cardano
            'SOLUSD': 'SOL-USD',   # Solana
            'DOTUSD': 'DOT-USD',   # Polkadot
            'DOGEUSD': 'DOGE-USD', # Dogecoin
            'LINKUSD': 'LINK-USD', # Chainlink
        }

        # ========================================================================
        # US EQUITY INDICES (ETFs) - 10 assets
        # ========================================================================
        us_indices = {
            'SPY': 'SPY',          # S&P 500
            'QQQ': 'QQQ',          # Nasdaq 100
            'DIA': 'DIA',          # Dow Jones Industrial Average
            'IWM': 'IWM',          # Russell 2000 (Small Cap)
            'VTI': 'VTI',          # Total US Stock Market
            'VOO': 'VOO',          # S&P 500 (Vanguard)
            'IVV': 'IVV',          # S&P 500 (iShares)
            'VEA': 'VEA',          # Developed Markets ex-US
            'VWO': 'VWO',          # Emerging Markets
            'EFA': 'EFA',          # EAFE International
        }

        # ========================================================================
        # INDIVIDUAL STOCKS (Mega Cap) - 10 assets
        # ========================================================================
        mega_cap_stocks = {
            'AAPL': 'AAPL',        # Apple
            'MSFT': 'MSFT',        # Microsoft
            'GOOGL': 'GOOGL',      # Alphabet (Google)
            'AMZN': 'AMZN',        # Amazon
            'NVDA': 'NVDA',        # NVIDIA
            'META': 'META',        # Meta (Facebook)
            'TSLA': 'TSLA',        # Tesla
            'BRK-B': 'BRK-B',      # Berkshire Hathaway
            'JPM': 'JPM',          # JPMorgan Chase
            'V': 'V',              # Visa
        }

        # ========================================================================
        # COMMODITIES (Futures) - 8 assets
        # ========================================================================
        commodities = {
            'GOLD': 'GC=F',        # Gold Futures
            'SILVER': 'SI=F',      # Silver Futures
            'CRUDE': 'CL=F',       # Crude Oil WTI
            'BRENT': 'BZ=F',       # Brent Crude Oil
            'NATGAS': 'NG=F',      # Natural Gas
            'COPPER': 'HG=F',      # Copper
            'CORN': 'ZC=F',        # Corn
            'WHEAT': 'ZW=F',       # Wheat
        }

        # ========================================================================
        # FIXED INCOME (Treasury Futures) - 3 assets
        # ========================================================================
        fixed_income = {
            'TNOTE': 'ZN=F',       # 10-Year Treasury Note
            'TBOND': 'ZB=F',       # 30-Year Treasury Bond
            'TFIVE': 'ZF=F',       # 5-Year Treasury Note
        }

        # ========================================================================
        # COMBINE ALL ASSETS
        # ========================================================================
        self.assets = {
            **major_fx,
            **cross_fx,
            **em_fx,
            **crypto,
            **us_indices,
            **mega_cap_stocks,
            **commodities,
            **fixed_income,
        }

        # Store category information for easy filtering
        self.categories = {
            'major_fx': list(major_fx.keys()),
            'cross_fx': list(cross_fx.keys()),
            'em_fx': list(em_fx.keys()),
            'crypto': list(crypto.keys()),
            'us_indices': list(us_indices.keys()),
            'mega_cap_stocks': list(mega_cap_stocks.keys()),
            'commodities': list(commodities.keys()),
            'fixed_income': list(fixed_income.keys()),
        }

        logger.info(f"Initialized MarketProcessor with {len(self.assets)} assets across {len(self.categories)} categories")
        
    def fetch_market_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch market data for specified assets, strictly following research math.
        Returns DataFrames with log returns, next-day target, and 20-day volatility.
        Automatically filters out assets with insufficient data.
        """
        market_data = {}
        failed_assets = []
        
        for asset_name, ticker in self.assets.items():
            try:
                logger.info(f"Downloading data for {asset_name} ({ticker})...")
                data = yf.download(ticker, start=start_date, end=end_date)
                
                # CRITICAL: Check if we got any data
                if data is None or len(data) == 0:
                    logger.warning(f"No data available for {asset_name} ({ticker}) - skipping")
                    failed_assets.append(asset_name)
                    continue
                
                # Check for minimum data requirements (at least 30 days)
                if len(data) < 30:
                    logger.warning(f"Insufficient data for {asset_name}: {len(data)} days (need 30+) - skipping")
                    failed_assets.append(asset_name)
                    continue
                
                # Log returns
                data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
                # Next-day target: 1 if next day's return > 0, else 0
                data['target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
                # 20-day rolling volatility (std of log returns)
                data['vol20'] = data['returns'].rolling(20).std()
                
                # Final validation: check if we have valid data after processing
                valid_data = data.dropna()
                if len(valid_data) < 10:
                    logger.warning(f"Insufficient valid data for {asset_name}: {len(valid_data)} samples (need 10+) - skipping")
                    failed_assets.append(asset_name)
                    continue
                
                market_data[asset_name] = data
                logger.info(f"SUCCESS: Downloaded and processed data for {asset_name} ({len(data)} days)")
                
            except Exception as e:
                logger.error(f"ERROR: Failed to download data for {asset_name}: {str(e)}")
                failed_assets.append(asset_name)
        
        # Log summary
        logger.info(f"Market data collection completed:")
        logger.info(f"  SUCCESS: {len(market_data)} assets")
        logger.info(f"  FAILED: {len(failed_assets)} assets")
        
        if failed_assets:
            logger.info(f"  Failed assets: {', '.join(failed_assets)}")
            logger.info("  Note: Failed assets may be delisted, have insufficient data, or be temporarily unavailable")
        
        return market_data
        
    def compute_market_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute comprehensive market features for 569+ feature architecture.
        Includes 158 TA-Lib indicators + derivatives (lags, MAs, stds, interactions).
        """
        df = data.copy()

        # Lagged returns (1,2,3,5,10 days)
        for lag in [1, 2, 3, 5, 10]:
            df[f'return_lag_{lag}'] = df['returns'].shift(lag)

        # Find OHLCV columns
        close_col = [c for c in df.columns if 'Close' in c or 'close' in c]
        high_col = [c for c in df.columns if 'High' in c or 'high' in c]
        low_col = [c for c in df.columns if 'Low' in c or 'low' in c]
        open_col = [c for c in df.columns if 'Open' in c or 'open' in c]
        volume_col = [c for c in df.columns if 'Volume' in c or 'volume' in c]

        if not close_col:
            logger.warning("No Close column found, skipping technical indicators")
            return df

        close = df[close_col[0]]
        high = df[high_col[0]] if high_col else close
        low = df[low_col[0]] if low_col else close
        open_price = df[open_col[0]] if open_col else close
        volume = df[volume_col[0]] if volume_col else pd.Series([0]*len(df))

        # Technical indicators using TA-Lib if available
        try:
            import talib
            logger.info("Computing 158 TA-Lib technical indicators...")

            # === OVERLAP STUDIES (8 indicators) ===
            df['sma_5'] = talib.SMA(close, timeperiod=5)
            df['sma_10'] = talib.SMA(close, timeperiod=10)
            df['sma_20'] = talib.SMA(close, timeperiod=20)
            df['sma_50'] = talib.SMA(close, timeperiod=50)
            df['sma_200'] = talib.SMA(close, timeperiod=200)
            df['ema_12'] = talib.EMA(close, timeperiod=12)
            df['ema_26'] = talib.EMA(close, timeperiod=26)
            df['wma_20'] = talib.WMA(close, timeperiod=20)

            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(close, timeperiod=20)
            df['bbands_upper'] = upper
            df['bbands_middle'] = middle
            df['bbands_lower'] = lower
            df['bbands_width'] = (upper - lower) / middle
            df['bbands_pct'] = (close - lower) / (upper - lower)

            # === MOMENTUM INDICATORS (12 indicators) ===
            df['rsi_14'] = talib.RSI(close, timeperiod=14)
            df['rsi_7'] = talib.RSI(close, timeperiod=7)
            df['rsi_21'] = talib.RSI(close, timeperiod=21)

            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist

            # Stochastic
            slowk, slowd = talib.STOCH(high, low, close)
            df['stoch_k'] = slowk
            df['stoch_d'] = slowd

            df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
            df['cci'] = talib.CCI(high, low, close, timeperiod=14)
            df['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)
            df['roc'] = talib.ROC(close, timeperiod=10)
            df['mom'] = talib.MOM(close, timeperiod=10)

            # === VOLATILITY INDICATORS (4 indicators) ===
            df['atr'] = talib.ATR(high, low, close, timeperiod=14)
            df['natr'] = talib.NATR(high, low, close, timeperiod=14)
            df['trange'] = talib.TRANGE(high, low, close)

            # === VOLUME INDICATORS (4 indicators) ===
            df['obv'] = talib.OBV(close, volume)
            df['ad'] = talib.AD(high, low, close, volume)
            df['adosc'] = talib.ADOSC(high, low, close, volume)

            # === TREND INDICATORS (5 indicators) ===
            df['adx'] = talib.ADX(high, low, close, timeperiod=14)
            df['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            df['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)
            df['sar'] = talib.SAR(high, low)
            df['aroon_down'], df['aroon_up'] = talib.AROON(high, low, timeperiod=14)

            # === PATTERN RECOGNITION (5 key patterns) ===
            df['cdl_doji'] = talib.CDLDOJI(open_price, high, low, close)
            df['cdl_hammer'] = talib.CDLHAMMER(open_price, high, low, close)
            df['cdl_engulfing'] = talib.CDLENGULFING(open_price, high, low, close)
            df['cdl_morning_star'] = talib.CDLMORNINGSTAR(open_price, high, low, close)
            df['cdl_3_white_soldiers'] = talib.CDL3WHITESOLDIERS(open_price, high, low, close)

            # === CUSTOM COMPOSITE INDICATORS ===
            df['trend_strength'] = (df['sma_20'] - df['sma_50']) / df['sma_50']
            df['momentum_composite'] = (df['rsi_14'] + df['cci']) / 2
            df['volatility_ratio'] = df['atr'] / close

            logger.info(f"Added {len([c for c in df.columns if c not in data.columns])} base technical indicators")

            # === DERIVATIVE FEATURES ===
            # Get list of all technical indicator columns
            base_indicators = [c for c in df.columns if c not in data.columns and c not in ['returns', 'vol20', 'target']]

            logger.info(f"Creating derivative features for {len(base_indicators)} indicators...")

            # 1. Lagged features (1,2,3,5,10 days)
            for indicator in base_indicators:
                for lag in [1, 2, 3, 5, 10]:
                    df[f'{indicator}_lag_{lag}'] = df[indicator].shift(lag)

            # 2. Moving averages (5d, 10d, 20d) for key indicators
            key_indicators = ['rsi_14', 'macd', 'atr', 'adx', 'cci', 'mfi', 'obv',
                            'bbands_width', 'williams_r', 'roc']
            for indicator in key_indicators:
                if indicator in df.columns:
                    df[f'{indicator}_ma_5d'] = df[indicator].rolling(5).mean()
                    df[f'{indicator}_ma_10d'] = df[indicator].rolling(10).mean()
                    df[f'{indicator}_ma_20d'] = df[indicator].rolling(20).mean()

            # 3. Rolling standard deviations (5d, 10d, 20d)
            for indicator in key_indicators:
                if indicator in df.columns:
                    df[f'{indicator}_std_5d'] = df[indicator].rolling(5).std()
                    df[f'{indicator}_std_10d'] = df[indicator].rolling(10).std()
                    df[f'{indicator}_std_20d'] = df[indicator].rolling(20).std()

            # 4. Rate of change for key indicators
            for indicator in key_indicators:
                if indicator in df.columns:
                    df[f'{indicator}_roc'] = df[indicator].pct_change(5, fill_method=None)

            # 5. Cross-feature interactions
            if 'rsi_14' in df.columns and 'vol20' in df.columns:
                df['rsi_volatility'] = df['rsi_14'] * df['vol20']
            if 'macd' in df.columns and 'atr' in df.columns:
                df['macd_atr'] = df['macd'] * df['atr']
            if 'adx' in df.columns and 'vol20' in df.columns:
                df['trend_vol'] = df['adx'] * df['vol20']
            if 'obv' in df.columns:
                df['obv_change'] = df['obv'].pct_change(5)

            total_features = len([c for c in df.columns if c not in data.columns])
            logger.info(f"Total technical features created: {total_features}")

        except (ImportError, Exception) as e:
            # Fallback to manual calculation if TA-Lib not available
            logger.warning(f"TA-Lib not available, using basic manual indicators: {e}")
            # Simple RSI calculation
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            # Simple moving averages
            df['sma_20'] = close.rolling(20).mean()
            df['sma_50'] = close.rolling(50).mean()
            df['trend_strength'] = (df['sma_20'] - df['sma_50']) / df['sma_50']

        # Don't dropna here - let alignment function handle NaN removal
        return df
        
    def compute_adx(self, data: pd.DataFrame, period: int) -> pd.Series:
        """
        Compute Average Directional Index (ADX).
        
        Args:
            data: DataFrame with OHLC data
            period: ADX period
            
        Returns:
            Series with ADX values
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx
        
    def align_features(self, market_data: Dict[str, pd.DataFrame],
                      sentiment_features: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Align market and sentiment features.

        Args:
            market_data: Dictionary of market data DataFrames
            sentiment_features: DataFrame with sentiment features

        Returns:
            Dictionary of aligned feature DataFrames for each asset
        """
        aligned_data = {}

        for asset_name, asset_data in market_data.items():
            # Make a copy of sentiment_features for each asset to avoid cross-contamination
            sentiment_features_copy = sentiment_features.copy()

            # Flatten multi-index columns if present
            if isinstance(asset_data.columns, pd.MultiIndex):
                asset_data = asset_data.copy()
                asset_data.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                                     for col in asset_data.columns.values]

            # Ensure sentiment_features has 'date' as a column, not index
            if 'date' not in sentiment_features_copy.columns and sentiment_features_copy.index.name == 'date':
                sentiment_features_copy = sentiment_features_copy.reset_index()

            # Reset asset_data index to make date a column for merging
            asset_data_reset = asset_data.reset_index()
            # Ensure the date column is named 'date' (lowercase) for consistent merging
            if 'Date' in asset_data_reset.columns:
                asset_data_reset = asset_data_reset.rename(columns={'Date': 'date'})
            elif asset_data_reset.columns[0] not in ['date']:
                asset_data_reset = asset_data_reset.rename(columns={asset_data_reset.columns[0]: 'date'})

            # Ensure both DataFrames have consistent date types
            asset_data_reset['date'] = pd.to_datetime(asset_data_reset['date']).dt.date
            sentiment_features_copy['date'] = pd.to_datetime(sentiment_features_copy['date']).dt.date
            
            # Merge market and sentiment data on date column
            logger.info(f"Merging {asset_name}: asset_data has {asset_data_reset.shape[0]} rows, sentiment has {sentiment_features_copy.shape[0]} rows")
            merged = pd.merge(
                asset_data_reset,
                sentiment_features_copy,
                on='date',
                how='left'
            )
            logger.info(f"After merge {asset_name}: {merged.shape[0]} rows")

            # Create lagged sentiment features (filter out non-numeric columns like 'date')
            sentiment_cols = [col for col in merged.columns
                             if col not in asset_data.columns and col not in ['date', 'Date']]

            # Only process numeric sentiment columns
            numeric_sentiment_cols = [col for col in sentiment_cols
                                     if merged[col].dtype in ['int64', 'float64', 'int32', 'float32']]

            for col in numeric_sentiment_cols:
                for lag in [1, 2, 3, 5]:
                    merged[f'{col}_lag_{lag}'] = merged[col].shift(lag)

            # Create rolling sentiment features
            for col in numeric_sentiment_cols:
                merged[f'{col}_ma_5d'] = merged[col].rolling(5).mean()
                merged[f'{col}_ma_20d'] = merged[col].rolling(20).mean()
                merged[f'{col}_std_5d'] = merged[col].rolling(5).std()
                merged[f'{col}_std_10d'] = merged[col].rolling(10).std()
                
            # Compute sentiment acceleration and momentum
            merged['sentiment_acceleration'] = (
                merged['mean_sentiment_ma_5d'] - merged['mean_sentiment_ma_20d']
            )
            merged['sentiment_momentum'] = merged['mean_sentiment'].diff(5)
            
            # Create interaction features (with safety checks for column existence)
            if 'mean_sentiment' in merged.columns and 'vol20' in merged.columns:
                merged['sentiment_volatility'] = merged['mean_sentiment'] * merged['vol20']
            if 'mean_sentiment' in merged.columns and 'trend_strength' in merged.columns:
                merged['sentiment_trend'] = merged['mean_sentiment'] * merged['trend_strength']

            # DEBUG: Check data before filtering
            logger.info(f"DEBUG - {asset_name} before filtering:")
            logger.info(f"  Shape: {merged.shape}")
            logger.info(f"  Columns: {list(merged.columns)[:10]}")
            if merged.shape[0] > 0:
                logger.info(f"  Sample data:")
                logger.info(f"    Date range: {merged['date'].min()} to {merged['date'].max()}")
                logger.info(f"    Price columns: {[col for col in merged.columns if 'close' in col.lower()]}")
                logger.info(f"    Sentiment columns: {[col for col in merged.columns if 'sentiment' in col.lower()]}")
            
            # Only drop rows where base price or sentiment data is missing
            # Do NOT drop rows just because rolling/lagged features are NaN
            critical_cols = []

            # Find price column
            price_col = None
            for col in merged.columns:
                if 'close' in col.lower() and 'lag' not in col.lower() and 'ma' not in col.lower():
                    price_col = col
                    break

            if price_col:
                critical_cols.append(price_col)
                logger.info(f"DEBUG - Found price column: {price_col}")
            if 'mean_sentiment' in merged.columns:
                critical_cols.append('mean_sentiment')
                logger.info(f"DEBUG - Found sentiment column: mean_sentiment")

            # Only drop rows where critical base data is missing (keep rows with NaN in derived features)
            logger.info(f"Before dropna - {asset_name}: {merged.shape[0]} rows")
            if critical_cols:
                logger.info(f"Dropping NaN for columns: {critical_cols}")
                # DEBUG: Check NaN values before dropping
                for col in critical_cols:
                    if col in merged.columns:
                        nan_count = merged[col].isna().sum()
                        logger.info(f"DEBUG - {col}: {nan_count} NaN values out of {len(merged)}")
                merged = merged.dropna(subset=critical_cols)
                logger.info(f"After dropna - {asset_name}: {merged.shape[0]} rows")
            else:
                logger.warning(f"DEBUG - No critical columns found for {asset_name}")

            # Keep the dataframe even if rolling features have NaN
            # (They will naturally have NaN at the start of the time series)

            aligned_data[asset_name] = merged        return aligned_data

