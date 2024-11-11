import glob
import json
import os
from datetime import datetime
import re
import traceback
import ollama
import yaml
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from ollama import generate
from ollama import chat
import talib
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('algo_agent.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MarketAnalysis:
    trend: str
    strength: float
    volatility: float
    support_levels: list
    resistance_levels: list
    volume_profile: dict
    risk_metrics: dict
    indicators: dict

class AlgoAgent:
    def __init__(self, company_name: str):
        self.company_name = company_name
        self.base_path = Path(os.getcwd())
        self.config = self.load_config()
        self.setup_api_client()
        
    def load_config(self) -> dict:
        try:
            config_path = Path("config/config.yaml")
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found at {config_path.absolute()}")
                
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
                
            self.api_provider = config.get('api_provider', 'ollama')
            self.model = config.get('ollama', {}).get('model', 'llama3.2:1b') # Options: "algo_DD", "llama3.2", "qwen2.5:1.5b","0xroyce/plutus", "llama3.2:1b"
            return config
            
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise

    def setup_api_client(self) -> None:
        try:
            logger.info(f"Using {self.api_provider} for generation")
        except Exception as e:
            logger.error(f"Error setting up API client: {e}")
            raise

    def load_historical_data(self) -> pd.DataFrame:
        try:
            csv_path = os.path.join(self.base_path, "agents", "backtesting_agent", 
                                  "historical_data", f"{self.company_name}_minute.csv")
            df = pd.read_csv(csv_path)
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df.set_index('Datetime', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            raise

    def calculate_indicators(self, df: pd.DataFrame) -> dict:
        indicators = {}
        
        # Moving Averages
        indicators['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
        indicators['EMA_50'] = talib.EMA(df['Close'], timeperiod=50)
        indicators['WMA_20'] = talib.WMA(df['Close'], timeperiod=20)
        
        # VWAP (simplified daily calculation)
        df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (df['Typical_Price'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        indicators['VWAP'] = df['VWAP']
        
        # Momentum Indicators
        indicators['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        indicators['MACD'], indicators['MACD_Signal'], indicators['MACD_Hist'] = talib.MACD(
            df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        indicators['STOCH_K'], indicators['STOCH_D'] = talib.STOCH(
            df['High'], df['Low'], df['Close'])
        indicators['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'])
        
        # Volatility Indicators
        indicators['BB_Upper'], indicators['BB_Middle'], indicators['BB_Lower'] = talib.BBANDS(
            df['Close'], timeperiod=20)
        indicators['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # Keltner Channels
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        indicators['KC_Middle'] = talib.EMA(typical_price, timeperiod=20)
        atr = indicators['ATR']
        indicators['KC_Upper'] = indicators['KC_Middle'] + (2 * atr)
        indicators['KC_Lower'] = indicators['KC_Middle'] - (2 * atr)
        
        # Volume Indicators
        indicators['OBV'] = talib.OBV(df['Close'], df['Volume'])
        indicators['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)
        
        # Trend Indicators
        indicators['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        indicators['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=20)
        
        # Ichimoku Cloud
        high_9 = df['High'].rolling(window=9).max()
        low_9 = df['Low'].rolling(window=9).min()
        indicators['ICHIMOKU_TENKAN'] = (high_9 + low_9) / 2
        
        high_26 = df['High'].rolling(window=26).max()
        low_26 = df['Low'].rolling(window=26).min()
        indicators['ICHIMOKU_KIJUN'] = (high_26 + low_26) / 2
        
        indicators['ICHIMOKU_SENKOU_A'] = ((indicators['ICHIMOKU_TENKAN'] + 
                                           indicators['ICHIMOKU_KIJUN']) / 2).shift(26)
        
        high_52 = df['High'].rolling(window=52).max()
        low_52 = df['Low'].rolling(window=52).min()
        indicators['ICHIMOKU_SENKOU_B'] = ((high_52 + low_52) / 2).shift(26)
        
        return indicators

    def analyze_market(self, df: pd.DataFrame, indicators: dict) -> MarketAnalysis:
        # Trend Analysis
        trend = self.determine_trend(df, indicators)
        strength = self.calculate_trend_strength(df, indicators)
        
        # Volatility Analysis
        volatility = self.calculate_volatility(df, indicators)
        
        # Support/Resistance Levels
        support_levels = self.find_support_levels(df)
        resistance_levels = self.find_resistance_levels(df)
        
        # Volume Profile
        volume_profile = self.analyze_volume(df, indicators)
        
        # Risk Metrics
        risk_metrics = self.calculate_risk_metrics(df)
        
        return MarketAnalysis(
            trend=trend,
            strength=strength,
            volatility=volatility,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            volume_profile=volume_profile,
            risk_metrics=risk_metrics,
            indicators=indicators
        )

    def determine_trend(self, df: pd.DataFrame, indicators: dict) -> str:
        # Combine multiple trend indicators for robust trend determination
        sma_trend = df['Close'].iloc[-1] > indicators['SMA_20'].iloc[-1]
        macd_trend = indicators['MACD'].iloc[-1] > indicators['MACD_Signal'].iloc[-1]
        adx_strong = indicators['ADX'].iloc[-1] > 25
        
        trend_signals = [sma_trend, macd_trend]
        bullish_count = sum(trend_signals)
        
        if bullish_count >= 2 and adx_strong:
            return "strongly_bullish"
        elif bullish_count >= 2:
            return "bullish"
        elif bullish_count <= 1 and adx_strong:
            return "strongly_bearish"
        elif bullish_count <= 1:
            return "bearish"
        return "neutral"

    def calculate_trend_strength(self, df: pd.DataFrame, indicators: dict) -> float:
        adx = indicators['ADX'].iloc[-1]
        trend_momentum = abs(df['Close'].pct_change(20).iloc[-1])
        volume_trend = df['Volume'].iloc[-1] / df['Volume'].rolling(20).mean().iloc[-1]
        
        # Combine factors with weights
        strength = (
            0.4 * min(adx / 100, 1) +
            0.4 * min(trend_momentum * 10, 1) +
            0.2 * min(volume_trend / 2, 1)
        )
        
        return min(max(strength, 0), 1)

    def calculate_volatility(self, df: pd.DataFrame, indicators: dict) -> float:
        atr_volatility = indicators['ATR'].iloc[-1] / df['Close'].iloc[-1]
        bb_volatility = (indicators['BB_Upper'].iloc[-1] - indicators['BB_Lower'].iloc[-1]) / indicators['BB_Middle'].iloc[-1]
        return (atr_volatility + bb_volatility) / 2

    def find_support_levels(self, df: pd.DataFrame) -> list:
        local_mins = argrelextrema(df['Low'].values, np.less, order=20)[0]
        levels = sorted(set(df['Low'].iloc[local_mins]))
        current_price = df['Close'].iloc[-1]
        return [level for level in levels if level < current_price]

    def find_resistance_levels(self, df: pd.DataFrame) -> list:
        local_maxs = argrelextrema(df['High'].values, np.greater, order=20)[0]
        levels = sorted(set(df['High'].iloc[local_maxs]))
        current_price = df['Close'].iloc[-1]
        return [level for level in levels if level > current_price]

    def analyze_volume(self, df: pd.DataFrame, indicators: dict) -> dict:
        return {
            'average_volume': float(df['Volume'].mean()),
            'volume_trend': float(indicators['OBV'].diff().iloc[-1]),
            'mfi_value': float(indicators['MFI'].iloc[-1]),
            'volume_price_correlation': float(df['Volume'].corr(df['Close']))
        }

    def calculate_risk_metrics(self, df: pd.DataFrame) -> dict:
        returns = df['Close'].pct_change().dropna()
        return {
            'volatility': float(returns.std() * np.sqrt(252)),
            'var_95': float(np.percentile(returns, 5)),
            'max_drawdown': float(self.calculate_max_drawdown(df)),
            'sharpe_ratio': float(self.calculate_sharpe_ratio(returns))
        }

    def calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        rolling_max = df['Close'].expanding().max()
        drawdowns = df['Close'] / rolling_max - 1.0
        return abs(drawdowns.min())

    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        risk_free_rate = 0.05  # 5% annual risk-free rate
        excess_returns = returns - risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


    def calculate_daily_averages(self, df: pd.DataFrame, days: int = 30) -> dict:
        """Calculate daily averages for technical indicators using available historical data."""
        daily_averages = {}
        
        try:
            # Convert index to datetime if not already
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
                
            # Ensure numeric columns are float64
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                df[col] = df[col].astype(np.float64)
                
            # Resample to daily frequency and aggregate
            daily_df = df.resample('D').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()  # Remove days with no data
            
            # Ensure resampled data is also float64
            for col in numeric_columns:
                daily_df[col] = daily_df[col].astype(np.float64)
            
            # Get last n available trading days
            daily_df = daily_df.tail(days)
            
            # Convert arrays to numpy arrays of float64
            high_arr = daily_df['High'].values.astype(np.float64)
            low_arr = daily_df['Low'].values.astype(np.float64)
            close_arr = daily_df['Close'].values.astype(np.float64)
            volume_arr = daily_df['Volume'].values.astype(np.float64)
            
            # Calculate indicators for each available trading day
            for i, (date, day_data) in enumerate(daily_df.iterrows(), 1):
                date_str = date.strftime('%Y-%m-%d')
                day_key = f'Day_{i}'
                day_indicators={}
                try:
                    # Calculate indicators for this day's data
                    day_slice = slice(max(0, i-20), i)  # Get relevant data window
                    
                    # Moving Averages
                    if len(close_arr) >= 20:
                        sma = talib.SMA(close_arr, timeperiod=20)
                        day_indicators['SMA_20'] = float(sma[-1]) if not np.isnan(sma[-1]) else None
                    else:
                        day_indicators['SMA_20'] = None
                        
                    if len(close_arr) >= 50:
                        ema = talib.EMA(close_arr, timeperiod=50)
                        day_indicators['EMA_50'] = float(ema[-1]) if not np.isnan(ema[-1]) else None
                    else:
                        day_indicators['EMA_50'] = None
                        
                    # RSI
                    if len(close_arr) >= 14:
                        rsi = talib.RSI(close_arr, timeperiod=14)
                        day_indicators['RSI'] = float(rsi[-1]) if not np.isnan(rsi[-1]) else None
                    else:
                        day_indicators['RSI'] = None
                        
                    # MACD
                    if len(close_arr) >= 26:
                        macd, signal, _ = talib.MACD(close_arr, fastperiod=12, slowperiod=26, signalperiod=9)
                        day_indicators['MACD'] = float(macd[-1]) if not np.isnan(macd[-1]) else None
                        day_indicators['MACD_Signal'] = float(signal[-1]) if not np.isnan(signal[-1]) else None
                    else:
                        day_indicators['MACD'] = None
                        day_indicators['MACD_Signal'] = None
                        
                    # ADX
                    if len(close_arr) >= 14:
                        adx = talib.ADX(high_arr, low_arr, close_arr, timeperiod=14)
                        day_indicators['ADX'] = float(adx[-1]) if not np.isnan(adx[-1]) else None
                    else:
                        day_indicators['ADX'] = None
                        
                    # CCI
                    if len(close_arr) >= 14:
                        cci = talib.CCI(high_arr, low_arr, close_arr, timeperiod=14)
                        day_indicators['CCI'] = float(cci[-1]) if not np.isnan(cci[-1]) else None
                    else:
                        day_indicators['CCI'] = None
                        
                    # MFI
                    if len(close_arr) >= 14:
                        mfi = talib.MFI(high_arr, low_arr, close_arr, volume_arr, timeperiod=14)
                        day_indicators['MFI'] = float(mfi[-1]) if not np.isnan(mfi[-1]) else None
                    else:
                        day_indicators['MFI'] = None
                        
                    # ATR
                    if len(close_arr) >= 14:
                        atr = talib.ATR(high_arr, low_arr, close_arr, timeperiod=14)
                        day_indicators['ATR'] = float(atr[-1]) if not np.isnan(atr[-1]) else None
                    else:
                        day_indicators['ATR'] = None
                        
                    # Stochastic
                    if len(close_arr) >= 14:
                        k, d = talib.STOCH(high_arr, low_arr, close_arr)
                        day_indicators['STOCH_K'] = float(k[-1]) if not np.isnan(k[-1]) else None
                        day_indicators['STOCH_D'] = float(d[-1]) if not np.isnan(d[-1]) else None
                    else:
                        day_indicators['STOCH_K'] = None
                        day_indicators['STOCH_D'] = None
                        
                    # Bollinger Bands Width
                    if len(close_arr) >= 20:
                        upper, middle, lower = talib.BBANDS(close_arr)
                        if not np.isnan(upper[-1]) and not np.isnan(middle[-1]) and not np.isnan(lower[-1]):
                            bb_width = (upper[-1] - lower[-1]) / middle[-1]
                            day_indicators['BB_Width'] = float(bb_width)
                        else:
                            day_indicators['BB_Width'] = None
                    else:
                        day_indicators['BB_Width'] = None
                        
                    # Add metadata
                    day_indicators['date'] = date_str
                    day_indicators['is_trading_day'] = True
                    day_indicators['day_of_week'] = date.strftime('%A')
                    day_indicators['close_price'] = float(day_data['Close'])
                    day_indicators['volume'] = float(day_data['Volume'])
                    
                    daily_averages[day_key] = day_indicators
                    
                except Exception as e:
                    logger.error(f"Error calculating indicators for {date_str}: {e}")
                    continue
                
            return daily_averages
            
        except Exception as e:
            logger.error(f"Error calculating daily averages: {e}")
            logger.error(traceback.format_exc())
            raise


    def format_daily_indicators(self, daily_avgs: dict) -> list:
        """Format daily indicators for the prompt."""
        daily_indicators = []
        headers = ['Date', 'Close', 'SMA_20', 'EMA_50', 'RSI', 'MACD', 'ADX', 'CCI', 'MFI', 'ATR', 'STOCH_K', 'STOCH_D', 'BB_Width']
        
        daily_indicators.append("Historical Data (Available Trading Days):")
        daily_indicators.append("-" * 150)
        daily_indicators.append(" | ".join(headers))
        daily_indicators.append("-" * 150)
        
        for day_key in sorted(daily_avgs.keys()):
            day_data = daily_avgs[day_key]
            row = [
                day_data.get('date', ''),
                f"{day_data.get('close_price', 'nan'):>8.2f}",
                f"{day_data.get('SMA_20', 'nan'):>8.2f}" if day_data.get('SMA_20') is not None else 'nan',
                f"{day_data.get('EMA_50', 'nan'):>8.2f}" if day_data.get('EMA_50') is not None else 'nan',
                f"{day_data.get('RSI', 'nan'):>6.2f}" if day_data.get('RSI') is not None else 'nan',
                f"{day_data.get('MACD', 'nan'):>7.3f}" if day_data.get('MACD') is not None else 'nan',
                f"{day_data.get('ADX', 'nan'):>6.2f}" if day_data.get('ADX') is not None else 'nan',
                f"{day_data.get('CCI', 'nan'):>7.2f}" if day_data.get('CCI') is not None else 'nan',
                f"{day_data.get('MFI', 'nan'):>6.2f}" if day_data.get('MFI') is not None else 'nan',
                f"{day_data.get('ATR', 'nan'):>7.3f}" if day_data.get('ATR') is not None else 'nan',
                f"{day_data.get('STOCH_K', 'nan'):>7.2f}" if day_data.get('STOCH_K') is not None else 'nan',
                f"{day_data.get('STOCH_D', 'nan'):>7.2f}" if day_data.get('STOCH_D') is not None else 'nan',
                f"{day_data.get('BB_Width', 'nan'):>8.3f}" if day_data.get('BB_Width') is not None else 'nan'
            ]
            daily_indicators.append(" | ".join(row))
        
        return daily_indicators

    def generate_strategy_prompt(self, analysis: MarketAnalysis) -> str:
        """Generate enhanced strategy prompt with daily averages."""
        # Get daily averages
        df = self.load_historical_data()
        daily_avgs = self.calculate_daily_averages(df)
        
        # Format daily indicator values
        indicator_lines = []
        indicators = ['SMA_20', 'EMA_50', 'RSI', 'MACD', 'ADX', 'CCI', 'MFI', 'ATR', 'STOCH_K', 'STOCH_D', 'BB_Width']
        
        for indicator in indicators:
            line = f"- {indicator}: "
            for day_num in range(1, 4):
                day_key = f'Day_{day_num}'
                if day_key in daily_avgs and indicator in daily_avgs[day_key]:
                    value = daily_avgs[day_key][indicator]
                    line += f"{value:.2f}"
                    if day_num < 3:
                        line += " : "
            indicator_lines.append(line)
        
        daily_indicators = "\n".join(indicator_lines)
        
        return f"""Analyze this market data for {self.company_name} and create an optimized trading strategy:

    Market Analysis:
    - Trend: {analysis.trend}
    - Trend Strength: {analysis.strength:.2f}
    - Volatility: {analysis.volatility:.4f}
    - Risk Metrics:
    * Daily Volatility: {analysis.risk_metrics['volatility']:.4f}
    * VaR (95%): {analysis.risk_metrics['var_95']:.4f}
    * Max Drawdown: {analysis.risk_metrics['max_drawdown']:.4f}
    * Sharpe Ratio: {analysis.risk_metrics['sharpe_ratio']:.2f}

    Current Daily Average Indicator Values (Day1 : Day2 : Day3):
    {daily_indicators}

    Support/Resistance:
    - Support Levels: {analysis.support_levels}
    - Resistance Levels: {analysis.resistance_levels}

    Volume Analysis:
    - Average Volume: {analysis.volume_profile['average_volume']:.2f}
    - MFI: {analysis.volume_profile['mfi_value']:.2f}

    Create a comprehensive trading strategy using these technical indicators:
    1. Moving Averages (SMA, EMA, WMA, VWAP)
    2. Momentum Indicators (RSI, MACD, Stochastic, Williams %R)
    3. Volatility Indicators (Bollinger Bands, ATR, Keltner)
    4. Volume Indicators (OBV, MFI)
    5. Trend Indicators (ADX, CCI, Ichimoku)

    Required JSON format:
    {
        "initial_capital": 100000,
        "commission": 0.002,
        "entry_conditions": [
            {
                "indicator": "indicator_name",
                "condition": "above/below/crossover/crossunder",
                "value": threshold_value,
                "timeframe": "timeframe"
            }
        ],
        "exit_conditions": [
            {
                "indicator": "indicator_name",
                "condition": "above/below/crossover/crossunder",
                "value": threshold_value,
                "timeframe": "timeframe"
            }
        ],
        "trading_hours": {
            "start": "09:15",
            "end": "15:20"
        },
        "risk_management": {
            "max_position_size": value,
            "stop_loss": value,
            "take_profit": value
        }
    }

    Optimize the strategy based on the 3-day indicator trends and current market conditions."""


    def validate_ollama_response(self, response_content: str) -> tuple[bool, str, dict]:
        """Validate Ollama response and identify missing/incorrect elements."""
        try:
            # Clean and parse response
            cleaned_response = self.clean_json_response(response_content)
            strategy = json.loads(cleaned_response)
            
            # Required fields and their types
            required_fields = {
                'initial_capital': (int, float),
                'commission': (float,),
                'entry_conditions': (list,),
                'exit_conditions': (list,),
                'trading_hours': (dict,),
                'risk_management': (dict,)
            }
            
            # Required fields in conditions
            condition_fields = {'indicator', 'condition', 'value', 'timeframe'}
            
            # Check for missing or invalid fields
            missing_fields = []
            invalid_fields = []
            
            # Validate main fields
            for field, types in required_fields.items():
                if field not in strategy:
                    missing_fields.append(field)
                elif not isinstance(strategy[field], types):
                    invalid_fields.append(field)
            
            # Validate entry conditions
            if 'entry_conditions' in strategy:
                for condition in strategy['entry_conditions']:
                    missing = condition_fields - set(condition.keys())
                    if missing:
                        missing_fields.append(f"entry_conditions.{','.join(missing)}")
                        
            # Validate exit conditions
            if 'exit_conditions' in strategy:
                for condition in strategy['exit_conditions']:
                    missing = condition_fields - set(condition.keys())
                    if missing:
                        missing_fields.append(f"exit_conditions.{','.join(missing)}")
            
            # Generate error message if needed
            if missing_fields or invalid_fields:
                error_msg = ""
                if missing_fields:
                    error_msg += f"Missing fields: {', '.join(missing_fields)}. "
                if invalid_fields:
                    error_msg += f"Invalid fields: {', '.join(invalid_fields)}. "
                error_msg += "Please write the complete JSON with all required fields following the exact format."
                return False, error_msg, strategy
            
            return True, "", strategy
            
        except json.JSONDecodeError:
            return False, "Invalid JSON format. Please provide a valid JSON response following the exact format.", {}
        except Exception as e:
            return False, f"Error in validation: {str(e)}", {}

    def generate_strategy(self) -> dict:
        """Generate strategy with improved prompting and daily averages."""
        try:
            # Load and analyze data
            logger.info("Loading historical data...")
            df = self.load_historical_data()
            
            logger.info("Calculating technical indicators...")
            indicators = self.calculate_indicators(df)
            
            logger.info("Performing market analysis...")
            analysis = self.analyze_market(df, indicators)

            # System prompt with example
            system_prompt = """You are an expert algorithmic trader for the Indian stock market.
                            You MUST return ONLY a valid JSON object with no additional text, comments, or markdown formatting.
    IMPORTANT: 
    following is just an example of the JSON format you need to return. You need to replace the example values with your own analysis based on the data provided.
    Return ONLY a JSON object with EXACTLY this structure, no other text or markdown (this is very important and just to be clear):
    {
        "initial_capital": 100000,
        "commission": 0.002,
        "entry_conditions": [
            {
                "indicator": "SMA",
                "condition": "crossover",
                "value": 240.5,
                "timeframe": "15m"
            }
        ],
        "exit_conditions": [
            {
                "indicator": "RSI",
                "condition": "above",
                "value": 70,
                "timeframe": "5m"
            }
        ],
        "trading_hours": {
            "start": "09:15",
            "end": "15:20"
        },
        "risk_management": {
            "max_position_size": 0.1,
            "stop_loss": 0.02,
            "take_profit": 0.03
        }
    }

        Constraints:
        1. indicators: ["SMA", "EMA", "WMA", "VWAP", "RSI", "MACD", "Stochastic_K", "Stochastic_D", "WILLR", "BB_Upper", "BB_Lower", "BB_Middle", "ATR", "KC_Upper", "KC_Lower", "OBV", "MFI", "ADX", "CCI"]
        2. conditions: ["above", "below", "crossover", "crossunder"]
        3. timeframes: ["1m", "5m", "15m", "30m", "1h"]
        
        """

            # Calculate daily averages
            daily_avgs = self.calculate_daily_averages(df)
            
            # Format daily indicators
            daily_indicators = self.format_daily_indicators(daily_avgs)
            
            # Create base user prompt
            user_prompt = f"""Create a trading strategy for {self.company_name} based on this analysis:

    Current Price Action:
    - Support Levels: {[round(s, 2) for s in analysis.support_levels[:3]]}
    - Resistance Levels: {[round(r, 2) for r in analysis.resistance_levels[:3]]}
    - Trend: {analysis.trend}
    - Trend Strength: {analysis.strength:.2f}
    - Volatility: {analysis.volatility:.4f}

    {chr(10).join(daily_indicators)}

    Risk Metrics:
    - Daily Volatility: {analysis.risk_metrics['volatility']:.4f}
    - VaR (95%): {analysis.risk_metrics['var_95']:.4f}
    - Max Drawdown: {analysis.risk_metrics['max_drawdown']:.4f}
    - Sharpe Ratio: {analysis.risk_metrics['sharpe_ratio']:.2f}

    Volume Analysis:
    - Average Volume: {analysis.volume_profile['average_volume']:.2f}
    - MFI: {analysis.volume_profile['mfi_value']:.2f}

    Requirements:
    1. Use at least 2 different types of conditions for entry_conditions and exit_conditions
    2. Each entry_conditions and exit_conditions should have "indicator", "condition", "value", and "timeframe" fields (dono't use any other field name).
    3. Set indicator values based on current market readings
    4. Use support/resistance levels when relevant
    5. Choose timeframes between 5m and 1h
    6. Include specific numeric values
    7. Optimize for current market conditions
    8. Strategy must be suitable for intraday trading (09:15 to 15:20)
    9. Use maximum 10% position size, 2% stop loss, and 3% take profit
    10. Use 0.2% commission rate
    11. Use 100,000 INR initial capital
    12. Focus on profitable and less risky trades
    13. Write clear entry condition with minimum 2 indicators and their conditions.
    14. Write clear exit condition with minimum 2 indicators and their conditions.
    15. Only write entry_conditions and exit_conditions, no other text or markdown (this is very important and just to be clear).
    16. Set stop-loss based on volatility ({analysis.volatility:.4f})
    17. Set take-profit based on volatility ({analysis.volatility:.4f}) and risk tolerance (3%)
    18. Use only these indicators: ["SMA", "EMA", "WMA", "VWAP", "RSI", "MACD", "Stochastic_K", "Stochastic_D", "WILLR", "BB_Upper", "BB_Lower", "BB_Middle", "ATR", "KC_Upper", "KC_Lower", "OBV", "MFI", "ADX", "CCI"]
    19. Use only these conditions: ["above", "below", "crossover", "crossunder"]
    20. Use only these timeframes: ["1m", "5m", "15m", "30m", "1h"]

    Important:
    - only write json format, no other text or markdown (this is very important and just to be clear).
    -Use ONLY these indicators:(you can add period for indicator)
        - "SMA", "EMA", "WMA", "VWAP", "RSI", "MACD", "Stochastic_K", "Stochastic_D", "WILLR", "BB_Upper", "BB_Lower", "BB_Middle", "ATR", "KC_Upper", "KC_Lower", "OBV", "MFI", "ADX", "CCI"

    -Use ONLY these conditions: 
        - "above", "below", "crossover", "crossunder"

    -Use ONLY these timeframes: 
        - "1m", "5m", "15m", "30m", "1h"
    """


            max_retries = 3
            current_retry = 0
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]

            while current_retry < max_retries:
                logger.info(f"Attempt {current_retry + 1} of {max_retries}")
                
                response = ollama.chat(
                    model=self.model,
                    messages=messages,
                    options={
                        'temperature': 0.7, # Lower for more focused responses, higher for more creativity
                        'top_p': 0.9, # Higher for more diverse responses, lower for more focused responses
                        'num_predict': 1000 # Maximum number of tokens to generate
                    }
                )

                cleaned_response = self.clean_json_response(response['message']['content'])
                
                try:
                    strategy = json.loads(cleaned_response)
                    
                    # Check for empty or missing conditions
                    if not strategy.get('entry_conditions') or not strategy.get('exit_conditions'):
                        error_msg = """
                        Your response is missing entry or exit conditions. Please provide a complete JSON with:
                        - At least 2 entry conditions
                        - At least 2 exit conditions
                        Each condition must have indicator, condition, value, and timeframe fields."""
                        
                        messages.append({'role': 'assistant', 'content': cleaned_response})
                        messages.append({'role': 'user', 'content': error_msg})
                        current_retry += 1
                        continue

                    # Validate conditions format
                    invalid_conditions = False
                    required_fields = {'indicator', 'condition', 'value', 'timeframe'}
                    
                    for condition in strategy.get('entry_conditions', []):
                        if not all(field in condition for field in required_fields):
                            invalid_conditions = True
                            break
                            
                    for condition in strategy.get('exit_conditions', []):
                        if not all(field in condition for field in required_fields):
                            invalid_conditions = True
                            break

                    if invalid_conditions:
                        error_msg = """Your conditions are missing required fields. Each condition must have:
                        {
                            "indicator": "one of the allowed indicators",
                            "condition": "one of [above, below, crossover, crossunder]",
                            "value": numeric_value,
                            "timeframe": "one of [1m, 5m, 15m, 30m, 1h]"
                        }"""
                        
                        messages.append({'role': 'assistant', 'content': cleaned_response})
                        messages.append({'role': 'user', 'content': error_msg})
                        current_retry += 1
                        continue

                    # If we get here, we have a valid strategy
                    logger.info("Valid strategy generated")
                    break

                except json.JSONDecodeError as e:
                    error_msg = f"""Your response is not valid JSON. Please provide a complete JSON object with this exact structure:
                    {{
                        "initial_capital": 100000,
                        "commission": 0.002,
                        "entry_conditions": [
                            {{
                                "indicator": "RSI",
                                "condition": "crossover",
                                "value": 30,
                                "timeframe": "5m"
                            }}
                        ],
                        "exit_conditions": [
                            {{
                                "indicator": "RSI",
                                "condition": "above",
                                "value": 70,
                                "timeframe": "5m"
                            }}
                        ],
                        "trading_hours": {{
                            "start": "09:15",
                            "end": "15:20"
                        }},
                        "risk_management": {{
                            "max_position_size": 0.1,
                            "stop_loss": 0.02,
                            "take_profit": 0.03
                        }}
                    }}"""
                    
                    messages.append({'role': 'assistant', 'content': response['message']['content']})
                    messages.append({'role': 'user', 'content': error_msg})
                    current_retry += 1
                    continue

            # If we didn't get a valid strategy after retries, use default
            if current_retry >= max_retries:
                logger.error("Failed to generate valid strategy after max retries")
                strategy = self.get_default_strategy()

            # Validate and enhance strategy
            strategy = self.validate_strategy(strategy)
            
            # Add metadata
            strategy.update({
                "generated_at": datetime.now().isoformat(),
                "company": self.company_name,
                "api_provider": self.api_provider,
                "model": self.model,
                "market_conditions": {
                    "trend": analysis.trend,
                    "strength": analysis.strength,
                    "volatility": analysis.volatility
                }
            })
            
            # Save conversation history and files
            conversation_history = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in messages])
            self.save_strategy(
                strategy,
                conversation_history,
                response['message']['content']
            )
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating strategy: {e}")
            default_strategy = self.get_default_strategy()
            self.save_strategy(
                default_strategy,
                "Error in strategy generation - using default",
                f"Error: {str(e)}"
            )
            return default_strategy


    def validate_indicators_and_conditions(self, strategy: dict) -> dict:
        """Validate and fix indicator names and conditions."""
        valid_indicators = {
            "SMA", "EMA", "WMA", "VWAP",
            "RSI", "MACD", "Stochastic_K", "Stochastic_D", "WILLR",
            "BB_Upper", "BB_Lower", "BB_Middle", "ATR", "KC_Upper", "KC_Lower",
            "OBV", "MFI",
            "ADX", "CCI"
        }
        
        valid_conditions = {"above", "below", "crossover", "crossunder"}
        valid_timeframes = {"1m", "5m", "15m", "30m", "1h"}
        
        def fix_condition(condition):
            if isinstance(condition, dict):
                # Fix indicator name
                if condition.get('indicator') not in valid_indicators:
                    condition['indicator'] = "SMA"  # Default to SMA if invalid
                    
                # Fix condition type
                if condition.get('condition') not in valid_conditions:
                    condition['condition'] = "crossover"  # Default to crossover if invalid
                    
                # Fix timeframe
                if condition.get('timeframe') not in valid_timeframes:
                    condition['timeframe'] = "5m"  # Default to 5m if invalid
                    
                # Ensure value is numeric
                try:
                    condition['value'] = float(condition['value'])
                except (ValueError, TypeError):
                    condition['value'] = 0.0
                    
            return condition
        
        # Fix entry conditions
        if 'entry_conditions' in strategy:
            strategy['entry_conditions'] = [fix_condition(c) for c in strategy['entry_conditions']]
            
        # Fix exit conditions
        if 'exit_conditions' in strategy:
            strategy['exit_conditions'] = [fix_condition(c) for c in strategy['exit_conditions']]
            
        return strategy

    def enhance_strategy(self, strategy: dict, analysis: MarketAnalysis) -> dict:
        """Enhance the strategy with additional conditions based on analysis."""
        # Start with the basic validated strategy
        enhanced = self.validate_strategy(strategy)
        
        # Add trend-following conditions if none exist
        if not enhanced['entry_conditions']:
            if analysis.trend in ['strongly_bullish', 'bullish']:
                enhanced['entry_conditions'].append({
                    "indicator": "RSI",
                    "condition": "crossover",
                    "value": 30,
                    "timeframe": "5m"
                })
                enhanced['entry_conditions'].append({
                    "indicator": "MACD",
                    "condition": "crossover",
                    "value": 0,
                    "timeframe": "5m"
                })
            elif analysis.trend in ['strongly_bearish', 'bearish']:
                enhanced['entry_conditions'].append({
                    "indicator": "RSI",
                    "condition": "crossunder",
                    "value": 70,
                    "timeframe": "5m"
                })
                enhanced['entry_conditions'].append({
                    "indicator": "MACD",
                    "condition": "crossunder",
                    "value": 0,
                    "timeframe": "5m"
                })
        
        # Adjust risk management based on volatility
        enhanced['risk_management'] = {
            "max_position_size": round(max(0.1, min(1.0, 1.0 / (analysis.volatility * 10))), 2),
            "stop_loss": round(max(0.01, min(0.1, analysis.volatility * 2)), 3),
            "take_profit": round(max(0.02, min(0.2, analysis.volatility * 4)), 3)
        }
        
        return enhanced

    def clean_json_response(self, response: str) -> str:
            """Clean and validate the Ollama response with improved JSON handling."""
            try:
                logger.debug(f"Original response: {response}")
                
                # Remove markdown and whitespace
                response = response.replace('```json', '').replace('```', '').strip()
                
                # Find the JSON content
                start = response.find('{')
                end = response.rfind('}') + 1
                
                if start >= 0 and end > start:
                    json_str = response[start:end]
                    logger.debug(f"Extracted JSON string: {json_str}")
                    
                    # Additional cleaning steps
                    # Remove comments
                    json_str = re.sub(r'//.*?\n', '\n', json_str)
                    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
                    
                    # Fix property names
                    json_str = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', json_str)
                    
                    # Fix trailing commas
                    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                    
                    # Fix string values
                    json_str = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_]*)(,|\})', r': "\1"\2', json_str)
                    
                    try:
                        parsed_json = json.loads(json_str)
                        logger.info("Successfully parsed JSON response")
                        return json.dumps(parsed_json)
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON parsing failed: {e}")
                        # Try to recover malformed JSON
                        recovered_json = self.recover_malformed_json(json_str)
                        if recovered_json:
                            return json.dumps(recovered_json)
                        else:
                            logger.error("Could not recover JSON structure")
                            return json.dumps(self.get_default_strategy())
                
                logger.error("No valid JSON structure found in response")
                return json.dumps(self.get_default_strategy())
                
            except Exception as e:
                logger.error(f"Error cleaning JSON response: {e}")
                return json.dumps(self.get_default_strategy())

    def recover_malformed_json(self, json_str: str) -> dict:
        """Attempt to recover malformed JSON response."""
        try:
            # Create a template for the expected structure
            template = {
                "initial_capital": 100000,
                "commission": 0.002,
                "entry_conditions": [],
                "exit_conditions": [],
                "trading_hours": {
                    "start": "09:15",
                    "end": "15:20"
                },
                "risk_management": {
                    "max_position_size": 0.1,
                    "stop_loss": 0.02,
                    "take_profit": 0.03
                }
            }
            
            # Try to extract values using regex patterns
            patterns = {
                'initial_capital': r'"initial_capital":\s*(\d+)',
                'commission': r'"commission":\s*([\d.]+)',
                'entry_conditions': r'"entry_conditions":\s*(\[.*?\])',
                'exit_conditions': r'"exit_conditions":\s*(\[.*?\])',
                'trading_hours': r'"trading_hours":\s*({.*?})',
                'risk_management': r'"risk_management":\s*({.*?})'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, json_str, re.DOTALL)
                if match:
                    value = match.group(1)
                    try:
                        if key in ['entry_conditions', 'exit_conditions', 'trading_hours', 'risk_management']:
                            template[key] = json.loads(value)
                        else:
                            template[key] = float(value)
                    except:
                        continue
            
            return template
            
        except Exception as e:
            logger.error(f"Error recovering JSON: {e}")
            return None


    def validate_strategy(self, strategy: dict) -> dict:
            """Validate and normalize strategy structure with defaults."""
            try:
                # Set up default values
                default_strategy = {
                    "initial_capital": 100000,
                    "commission": 0.002,
                    "entry_conditions": [],
                    "exit_conditions": [],
                    "trading_hours": {
                        "start": "09:15",
                        "end": "15:20"
                    },
                    "risk_management": {
                        "max_position_size": 0.1,
                        "stop_loss": 0.02,
                        "take_profit": 0.03
                    }
                }

                # Start with default values
                validated = default_strategy.copy()

                # Update with provided values if they exist
                if strategy is None:
                    return validated

                # Validate and update initial capital
                if 'initial_capital' in strategy:
                    validated['initial_capital'] = float(strategy.get('initial_capital', 100000))

                # Validate and update commission
                if 'commission' in strategy:
                    validated['commission'] = float(strategy.get('commission', 0.002))

                # Validate and update entry conditions
                if 'entry_conditions' in strategy and isinstance(strategy['entry_conditions'], list):
                    validated['entry_conditions'] = self.validate_conditions(strategy['entry_conditions'])

                # Validate and update exit conditions
                if 'exit_conditions' in strategy and isinstance(strategy['exit_conditions'], list):
                    validated['exit_conditions'] = self.validate_conditions(strategy['exit_conditions'])

                # Validate and update trading hours
                if 'trading_hours' in strategy and isinstance(strategy['trading_hours'], dict):
                    validated['trading_hours'] = self.validate_trading_hours(strategy['trading_hours'])
                
                # Validate and update risk management
                if 'risk_management' in strategy and isinstance(strategy['risk_management'], dict):
                    validated['risk_management'] = self.validate_risk_management(strategy['risk_management'])

                return validated

            except Exception as e:
                logger.error(f"Error in strategy validation: {e}")
                return default_strategy.copy()

    def validate_conditions(self, conditions: list) -> list:
        """Validate trading conditions."""
        valid_conditions = []
        valid_indicators = {
            'RSI': (0, 100),
            'MACD': None,
            'MACD_Signal': None,
            'SMA': None,
            'EMA': None,
            'WMA': None,
            'VWAP': None,
            'BB_Upper': None,
            'BB_Lower': None,
            'BB_Middle': None,
            'ATR': (0, None),
            'CCI': (-200, 200),
            'ADX': (0, 100),
            'MFI': (0, 100),
            'STOCH_K': (0, 100),
            'STOCH_D': (0, 100),
            'WILLR': (-100, 0),
            'OBV': None
        }
        
        valid_conditions_list = ['above', 'below', 'crossover', 'crossunder', 'between']
        
        for condition in conditions:
            if isinstance(condition, dict):
                indicator = condition.get('indicator', '').upper()
                if indicator in valid_indicators:
                    value = condition.get('value', 0)
                    cond_type = condition.get('condition', '').lower()
                    
                    # Validate value ranges
                    value_range = valid_indicators[indicator]
                    if value_range:
                        min_val, max_val = value_range
                        if min_val is not None:
                            value = max(min_val, float(value))
                        if max_val is not None:
                            value = min(max_val, float(value))
                    
                    # Validate condition type
                    if cond_type in valid_conditions_list:
                        valid_conditions.append({
                            'indicator': indicator,
                            'condition': cond_type,
                            'value': value,
                            'timeframe': condition.get('timeframe', '1m')
                        })
        
        return valid_conditions

    def validate_trading_hours(self, hours: dict) -> dict:
        """Validate trading hours with error handling."""
        try:
            default_hours = {"start": "09:15", "end": "15:20"}
            
            if not isinstance(hours, dict):
                return default_hours
                
            start_time = hours.get('start', '09:15')
            end_time = hours.get('end', '15:20')
            
            # Validate time format
            try:
                datetime.strptime(start_time, '%H:%M')
                datetime.strptime(end_time, '%H:%M')
            except ValueError:
                return default_hours
            
            # Ensure times are within market hours
            market_open = datetime.strptime('09:15', '%H:%M')
            market_close = datetime.strptime('15:20', '%H:%M')
            start = datetime.strptime(start_time, '%H:%M')
            end = datetime.strptime(end_time, '%H:%M')
            
            if start < market_open or start > market_close:
                start_time = '09:15'
            if end > market_close or end < market_open:
                end_time = '15:20'
            if start >= end:
                return default_hours
                
            return {
                "start": start_time,
                "end": end_time
            }
            
        except Exception as e:
            logger.error(f"Error validating trading hours: {e}")
            return {"start": "09:15", "end": "15:20"}

    def validate_risk_management(self, risk: dict) -> dict:
        """Validate risk management parameters."""
        if not isinstance(risk, dict):
            risk = {}
            
        return {
            "max_position_size": min(max(float(risk.get('max_position_size', 0.1)), 0.01), 1.0),
            "stop_loss": min(max(float(risk.get('stop_loss', 0.02)), 0.01), 0.1),
            "take_profit": min(max(float(risk.get('take_profit', 0.03)), 0.01), 0.2)
        }

    def get_default_strategy(self) -> dict:
        """Return a default strategy configuration."""
        return {
            "initial_capital": 100000,
            "commission": 0.002,
            "entry_conditions": [
                {
                    "indicator": "RSI",
                    "condition": "below",
                    "value": 30,
                    "timeframe": "5m"
                },
                {
                    "indicator": "MACD",
                    "condition": "crossover",
                    "value": 20,
                    "timeframe": "5m"
                }
            ],
            "exit_conditions": [
                {
                    "indicator": "RSI",
                    "condition": "above",
                    "value": 70,
                    "timeframe": "5m"
                },
                {
                    "indicator": "MACD",
                    "condition": "crossunder",
                    "value": 40,
                    "timeframe": "5m"
                }
            ],
            "trading_hours": {
                "start": "09:15",
                "end": "15:20"
            },
            "risk_management": {
                "max_position_size": 0.1,
                "stop_loss": 0.02,
                "take_profit": 0.03
            }
        }

    def get_next_algorithm_number(self) -> int:
            """Find the next available algorithm number."""
            try:
                output_path = os.path.join('output', 'algo')
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                    return 1
                    
                pattern = f"{self.company_name}_algorithm-*.json"
                existing_files = glob.glob(os.path.join(output_path, pattern))
                
                if not existing_files:
                    return 1
                    
                numbers = []
                for file in existing_files:
                    match = re.search(rf"{self.company_name}_algorithm-(\d+)\.json", file)
                    if match:
                        numbers.append(int(match.group(1)))
                
                return max(numbers) + 1 if numbers else 1
                
            except Exception as e:
                logger.error(f"Error finding next algorithm number: {e}")
                return 1

    def save_strategy(self, strategy: dict, ollama_input: str, ollama_output: str) -> None:
            """Save all strategy related files including Ollama input/output."""
            try:
                # Create output directory
                output_dir = os.path.join('output', 'algo')
                os.makedirs(output_dir, exist_ok=True)
                
                # Get next algorithm number
                algo_num = self.get_next_algorithm_number()
                base_filename = f"{self.company_name}_algorithm-{algo_num}"
                
                # Save Ollama input
                input_filepath = os.path.join(output_dir, f"{base_filename}_input.txt")
                with open(input_filepath, 'w', encoding='utf-8') as f:
                    f.write(ollama_input)
                logger.info(f"Ollama input saved to {input_filepath}")
                
                # Save raw Ollama output
                output_filepath = os.path.join(output_dir, f"{base_filename}_ollama_output.txt")
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    f.write(ollama_output)
                logger.info(f"Ollama raw output saved to {output_filepath}")
                
                # Save processed JSON strategy
                json_filepath = os.path.join(output_dir, f"{base_filename}.json")
                with open(json_filepath, 'w', encoding='utf-8') as f:
                    json.dump(strategy, f, indent=4)
                logger.info(f"Strategy JSON saved to {json_filepath}")
                
                # Generate and save Pine Script
                pine_script = self.generate_pine_script(strategy)
                pine_filepath = os.path.join(output_dir, f"{base_filename}.pine")
                with open(pine_filepath, 'w', encoding='utf-8') as f:
                    f.write(pine_script)
                logger.info(f"Pine Script saved to {pine_filepath}")
                
            except Exception as e:
                logger.error(f"Error saving strategy files: {e}")
                raise


    def generate_pine_script(self, strategy: dict) -> str:
        """Convert strategy to Pine Script format with error handling."""
        try:
            # Ensure strategy is validated before processing
            strategy = self.validate_strategy(strategy)

            pine_script = f"""
//@version=5
strategy("{self.company_name} Strategy", 
    overlay=true, 
    initial_capital={strategy.get('initial_capital', 100000)}, 
    commission_type=strategy.commission.percent,
    commission_value={strategy.get('commission', 0.002)})

// Trading Hours
var start_time = timestamp("{strategy['trading_hours']['start']}")
var end_time = timestamp("{strategy['trading_hours']['end']}")
is_trading_time = time >= start_time and time <= end_time

// Indicators
"""
            # Track which indicators are needed
            needed_indicators = set()
            
            # Check entry conditions
            for condition in strategy.get('entry_conditions', []):
                indicator = condition.get('indicator', '').upper()
                needed_indicators.add(indicator)
                
            # Check exit conditions
            for condition in strategy.get('exit_conditions', []):
                indicator = condition.get('indicator', '').upper()
                needed_indicators.add(indicator)
                
            # Add required indicators
            for indicator in needed_indicators:
                pine_script += self.get_indicator_pine_code(indicator)
            
            # Add entry conditions
            pine_script += "\n// Entry Conditions"
            for condition in strategy.get('entry_conditions', []):
                pine_script += self.get_condition_pine_code(condition, 'entry')
            
            # Add exit conditions
            pine_script += "\n// Exit Conditions"
            for condition in strategy.get('exit_conditions', []):
                pine_script += self.get_condition_pine_code(condition, 'exit')
            
            # Add risk management
            risk = strategy.get('risk_management', {})
            pine_script += f"""

// Risk Management
var stop_loss = {risk.get('stop_loss', 0.02)}
var take_profit = {risk.get('take_profit', 0.03)}
var max_pos_size = {risk.get('max_position_size', 0.1)}
"""
            
            return pine_script
            
        except Exception as e:
            logger.error(f"Error generating Pine Script: {e}")
            # Return a basic valid Pine Script if there's an error
            return f"""
//@version=5
strategy("{self.company_name} Basic Strategy", 
    overlay=true, 
    initial_capital=100000, 
    commission_type=strategy.commission.percent,
    commission_value=0.002)

// Basic Trading Hours
var start_time = timestamp("09:15")
var end_time = timestamp("15:20")
is_trading_time = time >= start_time and time <= end_time

// Basic Strategy
if is_trading_time
    strategy.close_all()
"""

    def get_indicator_pine_code(self, indicator: str) -> str:
        """Get Pine Script code for a specific indicator."""
        indicator_code = {
            'RSI': 'rsi = ta.rsi(close, 14)\n',
            'MACD': '[macd_line, signal_line, hist] = ta.macd(close, 12, 26, 9)\n',
            'SMA': 'sma_20 = ta.sma(close, 20)\n',
            'EMA': 'ema_50 = ta.ema(close, 50)\n',
            'BB': '[bb_upper, bb_middle, bb_lower] = ta.bb(close, 20, 2)\n',
            'ADX': 'adx = ta.adx(high, low, close, 14)\n',
            'CCI': 'cci = ta.cci(high, low, close, 20)\n',
            'MFI': 'mfi = ta.mfi(high, low, close, volume, 14)\n',
            'STOCH': '[stoch_k, stoch_d] = ta.stoch(high, low, close, 14, 3, 3)\n'
        }
        return indicator_code.get(indicator, '')

    def get_condition_pine_code(self, condition: dict, condition_type: str) -> str:
        """Get Pine Script code for a specific condition."""
        try:
            indicator = condition.get('indicator', '').upper()
            cond = condition.get('condition', '').lower()
            value = condition.get('value', 0)
            
            if condition_type == 'entry':
                action = 'strategy.entry("Long", strategy.long)'
            else:
                action = 'strategy.close("Long")'
            
            condition_templates = {
                'above': f'if {indicator.lower()} > {value} and is_trading_time\n    {action}\n',
                'below': f'if {indicator.lower()} < {value} and is_trading_time\n    {action}\n',
                'crossover': f'if ta.crossover({indicator.lower()}, {value}) and is_trading_time\n    {action}\n',
                'crossunder': f'if ta.crossunder({indicator.lower()}, {value}) and is_trading_time\n    {action}\n'
            }
            
            return condition_templates.get(cond, '')
            
        except Exception as e:
            logger.error(f"Error generating condition Pine code: {e}")
            return ''

    def run(self) -> None:
        """Main execution method."""
        try:
            logger.info(f"Starting algorithm generation for {self.company_name}")
            
            # Generate strategy
            strategy = self.generate_strategy()
            
            logger.info("Algorithm generation completed successfully")
            
        except Exception as e:
            logger.error(f"Error in run: {e}")
            raise

if __name__ == "__main__":
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('algo_agent.log')
            ]
        )
        
        # Company configuration
        COMPANY_NAME = "ZOMATO"
        
        # Initialize and run agent
        agent = AlgoAgent(COMPANY_NAME)
        agent.run()
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        traceback.print_exc()
        sys.exit(1)