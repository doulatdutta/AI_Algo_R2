import glob
import json
import os
from datetime import datetime
import re
import traceback
import ollama
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import talib
import sys
from dataclasses import dataclass, asdict
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any
import numpy as np
from json import JSONEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        if isinstance(obj, (pd.Index, pd.Series)):
            return obj.tolist()
        if isinstance(obj, bool):
            return bool(obj)
        if pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)

@dataclass
class MarketCondition:
    trend: str
    strength: float
    volatility: float
    support_levels: List[float]
    resistance_levels: List[float]
    
@dataclass
class TradingStrategy:
    entry_rules: Dict[str, Any]
    exit_rules: Dict[str, Any]
    position_sizing: Dict[str, float]
    risk_management: Dict[str, float]

class AlgoAgent:
    def __init__(self, company_name: str):
        """Initialize AlgoAgent with company name."""
        self.company_name = company_name
        self.base_path = Path(os.getcwd())
        self.config = self.load_config()
        self.setup_api_client()
        self.scaler = StandardScaler()
        
    def detect_support_resistance(self, df: pd.DataFrame, window: int = 20) -> tuple:
        """Detect support and resistance levels using local minima/maxima."""
        try:
            # Find local minima and maxima
            local_min = argrelextrema(df['Low'].values, np.less, order=window)[0]
            local_max = argrelextrema(df['High'].values, np.greater, order=window)[0]

            # Get unique support and resistance levels
            support_levels = sorted(df['Low'].iloc[local_min].unique().tolist())
            resistance_levels = sorted(df['High'].iloc[local_max].unique().tolist())

            return support_levels, resistance_levels

        except Exception as e:
            logger.error(f"Error detecting support/resistance levels: {e}")
            return [], []

    def get_default_strategy(self) -> Dict:
        """Return a default strategy if JSON generation fails."""
        return {
            "initial_capital": 100000,
            "commission": 0.002,
            "entry_conditions": [],
            "exit_conditions": [],
            "trading_hours": {  # Default trading hours for Indian market
                "start": "09:15",
                "end": "15:20"
            },
            "risk_management": {
                "max_position_size": 0.1,
                "stop_loss": 0.02,
                "take_profit": 0.03
            }
        }


    def json_to_pine(self, strategy_json: Dict) -> str:
        """Convert strategy JSON to Pine Script."""
        try:
            # Ensure strategy_json has the correct structure
            if isinstance(strategy_json.get('trading_hours', None), list):
                trading_hours = {
                    'start': '09:15',
                    'end': '15:20'
                }
            else:
                trading_hours = strategy_json.get('trading_hours', {
                    'start': '09:15',
                    'end': '15:20'
                })
            
            # Start with strategy declaration
            pine_script = f"""
    //@version=5
    strategy("{self.company_name} Strategy", 
        overlay=true, 
        initial_capital={strategy_json.get('initial_capital', 100000)}, 
        commission_type=strategy.commission.percent,
        commission_value={strategy_json.get('commission', 0.002)})

    // Trading Hours
    var start_time = timestamp("{trading_hours.get('start', '0915')}")
    var end_time = timestamp("{trading_hours.get('end', '1520')}")
    is_trading_time = time >= start_time and time <= end_time

    // Indicators
    rsi = ta.rsi(close, 14)
    [bb_upper, bb_middle, bb_lower] = ta.bb(close, 20, 2)
    sma20 = ta.sma(close, 20)
    ema50 = ta.ema(close, 50)

    // Entry Conditions"""

            # Add entry conditions with safe access
            entry_conditions = strategy_json.get('entry_conditions', [])
            if isinstance(entry_conditions, list):
                for condition in entry_conditions:
                    pine_script += self.convert_condition_to_pine(condition)

            # Add exit conditions with safe access
            pine_script += "\n// Exit Conditions"
            exit_conditions = strategy_json.get('exit_conditions', [])
            if isinstance(exit_conditions, list):
                for condition in exit_conditions:
                    pine_script += self.convert_condition_to_pine(condition)

            # Add risk management with safe access
            pine_script += "\n// Risk Management"
            risk = strategy_json.get('risk_management', {
                'stop_loss': 0.02,
                'take_profit': 0.03,
                'max_position_size': 0.1
            })
            
            if isinstance(risk, dict):
                pine_script += f"""
    var stop_loss = {risk.get('stop_loss', 0.02)}
    var take_profit = {risk.get('take_profit', 0.03)}
    var max_pos_size = {risk.get('max_position_size', 0.1)}"""

            return pine_script

        except Exception as e:
            logger.error(f"Error converting to Pine Script: {e}")
            raise

    def convert_condition_to_pine(self, condition: Dict) -> str:
        """Convert a single condition to Pine Script."""
        pine_code = "\n"
        
        if condition.get('type') == 'momentum':
            if condition['indicator'] == 'RSI':
                if condition['condition'] == 'crossover':
                    pine_code += f"""
    if ta.crossover(rsi, {condition['value']}) and is_trading_time
        strategy.entry("RSI Buy", strategy.long)"""
                elif condition['condition'] == 'crossunder':
                    pine_code += f"""
    if ta.crossunder(rsi, {condition['value']}) and is_trading_time
        strategy.entry("RSI Sell", strategy.short)"""
                    
        elif condition.get('type') == 'trend':
            if condition['indicator'] == 'MACD':
                if condition['condition'] == 'positive_crossover':
                    pine_code += """
    if ta.crossover(ta.macd(close, 12, 26, 9), ta.macd(close, 12, 26, 9)[1]) and is_trading_time
        strategy.entry("MACD Buy", strategy.long)"""
                elif condition['condition'] == 'negative_crossover':
                    pine_code += """
    if ta.crossunder(ta.macd(close, 12, 26, 9), ta.macd(close, 12, 26, 9)[1]) and is_trading_time
        strategy.entry("MACD Sell", strategy.short)"""
        
        return pine_code

    def save_algorithm(self, strategy_json: Dict, pine_script: str, output_filename: str) -> None:
        """Save the generated algorithm in both JSON and Pine Script formats."""
        try:
            output_path = os.path.join('output', 'algo')
            os.makedirs(output_path, exist_ok=True)
            
            # Add metadata to JSON
            strategy_json.update({
                "generated_at": datetime.now().isoformat(),
                "company": self.company_name,
                "api_provider": self.api_provider,
                "model": self.model
            })
            
            # Save JSON using NumpyEncoder
            json_path = os.path.join(output_path, f"{output_filename}.json")
            with open(json_path, 'w') as f:
                json.dump(strategy_json, f, indent=4, cls=NumpyEncoder)
                
            # Save Pine Script
            pine_path = os.path.join(output_path, f"{output_filename}.pine")
            with open(pine_path, 'w') as f:
                f.write(pine_script)
                
        except Exception as e:
            logger.error(f"Error saving algorithm: {e}")
            raise




    def load_config(self) -> dict:
        """Load configuration from config file."""
        try:
            config_path = Path("config/config.yaml")
            print(f"Loading config from: {config_path.absolute()}")
            
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found at {config_path.absolute()}")
                
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
                print("Config loaded successfully")
                
            # Get API provider
            self.api_provider = config.get('api_provider', 'ollama').lower()
            print(f"API Provider: {self.api_provider}")
            
            # Set up API configurations
            self.model = config.get('ollama', {}).get('model','qwen2.5:1.5b') # Options: "algo_DD", "llama3.2", "qwen2.5:1.5b","0xroyce/plutus"
            print(f"Using model: {self.model}")
            return config
            
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            traceback.print_exc()
            raise

    def setup_api_client(self) -> None:
        """Setup the appropriate API client based on configuration."""
        try:
            logger.info("Using Ollama for generation")
        except Exception as e:
            logger.error(f"Error setting up API client: {e}")
            raise

    def generate_chat_completion(self, system_prompt: str, user_prompt: str) -> str:
        """Generate completion using Ollama."""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ]
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            raise

    def perform_technical_analysis(self, df: pd.DataFrame) -> Dict:
        """Perform basic technical analysis on the data."""
        return {
            'trend': {
                'sma_20': float(df['SMA_20'].iloc[-1]),
                'ema_50': float(df['EMA_50'].iloc[-1]),
                'price_vs_sma': bool(df['Close'].iloc[-1] > df['SMA_20'].iloc[-1]),
                'price_vs_ema': bool(df['Close'].iloc[-1] > df['EMA_50'].iloc[-1])
            }
        }

    def analyze_historical_data(self) -> Dict:
        """Enhanced analysis of historical data with AI-driven pattern recognition."""
        try:
            # Load and preprocess data
            # csv_path = f"../backtesting_agent/historical_data/{self.company_name}_minute.csv"
            # Or use absolute path:
            csv_path = os.path.join(os.getcwd(), "agents", "backtesting_agent", "historical_data", f"{self.company_name}_minute.csv")
            df = self.load_and_preprocess_data(csv_path)
            
            # Calculate advanced technical indicators
            df = self.calculate_advanced_indicators(df)
            
            # Detect market conditions
            market_condition = self.detect_market_condition(df)
            
            # Identify patterns and trends
            patterns = self.identify_patterns(df)
            
            # Calculate risk metrics
            risk_metrics = self.calculate_risk_metrics(df)
            
            # Compile analysis results
            analysis_results = {
                'market_condition': asdict(market_condition),
                'patterns': patterns,
                'risk_metrics': risk_metrics,
                'technical_analysis': self.perform_technical_analysis(df)
            }
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in historical data analysis: {e}")
            raise


    def get_pattern_rules(self, patterns: Dict) -> List[Dict]:
        """Generate pattern-based trading rules."""
        rules = []
        
        if patterns.get('chart', {}).get('double_bottom', {}).get('detected'):
            rules.append({
                "type": "reversal",
                "pattern": "double_bottom",
                "action": "buy",
                "confidence": patterns['chart']['double_bottom']['confidence']
            })
            
        if patterns.get('chart', {}).get('head_shoulders', {}).get('detected'):
            rules.append({
                "type": "reversal",
                "pattern": "head_shoulders",
                "action": "sell",
                "confidence": patterns['chart']['head_shoulders']['confidence']
            })
            
        return rules

    def get_volume_rules(self, analysis: Dict) -> List[Dict]:
        """Generate volume-based trading rules."""
        return [
            {
                "type": "volume_confirmation",
                "condition": "volume_above_average",
                "lookback": 20,
                "threshold": 1.5
            }
        ]

    def get_timeframe_rules(self, trend: str, strength: float) -> List[Dict]:
        """Generate timeframe-specific trading rules."""
        return [
            {
                "timeframe": "1h",
                "type": "trend_alignment",
                "condition": trend,
                "confidence": strength
            },
            {
                "timeframe": "4h",
                "type": "trend_confirmation",
                "weight": 0.6
            }
        ]

    def get_next_algorithm_number(self) -> int:
        """Find the next available algorithm number."""
        try:
            output_path = os.path.join('output', 'algo')
            if not os.path.exists(output_path):
                os.makedirs(output_path)
                return 1
                
            pattern = f"{self.company_name}_algorithm-*.pine"
            existing_files = glob.glob(os.path.join(output_path, pattern))
            
            if not existing_files:
                return 1
                
            numbers = []
            for file in existing_files:
                match = re.search(rf"{self.company_name}_algorithm-(\d+)\.pine", file)
                if match:
                    numbers.append(int(match.group(1)))
            
            return max(numbers) + 1 if numbers else 1
            
        except Exception as e:
            logger.error(f"Error finding next algorithm number: {e}")
            return 1

    def clean_json_response(self, response: str) -> str:
        """Clean the AI response to ensure it's valid JSON."""
        try:
            # Remove any markdown code block indicators
            response = response.replace('```json', '').replace('```', '')
            
            # Find the first '{' and last '}' to extract just the JSON object
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start >= 0 and end > start:
                response = response[start:end]
            
            return response.strip()
        except Exception as e:
            logger.error(f"Error cleaning JSON response: {e}")
            return "{}"

    def calculate_alma(self, series: pd.Series, length: int = 10, offset: float = 0.85, sigma: float = 6) -> pd.Series:
        """Calculate Arnaud Legoux Moving Average."""
        try:
            window = np.arange(length)
            m = offset * (length - 1)
            s = length / sigma
            weights = np.exp(-((window - m) ** 2) / (2 * s * s))
            weights = weights / weights.sum()
            
            return pd.Series(
                np.convolve(series, weights, mode='valid'),
                index=series.index[length-1:]
            )
        except Exception as e:
            logger.error(f"Error calculating ALMA: {e}")
            raise


    def load_and_preprocess_data(self, csv_path: str) -> pd.DataFrame:
        """Load and preprocess historical data with advanced cleaning."""
        try:
            # Load data
            df = pd.read_csv(csv_path)
            
            # Convert to datetime and handle potential NaT values
            df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
            
            # Drop rows with NaT values in datetime
            df = df.dropna(subset=['Datetime'])
            
            # Set index after cleaning datetime
            df.set_index('Datetime', inplace=True)
            
            # Handle missing values in numeric columns
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                # Use ffill() and bfill() instead of fillna(method='ffill')
                df[col] = df[col].ffill().bfill()
            
            # Remove outliers using IQR method
            for col in numeric_columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
            
            # Sort index
            df = df.sort_index()
            
            # Remove any duplicate indices
            df = df[~df.index.duplicated(keep='first')]
            
            # Convert all numeric columns to float64 to avoid int32 issues
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].astype(np.float64)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise


    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators."""
        # Trend Indicators
        df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
        df['EMA_50'] = talib.EMA(df['Close'], timeperiod=50)
        df['ALMA_10'] = self.calculate_alma(df['Close'], length=10, offset=0.85, sigma=6)
        
        # Momentum Indicators
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(
            df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        
        # Volatility Indicators
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(
            df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        
        # Volume Indicators
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        df['ADI'] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Custom Indicators
        df['Price_Range'] = df['High'] - df['Low']
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
        
        return df

    def detect_market_condition(self, df: pd.DataFrame) -> MarketCondition:
        """Detect current market conditions using AI analysis."""
        # Determine trend
        trend = self.determine_trend(df)
        
        # Calculate trend strength
        strength = self.calculate_trend_strength(df)
        
        # Calculate volatility
        volatility = df['Volatility'].iloc[-1]
        
        # Detect support and resistance
        support_levels, resistance_levels = self.detect_support_resistance(df)
        
        return MarketCondition(
            trend=trend,
            strength=strength,
            volatility=volatility,
            support_levels=support_levels,
            resistance_levels=resistance_levels
        )

    def determine_trend(self, df: pd.DataFrame) -> str:
        """Determine market trend using multiple indicators."""
        # Price action analysis
        price_trend = 'bullish' if df['Close'].iloc[-1] > df['SMA_20'].iloc[-1] else 'bearish'
        
        # MACD analysis
        macd_trend = 'bullish' if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] else 'bearish'
        
        # RSI analysis
        rsi_trend = 'bullish' if df['RSI'].iloc[-1] > 50 else 'bearish'
        
        # Combine signals
        bullish_signals = sum([
            price_trend == 'bullish',
            macd_trend == 'bullish',
            rsi_trend == 'bullish'
        ])
        
        if bullish_signals >= 2:
            return 'bullish'
        elif bullish_signals <= 1:
            return 'bearish'
        return 'neutral'

    def calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength using multiple factors."""
        # ADX for trend strength
        adx = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14).iloc[-1]
        
        # Price momentum
        momentum = abs(df['Close'].pct_change(periods=20).iloc[-1])
        
        # Volume trend
        volume_trend = df['Volume'].iloc[-1] / df['Volume_MA'].iloc[-1]
        
        # Combine factors
        strength = (adx / 100 + momentum + (volume_trend - 1)) / 3
        return min(max(strength, 0), 1)  # Normalize between 0 and 1

    def identify_patterns(self, df: pd.DataFrame) -> Dict:
        """Identify chart patterns and candlestick patterns."""
        patterns = {}
        
        # Candlestick patterns
        patterns['candlestick'] = {
            'doji': talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close']).iloc[-1],
            'engulfing': talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close']).iloc[-1],
            'hammer': talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close']).iloc[-1]
        }
        
        # Chart patterns
        patterns['chart'] = {
            'double_bottom': self.detect_double_bottom(df),
            'head_shoulders': self.detect_head_shoulders(df)
        }
        
        return patterns

    def calculate_risk_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate risk metrics for strategy optimization."""
        return {
            'volatility': df['Volatility'].iloc[-1],
            'var_95': self.calculate_var(df, confidence=0.95),
            'sharpe_ratio': self.calculate_sharpe_ratio(df),
            'max_drawdown': self.calculate_max_drawdown(df)
        }

    def calculate_var(self, df: pd.DataFrame, confidence: float) -> float:
        """Calculate Value at Risk."""
        returns = df['Close'].pct_change().dropna()
        return abs(np.percentile(returns, (1 - confidence) * 100))

    def calculate_sharpe_ratio(self, df: pd.DataFrame) -> float:
        """Calculate Sharpe Ratio."""
        returns = df['Close'].pct_change().dropna()
        excess_returns = returns - 0.05/252  # Assuming 5% risk-free rate
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """Calculate Maximum Drawdown."""
        rolling_max = df['Close'].expanding().max()
        drawdowns = df['Close'] / rolling_max - 1.0
        return abs(drawdowns.min())


    def generate_strategy_json(self) -> Dict:
        """Generate enhanced trading strategy using AI analysis."""
        try:
            # Analyze historical data
            analysis_results = self.analyze_historical_data()
            
            # Generate AI prompts
            system_prompt = """You are an expert algorithmic trader for the Indian stock market. Analyze market data to determine optimal trading parameters.
            
            Return a JSON with this exact structure:
            {
                "indicators": [
                    {"type": "RSI/MACD/EMA", "name": "unique_name", "params": {"length": value}}
                ],
                "entry_conditions": [
                    {"indicator1": "unique_name", "indicator2": "value/signal", "condition": "crossover/crossunder", "action": "buy/sell", "size": 0.95}
                ],
                "exit_conditions": [
                    {"indicator1": "unique_name", "indicator2": "value/signal", "condition": "crossover/crossunder", "action": "exit"}
                ],
                "initial_capital": 100000,
                "commission": 0.002,
                "trading_hours": {
                    "start": "HH:MM",  // Choose optimal start time between 09:15-15:20
                    "end": "HH:MM"     // Choose optimal end time between start time-15:20
                },
                "risk_management": {
                    "max_position_size": value,  // Based on volatility (0.1-1.0)
                    "stop_loss": value,         // Based on ATR and volatility
                    "take_profit": value        // Based on support/resistance levels
                }
            }"""

            user_prompt = f"""Based on this analysis of {self.company_name}, create a strategy:
            
            Market Conditions:
            - Trend: {analysis_results['market_condition']['trend']}
            - Trend Strength: {analysis_results['market_condition']['strength']:.2f}
            - Volatility: {analysis_results['market_condition']['volatility']:.2f}
            - Support Levels: {analysis_results['market_condition']['support_levels']}
            - Resistance Levels: {analysis_results['market_condition']['resistance_levels']}
            - Risk Metrics:
            * Value at Risk (95%): {analysis_results['risk_metrics']['var_95']:.4f}
            * Max Drawdown: {analysis_results['risk_metrics']['max_drawdown']:.4f}
            * Sharpe Ratio: {analysis_results['risk_metrics']['sharpe_ratio']:.2f}

            Requirements:
            1. Analyze volatility patterns to determine optimal trading hours
            2. Set position size based on volatility ({analysis_results['market_condition']['volatility']:.4f}):
            - Higher volatility = smaller position size
            - Lower volatility = larger position size
            - Range: 0.1 to 1.0
            3. Calculate stop loss based on:
            - Volatility: {analysis_results['market_condition']['volatility']:.4f}
            - Value at Risk: {analysis_results['risk_metrics']['var_95']:.4f}
            - Support/Resistance levels
            4. Set take profit based on:
            - Recent price swings
            - Support/Resistance levels
            - Risk:Reward ratio (minimum 1:2)
            5. Optimize for maximum profitability and minimum risk
            6. Strategy should be suitable for intraday trading

            Return ONLY the JSON object with exact structure requested."""

            # Generate strategy using AI
            response = self.generate_chat_completion(system_prompt, user_prompt)
            strategy_json = json.loads(self.clean_json_response(response))
            
            # Clean redundant fields
            fields_to_remove = ['trades', 'trading_rules', 'entryConditions', 'exitConditions', 'symbols', 'timeframes']
            for field in fields_to_remove:
                strategy_json.pop(field, None)
            
            # Validate trading hours
            strategy_json['trading_hours'] = self.validate_trading_hours(strategy_json.get('trading_hours', {}))
            
            # Validate risk management parameters
            risk_management = strategy_json.get('risk_management', {})
            strategy_json['risk_management'] = {
                'max_position_size': min(max(float(risk_management.get('max_position_size', 0.5)), 0.1), 1.0),
                'stop_loss': min(max(float(risk_management.get('stop_loss', 0.02)), 0.01), 0.1),
                'take_profit': min(max(float(risk_management.get('take_profit', 0.03)), 0.02), 0.2)
            }

            # Add other required fields if missing (but keep risk management from Ollama)
            required_fields = {
                "indicators": [
                    {"type": "RSI", "name": "rsi", "params": {"length": 14}},
                    {"type": "MACD", "name": "macd", "params": {"fast": 12, "slow": 26, "signal": 9}}
                ],
                "entry_conditions": [
                    {"indicator1": "rsi", "indicator2": "30", "condition": "crossover", "action": "buy", "size": 0.95},
                    {"indicator1": "macd", "indicator2": "signal", "condition": "crossover", "action": "buy", "size": 0.95}
                ],
                "exit_conditions": [
                    {"indicator1": "rsi", "indicator2": "70", "condition": "crossover", "action": "exit"},
                    {"indicator1": "macd", "indicator2": "signal", "condition": "crossunder", "action": "exit"}
                ],
                "initial_capital": 100000,
                "commission": 0.002
            }

            # Ensure all required fields exist (except risk_management)
            for field, default_value in required_fields.items():
                if field not in strategy_json:
                    strategy_json[field] = default_value
                
            return strategy_json

        except Exception as e:
            logger.error(f"Error generating strategy JSON: {e}")
            return self.get_default_strategy()

    def validate_trading_hours(self, trading_hours: Dict) -> Dict:
        """Validate and correct trading hours."""
        try:
            start_time = datetime.strptime(trading_hours.get('start', '09:15'), '%H:%M')
            end_time = datetime.strptime(trading_hours.get('end', '15:20'), '%H:%M')
            market_open = datetime.strptime('09:15', '%H:%M')
            market_close = datetime.strptime('15:20', '%H:%M')
            
            # Ensure times are within market hours
            if start_time < market_open:
                start_time = market_open
            if end_time > market_close:
                end_time = market_close
            if start_time > end_time:
                start_time = market_open
                end_time = market_close
            
            return {
                "start": start_time.strftime('%H:%M'),
                "end": end_time.strftime('%H:%M')
            }
        except ValueError:
            return {"start": "09:15", "end": "15:20"}

    # def generate_strategy_json(self) -> Dict:
    #     """Generate enhanced trading strategy using AI analysis."""
    #     try:
    #         # Analyze historical data
    #         analysis_results = self.analyze_historical_data()
            
    #         # Generate AI prompts
    #         system_prompt = """You are an expert algorithmic trader. Generate a profitable trading strategy for the Indian stock market.
    #         Your response must be a valid JSON object with the following EXACT structure:
    #         {
    #             "indicators": [
    #                 {"type": "SMA/RSI/MACD/etc", "name": "indicator_name", "params": {"length": value, ...}},
    #                 ...
    #             ],
    #             "entry_conditions": [
    #                 {"indicator1": "indicator_name", "indicator2": "indicator_name", "condition": "crossover/crossunder", "action": "buy/sell", "size": 0.95},
    #                 ...
    #             ],
    #             "exit_conditions": [
    #                 {"indicator1": "indicator_name", "indicator2": "indicator_name", "condition": "crossover/crossunder", "action": "exit"},
    #                 ...
    #             ],
    #             "trading_hours": {"start": "09:15", "end": "15:20"},
    #             "initial_capital": 100000,
    #             "commission": 0.002,
    #             "risk_management": {
    #                 "max_position_size": 0.1,
    #                 "stop_loss": <calculated_value>,
    #                 "take_profit": <calculated_value>
    #             }
    #         }"""

    #         user_prompt = f"""Based on the following market analysis for {self.company_name}, create a specific trading strategy:

    # Market Analysis:
    # - Trend: {analysis_results['market_condition']['trend']}
    # - Strength: {analysis_results['market_condition']['strength']:.2f}
    # - Volatility: {analysis_results['market_condition']['volatility']:.2f}
    # - Support Levels: {analysis_results['market_condition']['support_levels']}
    # - Resistance Levels: {analysis_results['market_condition']['resistance_levels']}
    # - RSI: {analysis_results['technical_analysis']['trend'].get('rsi', 0)}
    # - MACD Status: {analysis_results['technical_analysis']['trend'].get('macd', 0)}

    # Required Indicators Examples:
    # 1. RSI with length 14 for momentum
    # 2. MACD(12,26,9) for trend
    # 3. SMA/EMA combinations for crossovers

    # Entry Conditions Examples:
    # 1. RSI crosses above 30 for oversold bounces
    # 2. MACD crosses above signal line for trend confirmation
    # 3. Price crosses above SMA for trend following

    # Exit Conditions Examples:
    # 1. RSI crosses above 70 for overbought exits
    # 2. MACD crosses below signal line for trend reversal
    # 3. Price crosses below EMA for trend exit

    # You must include at least 2 entry conditions and 2 exit conditions.
    # Risk management should be adjusted based on the volatility ({analysis_results['market_condition']['volatility']:.4f}).

    # Response must be valid JSON that matches the structure provided in the system prompt."""

    #         # Generate strategy using AI
    #         response = self.generate_chat_completion(system_prompt, user_prompt)
    #         strategy_json = json.loads(self.clean_json_response(response))
            
    #         # Ensure we have minimum required fields
    #         if not strategy_json.get('entry_conditions') or not strategy_json.get('exit_conditions'):
    #             logger.warning("Missing required conditions in generated strategy, using backup")
    #             backup_strategy = {
    #                 "indicators": [
    #                     {"type": "RSI", "name": "rsi", "params": {"length": 14}},
    #                     {"type": "MACD", "name": "macd", "params": {"fast": 12, "slow": 26, "signal": 9}}
    #                 ],
    #                 "entry_conditions": [
    #                     {"indicator1": "rsi", "indicator2": "30", "condition": "crossover", "action": "buy", "size": 0.95},
    #                     {"indicator1": "macd", "indicator2": "signal", "condition": "crossover", "action": "buy", "size": 0.95}
    #                 ],
    #                 "exit_conditions": [
    #                     {"indicator1": "rsi", "indicator2": "70", "condition": "crossover", "action": "exit"},
    #                     {"indicator1": "macd", "indicator2": "signal", "condition": "crossunder", "action": "exit"}
    #                 ],
    #                 "initial_capital": 100000,
    #                 "commission": 0.002,
    #                 "trading_hours": {
    #                     "start": "09:15",
    #                     "end": "15:20"
    #                 },
    #                 "risk_management": {
    #                     "max_position_size": 0.1,
    #                     "stop_loss": 0.02,
    #                     "take_profit": 0.03
    #                 }
    #             }
    #             strategy_json.update(backup_strategy)
    #             strategy_json.pop('entryConditions', None)
    #             strategy_json.pop('exitConditions', None)
            
    #         return strategy_json

    #     except Exception as e:
    #         logger.error(f"Error generating strategy JSON: {e}")
    #         return self.get_default_strategy()


    # def validate_and_enhance_strategy(self, strategy: Dict, analysis: Dict) -> Dict:
    #     """Validate and enhance the generated strategy based on analysis results."""
    #     try:
    #         # Ensure basic structure
    #         base_strategy = {
    #             "initial_capital": 100000,
    #             "commission": 0.002,
    #             "trading_hours": {
    #                 "start": "09:15",
    #                 "end": "15:20"
    #             },
    #             "entry_conditions": [],
    #             "exit_conditions": [],
    #             "risk_management": {
    #                 "max_position_size": 0.1,
    #                 "stop_loss": 0.02,
    #                 "take_profit": 0.03
    #             }
    #         }
            
    #         # Merge with provided strategy
    #         for key, value in strategy.items():
    #             if key == 'trading_hours' and not isinstance(value, dict):
    #                 continue  # Skip invalid trading_hours format
    #             base_strategy[key] = value

    #         # Add risk management rules based on volatility
    #         base_strategy['risk_management'] = {
    #             'max_position_size': float(min(0.1, 1.0 / analysis['risk_metrics']['volatility'])),
    #             'stop_loss': float(analysis['risk_metrics']['var_95'] * 2),
    #             'take_profit': float(analysis['risk_metrics']['var_95'] * 3)
    #         }
            
    #         # Adjust entry conditions based on market condition
    #         if analysis['market_condition']['trend'] == 'bullish':
    #             base_strategy['entry_conditions'].extend(self.get_bullish_conditions())
    #         elif analysis['market_condition']['trend'] == 'bearish':
    #             base_strategy['entry_conditions'].extend(self.get_bearish_conditions())
            
    #         return base_strategy

    #     except Exception as e:
    #         logger.error(f"Error in strategy validation: {e}")
    #         return self.get_default_strategy()

    # def get_bullish_conditions(self) -> List[Dict]:
    #     """Get additional conditions for bullish market."""
    #     return [
    #         {
    #             "type": "momentum",
    #             "indicator": "RSI",
    #             "condition": "crossover",
    #             "value": 40,
    #             "action": "buy"
    #         },
    #         {
    #             "type": "trend",
    #             "indicator": "MACD",
    #             "condition": "positive_crossover",
    #             "action": "buy"
    #         }
    #     ]

    # def get_bearish_conditions(self) -> List[Dict]:
    #     """Get additional conditions for bearish market."""
    #     return [
    #         {
    #             "type": "momentum",
    #             "indicator": "RSI",
    #             "condition": "crossunder",
    #             "value": 60,
    #             "action": "sell"
    #         },
    #         {
    #             "type": "trend",
    #             "indicator": "MACD",
    #             "condition": "negative_crossover",
    #             "action": "sell"
    #         }
    #     ]



    def generate_algorithms(self) -> None:
        """Generate enhanced trading algorithm with AI optimization."""
        logger.info(f"Starting advanced algorithm generation for {self.company_name}")
        
        try:
            # Analyze historical data
            logger.info("Performing comprehensive market analysis...")
            analysis_results = self.analyze_historical_data()
            
            # Get algorithm number
            algo_num = self.get_next_algorithm_number()
            output_filename = f"{self.company_name}_algorithm-{algo_num}"
            
            # Generate and optimize strategy
            logger.info("Generating and optimizing trading strategy...")
            strategy_json = self.generate_strategy_json()
            
            # Convert to Pine Script
            logger.info("Converting to Pine Script with advanced features...")
            pine_script = self.json_to_pine(strategy_json)
            
            # Save files
            logger.info("Saving algorithm files...")
            self.save_algorithm(strategy_json, pine_script, output_filename)
            
            # Generate strategy report
            self.generate_strategy_report(strategy_json, analysis_results, output_filename)
            
            logger.info(f"Algorithm generation completed successfully")
            
        except Exception as e:
            logger.error(f"Error in algorithm generation: {str(e)}")
            raise

    def generate_strategy_report(self, strategy: Dict, analysis: Dict, filename: str) -> None:
        """Generate a detailed strategy report with analysis insights."""
        try:
            # Convert numpy/pandas types to Python native types
            def convert_to_native_types(obj):
                if isinstance(obj, dict):
                    return {key: convert_to_native_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native_types(item) for item in obj]
                elif isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, pd.Timestamp):
                    return str(obj)
                elif pd.isna(obj):
                    return None
                return obj

            # Create report with converted types
            report = {
                "strategy_name": f"{self.company_name} Trading Strategy",
                "generation_date": datetime.now().isoformat(),
                "market_analysis": convert_to_native_types({
                    "trend": analysis['market_condition']['trend'],
                    "strength": analysis['market_condition']['strength'],
                    "volatility": analysis['market_condition']['volatility']
                }),
                "risk_metrics": convert_to_native_types(analysis['risk_metrics']),
                "strategy_rules": convert_to_native_types(strategy),
                "recommendations": convert_to_native_types(
                    self.generate_recommendations(analysis)
                ),
                "performance_metrics": convert_to_native_types({
                    "expected_sharpe": analysis['risk_metrics']['sharpe_ratio'],
                    "max_drawdown": analysis['risk_metrics']['max_drawdown'],
                    "var_95": analysis['risk_metrics']['var_95']
                }),
                "trading_parameters": convert_to_native_types({
                    "position_sizing": self.calculate_position_sizing(analysis),
                    "stop_loss_levels": self.calculate_stop_levels(analysis),
                    "take_profit_levels": self.calculate_profit_levels(analysis)
                })
            }
            
            # Add market condition specific recommendations
            report["market_specific_rules"] = convert_to_native_types(
                self.generate_market_specific_rules(analysis)
            )
            
            # Add pattern analysis
            report["pattern_analysis"] = convert_to_native_types({
                "detected_patterns": analysis.get('patterns', {}),
                "pattern_reliability": self.assess_pattern_reliability(analysis)
            })
            
            # Save report
            report_path = os.path.join('output', 'reports', f"{filename}_report.json")
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4, cls=NumpyEncoder)
                
            logger.info(f"Strategy report generated successfully: {report_path}")
                
        except Exception as e:
            logger.error(f"Error generating strategy report: {e}")
            raise

    def generate_recommendations(self, analysis: Dict) -> Dict:
        """Generate trading recommendations based on market analysis."""
        recommendations = {
            "entry_strategy": [],
            "exit_strategy": [],
            "risk_management": [],
            "timeframe_analysis": []
        }
        
        # Entry recommendations
        if analysis['market_condition']['trend'] == 'bullish':
            recommendations["entry_strategy"].append({
                "type": "entry",
                "action": "Long",
                "confidence": analysis['market_condition']['strength'],
                "conditions": [
                    "Wait for pullbacks to support levels",
                    "Confirm with volume increase",
                    "Check RSI for oversold conditions"
                ]
            })
        elif analysis['market_condition']['trend'] == 'bearish':
            recommendations["entry_strategy"].append({
                "type": "entry",
                "action": "Short",
                "confidence": analysis['market_condition']['strength'],
                "conditions": [
                    "Wait for rallies to resistance levels",
                    "Confirm with volume increase",
                    "Check RSI for overbought conditions"
                ]
            })
            
        # Exit recommendations based on volatility
        if analysis['market_condition']['volatility'] > 0.02:  # High volatility
            recommendations["exit_strategy"].extend([
                "Use wider stops due to high volatility",
                "Consider scaling out of positions",
                "Implement trailing stops"
            ])
        else:
            recommendations["exit_strategy"].extend([
                "Tighter stops can be used",
                "Consider full position exits",
                "Use fixed take-profit levels"
            ])
            
        return recommendations

    def calculate_position_sizing(self, analysis: Dict) -> Dict:
        """Calculate optimal position sizing based on risk analysis."""
        volatility = analysis['market_condition']['volatility']
        trend_strength = analysis['market_condition']['strength']
        
        # Base position size on volatility and trend strength
        base_size = min(0.1, 1.0 / volatility)  # Lower volatility allows larger position
        adjusted_size = base_size * trend_strength  # Strong trends allow larger positions
        
        return {
            "base_size": round(base_size, 3),
            "adjusted_size": round(adjusted_size, 3),
            "max_position": round(min(adjusted_size, 0.2), 3),  # Never exceed 20% of portfolio
            "scaling_rules": {
                "scale_in_levels": [0.3, 0.3, 0.4],
                "scale_out_levels": [0.4, 0.3, 0.3]
            }
        }

    def calculate_stop_levels(self, analysis: Dict) -> Dict:
        """Calculate optimal stop loss levels based on market conditions."""
        atr = analysis.get('technical_analysis', {}).get('atr', 0)
        volatility = analysis['market_condition']['volatility']
        
        return {
            "initial_stop": round(atr * 2, 2),
            "trailing_stop": round(atr * 3, 2),
            "volatility_adjusted_stop": round(atr * (1 + volatility * 10), 2),
            "time_based_stop": {
                "bars": 5,
                "condition": "No new high/low"
            }
        }

    def calculate_profit_levels(self, analysis: Dict) -> Dict:
        """Calculate take profit levels based on market analysis."""
        atr = analysis.get('technical_analysis', {}).get('atr', 0)
        trend_strength = analysis['market_condition']['strength']
        
        return {
            "target_1": round(atr * 3, 2),
            "target_2": round(atr * 5, 2),
            "target_3": round(atr * 8, 2),
            "trailing_profit": {
                "activation": round(atr * 4, 2),
                "trail_amount": round(atr * 2, 2)
            },
            "adjust_factor": round(trend_strength, 2)
        }

    def generate_market_specific_rules(self, analysis: Dict) -> Dict:
        """Generate specific trading rules based on current market conditions."""
        trend = analysis['market_condition']['trend']
        strength = analysis['market_condition']['strength']
        volatility = analysis['market_condition']['volatility']
        
        rules = {
            "trend_rules": self.get_trend_specific_rules(trend, strength),
            "volatility_rules": self.get_volatility_rules(volatility),
            "pattern_rules": self.get_pattern_rules(analysis.get('patterns', {})),
            "volume_rules": self.get_volume_rules(analysis),
            "timeframe_rules": self.get_timeframe_rules(trend, strength)
        }
        
        return rules

    def get_trend_specific_rules(self, trend: str, strength: float) -> List[Dict]:
        """Generate trend-specific trading rules."""
        rules = []
        
        if trend == 'bullish':
            rules.extend([
                {
                    "type": "entry",
                    "condition": "price_above_ema",
                    "timeframe": "1h",
                    "confidence": strength
                },
                {
                    "type": "confirmation",
                    "indicator": "MACD",
                    "condition": "positive_crossover"
                }
            ])
        elif trend == 'bearish':
            rules.extend([
                {
                    "type": "entry",
                    "condition": "price_below_ema",
                    "timeframe": "1h",
                    "confidence": strength
                },
                {
                    "type": "confirmation",
                    "indicator": "MACD",
                    "condition": "negative_crossover"
                }
            ])
            
        return rules

    def get_volatility_rules(self, volatility: float) -> List[Dict]:
        """Generate volatility-based trading rules."""
        if volatility > 0.02:  # High volatility
            return [
                {
                    "type": "position_size",
                    "action": "reduce",
                    "factor": 0.7
                },
                {
                    "type": "stop_loss",
                    "action": "widen",
                    "factor": 1.5
                }
            ]
        else:  # Low volatility
            return [
                {
                    "type": "position_size",
                    "action": "increase",
                    "factor": 1.2
                },
                {
                    "type": "stop_loss",
                    "action": "tighten",
                    "factor": 0.8
                }
            ]

    def assess_pattern_reliability(self, analysis: Dict) -> Dict:
        """Assess the reliability of detected patterns."""
        patterns = analysis.get('patterns', {})
        reliability_scores = {}
        
        for pattern_type, pattern_data in patterns.items():
            if isinstance(pattern_data, dict):
                reliability = self.calculate_pattern_reliability(
                    pattern_data,
                    analysis['market_condition']['trend'],
                    analysis['market_condition']['strength']
                )
                reliability_scores[pattern_type] = reliability
                
        return reliability_scores

    def calculate_pattern_reliability(self, pattern_data: Dict, trend: str, strength: float) -> float:
        """Calculate the reliability score for a pattern."""
        # Base score on pattern completion and clarity
        base_score = 0.7
        
        # Adjust based on trend alignment
        if trend == 'bullish' and pattern_data.get('type') == 'bullish':
            base_score += 0.2
        elif trend == 'bearish' and pattern_data.get('type') == 'bearish':
            base_score += 0.2
            
        # Adjust based on trend strength
        base_score *= (0.5 + strength)
        
        return round(min(base_score, 1.0), 2)

    def detect_double_bottom(self, df: pd.DataFrame, window: int = 20) -> Dict:
        """Detect double bottom pattern in price data."""
        try:
            # Find local minima
            local_min = argrelextrema(df['Low'].values, np.less, order=window)[0]
            
            if len(local_min) < 2:
                return {"detected": False}
                
            # Check last two minima for double bottom
            last_mins = df['Low'].iloc[local_min[-2:]]
            price_diff = abs(last_mins.iloc[0] - last_mins.iloc[1])
            avg_price = last_mins.mean()
            
            is_double_bottom = price_diff / avg_price < 0.02  # 2% tolerance
            
            return {
                "detected": is_double_bottom,
                "confidence": 0.8 if is_double_bottom else 0,
                "first_bottom": last_mins.iloc[0],
                "second_bottom": last_mins.iloc[1]
            }
            
        except Exception as e:
            logger.error(f"Error detecting double bottom: {e}")
            return {"detected": False}

    def detect_head_shoulders(self, df: pd.DataFrame, window: int = 20) -> Dict:
        """Detect head and shoulders pattern."""
        try:
            # Find local maxima
            local_max = argrelextrema(df['High'].values, np.greater, order=window)[0]
            
            if len(local_max) < 3:
                return {"detected": False}
                
            # Get last three peaks
            last_peaks = df['High'].iloc[local_max[-3:]]
            
            # Check if middle peak is highest
            is_head_shoulders = (last_peaks.iloc[1] > last_peaks.iloc[0] and 
                               last_peaks.iloc[1] > last_peaks.iloc[2] and
                               abs(last_peaks.iloc[0] - last_peaks.iloc[2]) / last_peaks.iloc[1] < 0.1)
                               
            return {
                "detected": is_head_shoulders,
                "confidence": 0.8 if is_head_shoulders else 0,
                "left_shoulder": last_peaks.iloc[0],
                "head": last_peaks.iloc[1],
                "right_shoulder": last_peaks.iloc[2]
            }
            
        except Exception as e:
            logger.error(f"Error detecting head and shoulders: {e}")
            return {"detected": False}


if __name__ == "__main__":
    print("Starting execution...")
    try:
        company_name = "ZOMATO"  # Define company name once
        
        # Set up basic console logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),  # Console handler
                logging.FileHandler('algo_agent.log')  # File handler
            ]
        )
        
        print("Checking file paths...")
        # Check if required files and directories exist
        config_path = Path("config/config.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path.absolute()}")
            
        data_path = Path(os.path.join(os.getcwd(), "agents", "backtesting_agent", "historical_data", f"{company_name}_minute.csv"))
        if not data_path.exists():
            raise FileNotFoundError(f"Historical data file not found at {data_path.absolute()}")
            
        print("Initializing AlgoAgent...")
        algo_agent = AlgoAgent(company_name)
        
        print("Generating algorithms...")
        algo_agent.generate_algorithms()
        
        print("Execution completed successfully!")
        
    except FileNotFoundError as e:
        print(f"File not found error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)