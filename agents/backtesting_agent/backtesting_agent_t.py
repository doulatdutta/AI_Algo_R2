#import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import talib
import logging
import json
import glob
import plotly.graph_objects as go
import plotly.express as px
import shutil
import os
import warnings
#form

from pathlib import Path
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple, Optional,Union
from backtesting import Backtest, Strategy
from plotly.subplots import make_subplots
from dataclasses import dataclass
from scipy import stats


warnings.filterwarnings('ignore')


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradeMetrics:
    """Container for trade metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    avg_trade_duration: float
    risk_reward_ratio: float

class TechnicalIndicators:
    """Technical indicator calculations with comprehensive error handling and validation."""
    
    @staticmethod
    def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, 
                      volume: pd.Series) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            volume: Series of volume data
            
        Returns:
            pd.Series: VWAP values
        """
        try:
            # Input validation
            if not all(isinstance(x, pd.Series) for x in [high, low, close, volume]):
                raise ValueError("All inputs must be pandas Series")
                
            if not all(len(x) == len(high) for x in [low, close, volume]):
                raise ValueError("All inputs must have the same length")
                
            # Calculate typical price
            typical_price = (high + low + close) / 3
            
            # Calculate VWAP
            vwap = (typical_price * volume).cumsum() / volume.cumsum()
            
            return vwap
            
        except Exception as e:
            logger.error(f"Error calculating VWAP: {str(e)}")
            return pd.Series(np.nan, index=high.index)

    @staticmethod
    def calculate_supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                           period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate SuperTrend indicator.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            period: ATR period
            multiplier: ATR multiplier
            
        Returns:
            Tuple[pd.Series, pd.Series]: (SuperTrend values, SuperTrend direction)
        """
        try:
            # Input validation
            if not all(isinstance(x, pd.Series) for x in [high, low, close]):
                raise ValueError("Price inputs must be pandas Series")
                
            if not all(len(x) == len(high) for x in [low, close]):
                raise ValueError("All price inputs must have the same length")
                
            # Calculate ATR
            atr = TechnicalIndicators.calculate_atr(high, low, close, period)
            
            # Calculate basic upper and lower bands
            basic_ub = (high + low) / 2 + multiplier * atr
            basic_lb = (high + low) / 2 - multiplier * atr
            
            # Initialize final upper and lower bands
            final_ub = pd.Series(0.0, index=close.index)
            final_lb = pd.Series(0.0, index=close.index)
            
            # Initialize SuperTrend and direction
            supertrend = pd.Series(0.0, index=close.index)
            direction = pd.Series(1, index=close.index)  # 1 for uptrend, -1 for downtrend
            
            # Calculate SuperTrend
            for i in range(period, len(close)):
                final_ub[i] = basic_ub[i] if (
                    basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]
                ) else final_ub[i-1]
                
                final_lb[i] = basic_lb[i] if (
                    basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]
                ) else final_lb[i-1]
                
                if close[i] > final_ub[i-1]:
                    supertrend[i] = final_lb[i]
                    direction[i] = 1
                elif close[i] < final_lb[i-1]:
                    supertrend[i] = final_ub[i]
                    direction[i] = -1
                else:
                    supertrend[i] = supertrend[i-1]
                    direction[i] = direction[i-1]
            
            return supertrend, direction
            
        except Exception as e:
            logger.error(f"Error calculating SuperTrend: {str(e)}")
            return pd.Series(np.nan, index=high.index), pd.Series(np.nan, index=high.index)

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            period: ATR period
            
        Returns:
            pd.Series: ATR values
        """
        try:
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR
            atr = tr.rolling(window=period).mean()
            
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series(np.nan, index=high.index)

    @staticmethod
    def calculate_psar(high: pd.Series, low: pd.Series, close: pd.Series,
                      af_start: float = 0.02, af_step: float = 0.02, 
                      af_max: float = 0.2) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Parabolic SAR indicator.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            af_start: Starting acceleration factor
            af_step: Acceleration factor step
            af_max: Maximum acceleration factor
            
        Returns:
            Tuple[pd.Series, pd.Series]: (PSAR values, PSAR direction)
        """
        try:
            psar = pd.Series(0.0, index=close.index)
            direction = pd.Series(1, index=close.index)  # 1 for uptrend, -1 for downtrend
            af = pd.Series(af_start, index=close.index)
            ep = pd.Series(0.0, index=close.index)
            
            # Initialize values
            direction[0] = 1 if close[0] > close[1] else -1
            psar[0] = high[0] if direction[0] < 0 else low[0]
            ep[0] = high[0] if direction[0] > 0 else low[0]
            
            # Calculate PSAR
            for i in range(2, len(close)):
                # Update PSAR
                psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
                
                # Update direction
                if direction[i-1] > 0:
                    if low[i] < psar[i]:
                        direction[i] = -1
                        psar[i] = ep[i-1]
                        ep[i] = low[i]
                        af[i] = af_start
                    else:
                        direction[i] = 1
                        if high[i] > ep[i-1]:
                            ep[i] = high[i]
                            af[i] = min(af[i-1] + af_step, af_max)
                        else:
                            ep[i] = ep[i-1]
                            af[i] = af[i-1]
                else:
                    if high[i] > psar[i]:
                        direction[i] = 1
                        psar[i] = ep[i-1]
                        ep[i] = high[i]
                        af[i] = af_start
                    else:
                        direction[i] = -1
                        if low[i] < ep[i-1]:
                            ep[i] = low[i]
                            af[i] = min(af[i-1] + af_step, af_max)
                        else:
                            ep[i] = ep[i-1]
                            af[i] = af[i-1]
                            
                # Ensure PSAR is below price in uptrend and above in downtrend
                if direction[i] > 0:
                    psar[i] = min(psar[i], low[i-1], low[i-2])
                else:
                    psar[i] = max(psar[i], high[i-1], high[i-2])
            
            return psar, direction
            
        except Exception as e:
            logger.error(f"Error calculating PSAR: {str(e)}")
            return pd.Series(np.nan, index=high.index), pd.Series(np.nan, index=high.index)

    @staticmethod
    def calculate_keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series,
                                 period: int = 20, atr_mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Keltner Channels.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            period: EMA period
            atr_mult: ATR multiplier
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: (Upper band, Middle line, Lower band)
        """
        try:
            # Calculate middle line (EMA)
            middle = close.ewm(span=period, adjust=False).mean()
            
            # Calculate ATR
            atr = TechnicalIndicators.calculate_atr(high, low, close, period)
            
            # Calculate upper and lower bands
            upper = middle + atr_mult * atr
            lower = middle - atr_mult * atr
            
            return upper, middle, lower
            
        except Exception as e:
            logger.error(f"Error calculating Keltner Channels: {str(e)}")
            return (pd.Series(np.nan, index=high.index), 
                   pd.Series(np.nan, index=high.index),
                   pd.Series(np.nan, index=high.index))

    @staticmethod
    def calculate_ssl_channels(high: pd.Series, low: pd.Series, close: pd.Series,
                             period: int = 10) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate SSL (Schaffman Support/resistance Lines) Channels.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            period: MA period
            
        Returns:
            Tuple[pd.Series, pd.Series]: (Upper SSL, Lower SSL)
        """
        try:
            # Calculate moving averages of highs and lows
            ma_high = high.rolling(window=period).mean()
            ma_low = low.rolling(window=period).mean()
            
            # Initialize SSL channels
            ssl_upper = pd.Series(0.0, index=close.index)
            ssl_lower = pd.Series(0.0, index=close.index)
            
            # Calculate SSL values
            for i in range(period, len(close)):
                if close[i-1] > ma_high[i-1]:
                    ssl_upper[i] = ma_high[i]
                    ssl_lower[i] = ma_low[i]
                else:
                    ssl_upper[i] = ma_low[i]
                    ssl_lower[i] = ma_high[i]
            
            return ssl_upper, ssl_lower
            
        except Exception as e:
            logger.error(f"Error calculating SSL Channels: {str(e)}")
            return pd.Series(np.nan, index=high.index), pd.Series(np.nan, index=high.index)

    @staticmethod
    def calculate_ttm_squeeze(high: pd.Series, low: pd.Series, close: pd.Series,
                            bb_period: int = 20, kc_period: int = 20, 
                            bb_mult: float = 2.0, kc_mult: float = 1.5) -> Tuple[pd.Series, bool]:
        """
        Calculate TTM Squeeze indicator.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            bb_period: Bollinger Bands period
            kc_period: Keltner Channel period
            bb_mult: Bollinger Bands multiplier
            kc_mult: Keltner Channel multiplier
            
        Returns:
            Tuple[pd.Series, bool]: (Momentum, Squeeze status)
        """
        try:
            # Calculate Bollinger Bands
            bb_middle = close.rolling(window=bb_period).mean()
            bb_std = close.rolling(window=bb_period).std()
            bb_upper = bb_middle + bb_mult * bb_std
            bb_lower = bb_middle - bb_mult * bb_std
            
            # Calculate Keltner Channels
            kc_middle = close.rolling(window=kc_period).mean()
            atr = TechnicalIndicators.calculate_atr(high, low, close, kc_period)
            kc_upper = kc_middle + kc_mult * atr
            kc_lower = kc_middle - kc_mult * atr
            
            # Calculate squeeze
            squeeze = (bb_upper <= kc_upper) & (bb_lower >= kc_lower)
            
            # Calculate momentum
            highest_high = high.rolling(window=bb_period).max()
            lowest_low = low.rolling(window=bb_period).min()
            avg = (highest_high + lowest_low) / 2
            momentum = close - avg
            
            return momentum, squeeze
            
        except Exception as e:
            logger.error(f"Error calculating TTM Squeeze: {str(e)}")
            return pd.Series(np.nan, index=high.index), pd.Series(False, index=high.index)


class JSONStrategy(Strategy):
    """Strategy implementation based on JSON configuration with enhanced features."""
    
    def init(self):
        """Initialize strategy with comprehensive setup and validation."""
        try:
            if not hasattr(self, 'json_config'):
                raise ValueError("Strategy configuration not provided")
            
            logger.info("Initializing strategy...")
            logger.debug(f"Strategy config: {self.json_config}")
            
            # Initialize containers
            self.indicators = {}
            self.trades_log = []
            self.daily_stats = {}
            self.current_position = 0
            self.entry_price = 0
            self.entry_time = None
            self.entry_idx = None
            self.stop_loss = None
            self.take_profit = None
            
            # Initialize risk management parameters
            self.risk_params = self.json_config.get('risk_management', {
                'max_position_size': 0.1,
                'stop_loss': 0.02,
                'take_profit': 0.03,
                'max_daily_loss': 0.05,
                'trailing_stop': False,
                'trailing_stop_atr_mult': 2.0
            })
            
            # Add base price data
            self.indicators['close'] = self.data.Close
            self.indicators['high'] = self.data.High
            self.indicators['low'] = self.data.Low
            self.indicators['volume'] = self.data.Volume
            
            # Initialize indicators
            for indicator in self.json_config.get('indicators', []):
                self.initialize_indicator(indicator)
                
            logger.info("Strategy initialization completed")
            
        except Exception as e:
            logger.error(f"Error in strategy initialization: {e}")
            logger.debug("Exception details:", exc_info=True)
            raise

    def initialize_indicator(self, indicator_config: Dict):
        """Initialize technical indicators with error handling."""
        try:
            indicator_type = indicator_config['type'].lower()
            params = indicator_config.get('params', {})
            name = indicator_config['name']
            
            # Indicator mapping
            indicator_functions = {
                'rsi': lambda: self.I(talib.RSI, self.data.Close, 
                                    timeperiod=params.get('length', 14)),
                'sma': lambda: self.I(talib.SMA, self.data.Close, 
                                    timeperiod=params.get('length', 20)),
                'ema': lambda: self.I(talib.EMA, self.data.Close, 
                                    timeperiod=params.get('length', 20)),
                'macd': lambda: self.initialize_macd(params),
                'bbands': lambda: self.initialize_bbands(params),
                'supertrend': lambda: self.initialize_supertrend(params),
                'atr': lambda: self.I(talib.ATR, self.data.High, self.data.Low, 
                                    self.data.Close, timeperiod=params.get('length', 14)),
                'stoch': lambda: self.I(talib.STOCH, self.data.High, self.data.Low, 
                                    self.data.Close, params.get('k_period', 14),
                                    params.get('d_period', 3)),
                'adx': lambda: self.I(talib.ADX, self.data.High, self.data.Low, 
                                    self.data.Close, timeperiod=params.get('length', 14))
            }
            
            # Calculate indicator
            if indicator_type in indicator_functions:
                result = indicator_functions[indicator_type]()
                
                # Handle tuple results (like MACD)
                if isinstance(result, tuple):
                    for i, component in enumerate(['', '_signal', '_hist']):
                        self.indicators[f"{name}{component}"] = result[i]
                else:
                    self.indicators[name] = result
                    
                logger.info(f"Initialized {name} indicator")
            else:
                logger.warning(f"Unsupported indicator type: {indicator_type}")
                
        except Exception as e:
            logger.error(f"Error initializing {indicator_config['type']}: {e}")
            raise

    def initialize_macd(self, params: Dict) -> Tuple:
        """Initialize MACD indicator with custom parameters."""
        try:
            fast = params.get('fast', 12)
            slow = params.get('slow', 26)
            signal = params.get('signal', 9)
            
            macd, signal, hist = self.I(talib.MACD, self.data.Close,
                                      fastperiod=fast,
                                      slowperiod=slow,
                                      signalperiod=signal)
            return macd, signal, hist
            
        except Exception as e:
            logger.error(f"Error initializing MACD: {e}")
            raise

    def initialize_bbands(self, params: Dict) -> Tuple:
        """Initialize Bollinger Bands with custom parameters."""
        try:
            period = params.get('length', 20)
            dev_up = params.get('dev_up', 2)
            dev_down = params.get('dev_down', 2)
            
            upper, middle, lower = self.I(talib.BBANDS, self.data.Close,
                                        timeperiod=period,
                                        nbdevup=dev_up,
                                        nbdevdn=dev_down)
            return upper, middle, lower
            
        except Exception as e:
            logger.error(f"Error initializing Bollinger Bands: {e}")
            raise

    def initialize_supertrend(self, params: Dict) -> Tuple:
        """Initialize SuperTrend indicator with custom parameters."""
        try:
            period = params.get('period', 10)
            multiplier = params.get('multiplier', 3.0)
            
            atr = self.I(talib.ATR, self.data.High, self.data.Low, 
                        self.data.Close, timeperiod=period)
            
            # Calculate basic upper and lower bands
            basic_ub = (self.data.High + self.data.Low) / 2 + multiplier * atr
            basic_lb = (self.data.High + self.data.Low) / 2 - multiplier * atr
            
            # Initialize final bands and direction
            final_ub = self.I(lambda: pd.Series(basic_ub))
            final_lb = self.I(lambda: pd.Series(basic_lb))
            supertrend = self.I(lambda: pd.Series(0.0, index=self.data.index))
            direction = self.I(lambda: pd.Series(1, index=self.data.index))
            
            return supertrend, direction
            
        except Exception as e:
            logger.error(f"Error initializing SuperTrend: {e}")
            raise

    def is_trading_hours(self) -> bool:
        """Check if current time is within configured trading hours."""
        try:
            current_time = self.data.index[-1].time()
            trading_hours = self.json_config.get('trading_hours', {
                'start': '09:15',
                'end': '15:20'
            })
            
            start_time = datetime.strptime(trading_hours['start'], '%H:%M').time()
            end_time = datetime.strptime(trading_hours['end'], '%H:%M').time()
            
            return start_time <= current_time <= end_time
            
        except Exception as e:
            logger.error(f"Error checking trading hours: {e}")
            return False

    def check_conditions(self, conditions: List[Dict]) -> bool:
        """Evaluate trading conditions with comprehensive logging."""
        try:
            for condition in conditions:
                ind1_name = condition['indicator1']
                ind2_name = condition['indicator2']
                condition_type = condition['condition']
                
                # Get indicator values
                if ind1_name not in self.indicators:
                    logger.warning(f"Indicator {ind1_name} not found")
                    continue
                    
                ind1_values = self.indicators[ind1_name]
                
                # Handle numeric values and indicator references
                if ind2_name.replace('.', '').isdigit():
                    ind2_values = np.full_like(ind1_values, float(ind2_name))
                elif ind2_name == 'signal' and f"{ind1_name}_signal" in self.indicators:
                    ind2_values = self.indicators[f"{ind1_name}_signal"]
                elif ind2_name in self.indicators:
                    ind2_values = self.indicators[ind2_name]
                else:
                    logger.warning(f"Invalid indicator2 reference: {ind2_name}")
                    continue
                
                # Get recent values
                if len(ind1_values) < 2 or len(ind2_values) < 2:
                    continue
                    
                curr1, prev1 = ind1_values[-1], ind1_values[-2]
                curr2, prev2 = ind2_values[-1], ind2_values[-2]
                
                # Check conditions
                if condition_type == 'crossover':
                    if prev1 <= prev2 and curr1 > curr2:
                        logger.info(f"Crossover detected: {ind1_name}({curr1:.2f}) crossed above {ind2_name}({curr2:.2f})")
                        return True
                elif condition_type == 'crossunder':
                    if prev1 >= prev2 and curr1 < curr2:
                        logger.info(f"Crossunder detected: {ind1_name}({curr1:.2f}) crossed below {ind2_name}({curr2:.2f})")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking conditions: {e}")
            return False

    def next(self):
        """Execute trading logic with risk management."""
        try:
            if not self.is_trading_hours():
                return
                
            current_price = self.data.Close[-1]
            
            # Check stop loss and take profit if in position
            if self.position and self.position.is_long:
                if current_price <= self.stop_loss or current_price >= self.take_profit:
                    self.position.close()
                    self.log_trade('exit', self.position.size)
                    logger.info(f"SL/TP triggered at {current_price:.2f}")
                    return
            
            # Check entry conditions if not in position
            if not self.position:
                for condition in self.json_config['entry_conditions']:
                    if self.check_conditions([condition]):
                        # Calculate position size
                        size = self.calculate_position_size(condition, current_price)
                        
                        if condition['action'] == 'buy':
                            self.buy(size=size)
                            self.entry_price = current_price
                            self.entry_time = self.data.index[-1]
                            self.entry_idx = len(self.data) - 1
                            
                            # Set stop loss and take profit
                            self.stop_loss = current_price * (1 - self.risk_params['stop_loss'])
                            self.take_profit = current_price * (1 + self.risk_params['take_profit'])
                            
                            self.log_trade('entry', size)
                            break
            
            # Check exit conditions if in position
            elif self.position:
                for condition in self.json_config['exit_conditions']:
                    if self.check_conditions([condition]):
                        self.position.close()
                        self.log_trade('exit', self.position.size)
                        break
                        
        except Exception as e:
            logger.error(f"Error in next(): {e}")
            logger.debug("Exception details:", exc_info=True)

    def calculate_position_size(self, condition: Dict, current_price: float) -> float:
        """Calculate position size based on risk parameters."""
        try:
            # Get base size from condition or default
            base_size = condition.get('size', 0.95)
            
            # Apply risk management rules
            max_position = self.risk_params['max_position_size']
            position_size = min(base_size, max_position)
            
            # Calculate size in units based on available equity
            units = (self.equity * position_size) / current_price
            
            return units
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    def log_trade(self, action: str, size: float):
        """Enhanced trade logging with performance metrics."""
        try:
            trade = {
                'time': self.data.index[-1],
                'action': action,
                'price': self.data.Close[-1],
                'size': size,
                'equity': self.equity,
                'pnl': self.position.pnl if self.position else 0,
                'entry_idx': self.entry_idx if action == 'exit' else len(self.data) - 1,
                'exit_idx': len(self.data) - 1 if action == 'exit' else None,
                'indicators': {
                    name: values[-1] for name, values in self.indicators.items()
                }
            }
            
            self.trades_log.append(trade)
            
            # Update daily statistics
            date = self.data.index[-1].date()
            if date not in self.daily_stats:
                self.daily_stats[date] = {
                    'trades': 0,
                    'profit': 0,
                    'wins': 0,
                    'losses': 0
                }
            
            if action == 'exit':
                self.daily_stats[date]['trades'] += 1
                pnl = trade['pnl']
                self.daily_stats[date]['profit'] += pnl
                if pnl > 0:
                    self.daily_stats[date]['wins'] += 1
                else:
                    self.daily_stats[date]['losses'] += 1
            
            logger.info(f"Trade logged: {action} | Price: {trade['price']:.2f} | Size: {size:.2f} | Equity: {trade['equity']:.2f}")
            
        except Exception as e:
            logger.error(f"Error logging trade: {e}")


class BacktestingAgent:
    """Main backtesting agent class with comprehensive analysis and reporting."""
    
    def __init__(self):
        """Initialize backtesting agent with proper directory structure."""
        try:
            self.base_path = Path.cwd()
            self.output_dir = self.base_path / 'output'
            self.data_dir = self.base_path / 'agents' / 'backtesting_agent' / 'historical_data'
            self.technical_indicators = {}
            self.scaler = None  # Will be initialized when needed
            
            # Create directory structure
            self.validate_directory_structure()
            
            logger.info("BacktestingAgent initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing BacktestingAgent: {e}")
            raise

    def validate_strategy(self, strategy: Dict, company_name: str) -> Dict:
        """Enhanced strategy validation with comprehensive checks."""
        try:
            # Define required fields and their default values
            required_fields = {
                'indicators': [],
                'entry_conditions': [],
                'exit_conditions': [],
                'trading_hours': {'start': '09:15', 'end': '15:20'},
                'initial_capital': 100000,
                'commission': 0.002,
                'risk_management': {
                    'max_position_size': 0.1,
                    'stop_loss': 0.02,
                    'take_profit': 0.03
                }
            }
            
            # Start with default strategy
            validated_strategy = self.get_default_strategy(company_name)
            
            # Update with provided strategy
            for field, default_value in required_fields.items():
                if field in strategy:
                    if isinstance(strategy[field], type(default_value)):
                        validated_strategy[field] = strategy[field]
                    else:
                        logger.warning(f"Invalid type for {field}, using default")
                        
            # Validate trading hours
            trading_hours = validated_strategy['trading_hours']
            try:
                start = datetime.strptime(trading_hours['start'], '%H:%M').time()
                end = datetime.strptime(trading_hours['end'], '%H:%M').time()
                market_open = datetime.strptime('09:15', '%H:%M').time()
                market_close = datetime.strptime('15:20', '%H:%M').time()
                
                if start < market_open or end > market_close:
                    logger.warning("Invalid trading hours, resetting to market hours")
                    validated_strategy['trading_hours'] = {
                        'start': '09:15',
                        'end': '15:20'
                    }
            except ValueError:
                logger.warning("Invalid time format, using default trading hours")
                validated_strategy['trading_hours'] = {
                    'start': '09:15',
                    'end': '15:20'
                }
                
            # Validate indicators and conditions
            validated_strategy['indicators'] = self._validate_indicators(
                validated_strategy.get('indicators', [])
            )
            
            indicator_names = [ind['name'] for ind in validated_strategy['indicators']]
            validated_strategy['entry_conditions'] = self._validate_conditions(
                validated_strategy.get('entry_conditions', []), 
                indicator_names
            )
            validated_strategy['exit_conditions'] = self._validate_conditions(
                validated_strategy.get('exit_conditions', []), 
                indicator_names
            )
            
            return validated_strategy
            
        except Exception as e:
            logger.error(f"Error validating strategy: {e}")
            return self.get_default_strategy(company_name)

    def _validate_indicators(self, indicators: List[Dict]) -> List[Dict]:
        """Validate and normalize indicator configurations."""
        try:
            valid_indicators = []
            supported_types = {'rsi', 'macd', 'sma', 'ema', 'bbands', 'supertrend', 'atr'}
            
            for ind in indicators:
                if not isinstance(ind, dict):
                    continue
                    
                ind_type = ind.get('type', '').lower()
                if ind_type not in supported_types:
                    continue
                    
                valid_ind = {
                    'type': ind_type,
                    'name': ind.get('name', ind_type),
                    'params': {}
                }
                
                # Validate parameters based on indicator type
                params = ind.get('params', {})
                if ind_type == 'rsi':
                    valid_ind['params']['length'] = int(params.get('length', 14))
                elif ind_type == 'macd':
                    valid_ind['params'].update({
                        'fast': int(params.get('fast', 12)),
                        'slow': int(params.get('slow', 26)),
                        'signal': int(params.get('signal', 9))
                    })
                elif ind_type in {'sma', 'ema'}:
                    valid_ind['params']['length'] = int(params.get('length', 20))
                elif ind_type == 'bbands':
                    valid_ind['params'].update({
                        'length': int(params.get('length', 20)),
                        'dev_up': float(params.get('dev_up', 2.0)),
                        'dev_down': float(params.get('dev_down', 2.0))
                    })
                
                valid_indicators.append(valid_ind)
                
            return valid_indicators
            
        except Exception as e:
            logger.error(f"Error validating indicators: {e}")
            return []

    def _validate_conditions(self, conditions: List[Dict], indicator_names: List[str]) -> List[Dict]:
        """Validate trading conditions against available indicators."""
        try:
            valid_conditions = []
            required_fields = {'indicator1', 'indicator2', 'condition', 'action'}
            
            for condition in conditions:
                if not isinstance(condition, dict):
                    continue
                    
                # Check required fields
                if not all(field in condition for field in required_fields):
                    continue
                    
                # Validate indicator references
                ind1 = condition['indicator1']
                ind2 = condition['indicator2']
                
                if ind1 not in indicator_names and not ind1.startswith('price'):
                    continue
                    
                if not ind2.isdigit() and ind2 not in indicator_names and not ind2.startswith('price'):
                    continue
                    
                # Validate condition type
                if condition['condition'] not in {'crossover', 'crossunder', 'above', 'below'}:
                    continue
                    
                # Validate action
                if condition['action'] not in {'buy', 'sell', 'exit'}:
                    continue
                    
                valid_conditions.append(condition)
                
            return valid_conditions
            
        except Exception as e:
            logger.error(f"Error validating conditions: {e}")
            return []


    def _validate_backtest_data(self, data: pd.DataFrame) -> bool:
        """Validate data before running backtest."""
        try:
            if data.empty:
                return False
                
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                return False
                
            # Check for sufficient data points
            if len(data) < 100:  # Minimum required for meaningful analysis
                return False
                
            # Check for too many missing values
            missing_threshold = 0.1  # 10% missing values threshold
            if data[required_columns].isnull().sum().max() / len(data) > missing_threshold:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating backtest data: {e}")
            return False

    def _validate_strategy(self, strategy: Dict) -> bool:
        """Validate strategy configuration."""
        try:
            required_fields = ['indicators', 'entry_conditions', 'exit_conditions']
            if not all(field in strategy for field in required_fields):
                return False
                
            # Check indicators configuration
            if not strategy['indicators']:
                return False
                
            # Check trading conditions
            if not strategy['entry_conditions'] or not strategy['exit_conditions']:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating strategy: {e}")
            return False

    def _prepare_backtest_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for backtesting."""
        try:
            # Create copy to avoid modifying original data
            data_copy = data.copy()
            
            # Handle missing values
            for col in ['Open', 'High', 'Low', 'Close']:
                data_copy[col] = data_copy[col].fillna(method='ffill')
            
            # Handle volume missing values separately
            data_copy['Volume'] = data_copy['Volume'].fillna(data_copy['Volume'].mean())
            
            # Remove any remaining NaN values
            data_copy = data_copy.dropna()
            
            return data_copy
            
        except Exception as e:
            logger.error(f"Error preparing backtest data: {e}")
            return data

    def initialize_indicator(self, indicator_config: Dict):
        """Initialize technical indicators with comprehensive validation."""
        try:
            indicator_type = indicator_config['type'].lower()
            params = indicator_config.get('params', {})
            name = indicator_config['name']
            
            # Add base price data
            if not hasattr(self, 'base_data_added'):
                self.indicators['close'] = self.data.Close
                self.indicators['high'] = self.data.High
                self.indicators['low'] = self.data.Low
                self.indicators['volume'] = self.data.Volume
                self.base_data_added = True
            
            # Initialize indicator based on type
            if indicator_type == 'rsi':
                self.indicators[name] = talib.RSI(
                    self.data.Close, 
                    timeperiod=params.get('length', 14)
                )
            elif indicator_type == 'macd':
                macd, signal, hist = talib.MACD(
                    self.data.Close,
                    fastperiod=params.get('fast', 12),
                    slowperiod=params.get('slow', 26),
                    signalperiod=params.get('signal', 9)
                )
                self.indicators[name] = macd
                self.indicators[f"{name}_signal"] = signal
                self.indicators[f"{name}_hist"] = hist
            elif indicator_type == 'bbands':
                upper, middle, lower = talib.BBANDS(
                    self.data.Close,
                    timeperiod=params.get('length', 20),
                    nbdevup=params.get('dev_up', 2),
                    nbdevdn=params.get('dev_down', 2)
                )
                self.indicators[f"{name}_upper"] = upper
                self.indicators[f"{name}_middle"] = middle
                self.indicators[f"{name}_lower"] = lower
            
            logger.info(f"Initialized indicator: {name}")
            
        except Exception as e:
            logger.error(f"Error initializing indicator: {e}")
            raise
        
    def initialize_alma(self, params: Dict) -> np.ndarray:
        """Initialize ALMA (Arnaud Legoux Moving Average) indicator."""
        try:
            length = params.get('length', 20)
            offset = params.get('offset', 0.85)
            sigma = params.get('sigma', 6)
            source = self.data[params.get('source', 'Close')].values
            
            # Calculate ALMA
            m = offset * (length - 1)
            s = length / sigma
            weights = np.zeros(length)
            
            for i in range(length):
                weights[i] = np.exp(-((i - m) ** 2) / (2 * s * s))
                
            weights = weights / np.sum(weights)
            alma = np.convolve(source, weights[::-1], mode='valid')
            
            # Pad beginning to match input length
            padding = np.full(length - 1, np.nan)
            return np.concatenate([padding, alma])
            
        except Exception as e:
            logger.error(f"Error calculating ALMA: {e}")
            return np.full_like(self.data.Close.values, np.nan)
        
    def load_latest_strategy(self, company_name: str) -> Tuple[Dict, int]:
            """Load the latest JSON strategy file."""
            try:
                pattern = str(self.output_dir / 'algo' / f"{company_name}_algorithm-*.json")
                files = glob.glob(pattern)
                
                if not files:
                    logger.warning(f"No strategy files found for {company_name}")
                    return self.get_default_strategy(company_name), 1
                
                # Get latest file based on algorithm number
                latest_file = max(files, key=lambda x: int(x.split('-')[-1].split('.')[0]))
                algo_num = int(latest_file.split('-')[-1].split('.')[0])
                
                with open(latest_file, 'r') as f:
                    strategy = json.load(f)
                
                # Validate and update strategy
                strategy = self.validate_strategy(strategy, company_name)
                
                logger.info(f"Loaded strategy {algo_num} for {company_name}")
                return strategy, algo_num
                
            except Exception as e:
                logger.error(f"Error loading strategy: {e}")
                return self.get_default_strategy(company_name), 1

    def get_default_strategy(self, company_name: str) -> Dict:
        """Return a default strategy with basic indicators and conditions."""
        default_strategy = {
            "indicators": [
                {
                    "type": "RSI",
                    "name": "rsi",
                    "params": {"length": 14}
                },
                {
                    "type": "MACD",
                    "name": "macd",
                    "params": {
                        "fast": 12,
                        "slow": 26,
                        "signal": 9
                    }
                },
                {
                    "type": "SMA",
                    "name": "sma_20",
                    "params": {"length": 20}
                },
                {
                    "type": "EMA",
                    "name": "ema_50",
                    "params": {"length": 50}
                }
            ],
            "entry_conditions": [
                {
                    "indicator1": "rsi",
                    "indicator2": "30",
                    "condition": "crossover",
                    "action": "buy",
                    "size": 0.95
                },
                {
                    "indicator1": "macd",
                    "indicator2": "signal",
                    "condition": "crossover",
                    "action": "buy",
                    "size": 0.95
                }
            ],
            "exit_conditions": [
                {
                    "indicator1": "rsi",
                    "indicator2": "70",
                    "condition": "crossover",
                    "action": "exit"
                },
                {
                    "indicator1": "macd",
                    "indicator2": "signal",
                    "condition": "crossunder",
                    "action": "exit"
                }
            ],
            "trading_hours": {
                "start": "09:15",
                "end": "15:20"
            },
            "initial_capital": 100000,
            "commission": 0.002,
            "risk_management": {
                "max_position_size": 0.1,
                "stop_loss": 0.02,
                "take_profit": 0.03
            }
        }
        
        # Save default strategy
        output_dir = self.output_dir / 'algo'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        strategy_file = output_dir / f"{company_name}_algorithm-1.json"
        with open(strategy_file, 'w') as f:
            json.dump(default_strategy, f, indent=4)
        
        logger.info(f"Created default strategy for {company_name}")
        return default_strategy

    def load_historical_data(self, company_name: str, start_date: str = None, 
                           end_date: str = None) -> pd.DataFrame:
        """Load and preprocess historical data with extended columns."""
        try:
            data_file = self.data_dir / f"{company_name}_minute.csv"
            
            if not data_file.exists():
                raise FileNotFoundError(f"Historical data file not found: {data_file}")
            
            # Read CSV with all columns
            df = pd.read_csv(data_file)
            
            # Handle timezone-aware datetime
            df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_localize(None)
            df = df.set_index('Datetime')
            
            # Filter by date range
            if start_date:
                start_datetime = pd.to_datetime(start_date).tz_localize(None)
                df = df[df.index >= start_datetime]
            if end_date:
                end_datetime = pd.to_datetime(end_date).tz_localize(None)
                df = df[df.index <= end_datetime]
            
            # Ensure required columns exist and are numeric
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle missing values
            df['Volume'] = df['Volume'].replace(0, np.nan)
            df['Volume'] = df['Volume'].fillna(df['Volume'].mean())
            
            # Calculate basic technical indicators
            df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
            df['EMA_50'] = talib.EMA(df['Close'], timeperiod=50)
            df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
            macd, signal, hist = talib.MACD(df['Close'])
            df['MACD'] = macd
            df['MACD_Signal'] = signal
            df['MACD_Hist'] = hist
            
            # Store indicators for strategy use
            self.technical_indicators = {
                'sma_20': df['SMA_20'].values,
                'ema_50': df['EMA_50'].values,
                'rsi': df['RSI'].values,
                'macd': df['MACD'].values,
                'macd_signal': df['MACD_Signal'].values,
                'macd_hist': df['MACD_Hist'].values
            }
            
            logger.info(f"Loaded {len(df)} records from {df.index[0]} to {df.index[-1]}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            raise

    def validate_directory_structure(self):
        """Validate and create required directory structure."""
        try:
            required_dirs = [
                self.data_dir,
                self.output_dir / 'backtest_results',
                self.output_dir / 'algo',
                self.output_dir / 'reports'
            ]
            
            for directory in required_dirs:
                directory.mkdir(parents=True, exist_ok=True)
                
            logger.info("Directory structure validated and created")
            
        except Exception as e:
            logger.error(f"Error validating directory structure: {e}")
            raise


    def _execute_backtest(self, strategy: Dict, data: pd.DataFrame) -> Dict:
        """Execute backtest with the given strategy and data."""
        try:
            # Initialize strategy class
            def strategy_factory(json_config, technical_indicators):
                class CurrentStrategy(JSONStrategy):
                    pass
                CurrentStrategy.json_config = json_config
                CurrentStrategy.technical_indicators = technical_indicators
                return CurrentStrategy

            # Create and run backtest
            backtest = Backtest(
                data,
                strategy_factory(strategy, self.technical_indicators),
                cash=strategy.get('initial_capital', 100000),
                commission=strategy.get('commission', 0.002),
                exclusive_orders=True
            )

            # Run backtest and get results
            stats = backtest.run()

            # Calculate additional metrics
            stats['Start'] = data.index[0]
            stats['End'] = data.index[-1]
            
            # Calculate annualized return
            days = (data.index[-1] - data.index[0]).days
            if days > 0:
                total_return = stats.get('Return [%]', 0)
                annual_return = (1 + total_return/100)**(252/days) - 1
                stats['Return (Ann.) [%]'] = annual_return * 100
            else:
                stats['Return (Ann.) [%]'] = 0.0

            # Ensure all required metrics exist
            required_metrics = {
                'Return [%]': 0.0,
                'Sharpe Ratio': 0.0,
                'Sortino Ratio': 0.0,
                'Max. Drawdown [%]': 0.0,
                'Win Rate [%]': 0.0,
                '# Trades': 0,
                'Profit Factor': 0.0,
                'Avg. Trade [%]': 0.0,
                'Best Trade [%]': 0.0,
                'Worst Trade [%]': 0.0,
                '_trades': [],
                '_equity_curve': pd.DataFrame({'Equity': data['Close']})
            }

            # Update with actual values while keeping defaults for missing ones
            for key, default_value in required_metrics.items():
                if key not in stats:
                    stats[key] = default_value

            return stats

        except Exception as e:
            logger.error(f"Error executing backtest: {e}")
            logger.debug("Exception details:", exc_info=True)
            
            # Return minimal valid stats with defaults
            return {
                'Return [%]': 0.0,
                'Return (Ann.) [%]': 0.0,
                'Sharpe Ratio': 0.0,
                'Sortino Ratio': 0.0,
                'Max. Drawdown [%]': 0.0,
                'Win Rate [%]': 0.0,
                '# Trades': 0,
                'Profit Factor': 0.0,
                'Avg. Trade [%]': 0.0,
                'Best Trade [%]': 0.0,
                'Worst Trade [%]': 0.0,
                '_trades': [],
                '_equity_curve': pd.DataFrame({'Equity': data['Close']}),
                'Start': data.index[0],
                'End': data.index[-1]
            }


    def run_backtest(self, company_name: str, start_date: str = None, 
                    end_date: str = None) -> Dict:
        """Run backtest for the given company"""
        try:
            # Load and validate strategy
            strategy, algo_num = self.load_latest_strategy(company_name)
            
            # Load and prepare data
            data = self.load_historical_data(company_name, start_date, end_date)
            
            # Validate data before proceeding
            if data.empty:
                raise ValueError("No data available for backtesting")

            # Execute backtest
            stats = self._execute_backtest(strategy, data)
            
            # Create report directory
            report_dir = self.output_dir / 'backtest_results' / f"{company_name}_algo{algo_num}"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # Analyze trades
            if self.validate_trades_data(stats.get('_trades', [])):
                trade_metrics = self.analyze_trades(stats['_trades'])
                stats.update(trade_metrics)
            else:
                logger.warning("Invalid trades data - using default metrics")
                stats.update(self.analyze_trades([]))
            
            # Generate reports with error handling
            try:
                self.generate_reports(stats, company_name, algo_num, data)
            except Exception as e:
                logger.error(f"Error in report generation: {e}")
                
            # Update basic results
            try:
                self.update_basic_results(stats, report_dir)
            except Exception as e:
                logger.error(f"Error updating basic results: {e}")
                
            return stats
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            raise


    def _save_basic_results(self, stats: Dict, report_dir: Path):
        """Save basic backtest results as fallback."""
        try:
            basic_stats = {
                'total_trades': stats.get('# Trades', 0),
                'total_return': stats.get('Return [%]', 0),
                'sharpe_ratio': stats.get('Sharpe Ratio', 0),
                'max_drawdown': stats.get('Max. Drawdown [%]', 0)
            }
            
            with open(report_dir / 'basic_results.json', 'w') as f:
                json.dump(basic_stats, f, indent=4)
                
        except Exception as e:
            logger.error(f"Error saving basic results: {e}")
            logger.error(f"Error running backtest: {e}")
            logger.debug("Exception details:", exc_info=True)
            raise

    def generate_reports(self, stats: Dict, company_name: str, algo_num: int, data: pd.DataFrame):
        """Generate comprehensive backtest reports with error handling."""
        try:
            report_dir = self.output_dir / 'backtest_results' / f"{company_name}_algo{algo_num}"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # List of report generation functions
            report_functions = [
                (self.save_summary_stats, (stats, report_dir)),
                (self.plot_equity_curve, (stats, report_dir)),
                (self.plot_drawdown_analysis, (stats, report_dir)),
                (self.generate_trade_analysis, (stats, report_dir)),
                (self.plot_monthly_returns, (stats, report_dir))
            ]
            
            # Execute each report function with error handling
            for func, args in report_functions:
                try:
                    func(*args)
                except Exception as e:
                    logger.error(f"Error generating {func.__name__}: {str(e)}")
                    # Continue with other reports even if one fails
                    continue
            
            logger.info(f"Reports generated successfully for {company_name} algorithm {algo_num}")
            
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
            raise

    def save_summary_stats(self, stats: Dict, report_dir: Path):
        """Save detailed summary statistics."""
        try:
            with open(report_dir / 'summary_stats.txt', 'w') as f:
                f.write("Performance Summary\n")
                f.write("=" * 50 + "\n\n")
                
                # Trading Performance
                f.write("Trading Performance:\n")
                f.write(f"Total Return: {stats['Return [%]']:.2f}%\n")
                f.write(f"Annual Return: {stats['Return (Ann.) [%]']:.2f}%\n")
                f.write(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}\n")
                f.write(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%\n")
                f.write(f"Win Rate: {stats['Win Rate [%]']:.2f}%\n\n")
                
                # Risk Metrics
                f.write("Risk Metrics:\n")
                f.write(f"Volatility (Ann.): {stats['Volatility (Ann.) [%]']:.2f}%\n")
                f.write(f"Sortino Ratio: {stats['Sortino Ratio']:.2f}\n")
                f.write(f"Calmar Ratio: {stats['Calmar Ratio']:.2f}\n")
                f.write(f"Average Drawdown: {stats['Avg. Drawdown [%]']:.2f}%\n\n")
                
                # Trade Statistics
                f.write("Trade Statistics:\n")
                f.write(f"Total Trades: {stats['# Trades']}\n")
                f.write(f"Best Trade: {stats['Best Trade [%]']:.2f}%\n")
                f.write(f"Worst Trade: {stats['Worst Trade [%]']:.2f}%\n")
                f.write(f"Average Trade: {stats['Avg. Trade [%]']:.2f}%\n")
                f.write(f"Profit Factor: {stats['Profit Factor']:.2f}\n\n")
                
                # Time Analysis
                f.write("Time Analysis:\n")
                f.write(f"Start: {stats['Start']}\n")
                f.write(f"End: {stats['End']}\n")
                f.write(f"Duration: {stats['Duration']}\n")
                f.write(f"Exposure Time: {stats['Exposure Time [%]']:.2f}%\n")
                
        except Exception as e:
            logger.error(f"Error saving summary stats: {e}")
            raise

    def generate_trade_analysis(self, stats: Dict, report_dir: Path):
        """Generate detailed trade analysis with visualization."""
        try:
            # Extract trades from stats and convert to DataFrame
            trades = stats.get('_trades', [])
            
            # Check if trades list is empty using proper method
            if len(trades) == 0:
                logger.warning("No trades found for analysis")
                return
                
            trades_df = pd.DataFrame([{
                'Entry Time': t.entry_time,
                'Exit Time': t.exit_time,
                'Entry Price': t.entry_price,
                'Exit Price': t.exit_price,
                'Size': t.size,
                'PnL': t.pl,
                'Return': t.pl_pct,
                'Duration': (t.exit_time - t.entry_time).total_seconds() / 3600
            } for t in trades])
                
            # Generate trade analysis plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
                
            # PnL Distribution
            sns.histplot(data=trades_df['PnL'], ax=ax1, bins=20)
            ax1.set_title('Trade PnL Distribution')
            ax1.set_xlabel('PnL ($)')
                
            # Return vs Duration
            ax2.scatter(trades_df['Duration'], trades_df['Return'])
            ax2.set_title('Trade Return vs Duration')
            ax2.set_xlabel('Duration (hours)')
            ax2.set_ylabel('Return (%)')
                
            # Cumulative PnL
            cumulative_pnl = trades_df['PnL'].cumsum()
            ax3.plot(range(len(cumulative_pnl)), cumulative_pnl)
            ax3.set_title('Cumulative PnL')
            ax3.set_xlabel('Trade Number')
            ax3.set_ylabel('Cumulative PnL ($)')
                
            # Win/Loss Distribution
            trades_df['Result'] = trades_df['PnL'].apply(
                lambda x: 'Win' if x > 0 else 'Loss'
            )
            sns.boxplot(data=trades_df, x='Result', y='PnL', ax=ax4)
            ax4.set_title('PnL by Trade Outcome')
                
            plt.tight_layout()
            plt.savefig(report_dir / 'trade_analysis.png')
            plt.close()
                
            # Save trade data
            trades_df.to_csv(report_dir / 'trades.csv', index=False)
                
            # Calculate and save additional statistics
            stats_df = pd.DataFrame({
                'Metric': [
                    'Total Trades',
                    'Winning Trades',
                    'Losing Trades',
                    'Win Rate',
                    'Average Win',
                    'Average Loss',
                    'Profit Factor',
                    'Average Duration'
                ],
                'Value': [
                    len(trades_df),
                    len(trades_df[trades_df['PnL'] > 0]),
                    len(trades_df[trades_df['PnL'] < 0]),
                    f"{(len(trades_df[trades_df['PnL'] > 0]) / len(trades_df) * 100):.2f}%",
                    f"${trades_df[trades_df['PnL'] > 0]['PnL'].mean():.2f}",
                    f"${trades_df[trades_df['PnL'] < 0]['PnL'].mean():.2f}",
                    f"{abs(trades_df[trades_df['PnL'] > 0]['PnL'].sum() / trades_df[trades_df['PnL'] < 0]['PnL'].sum()):.2f}",
                    f"{trades_df['Duration'].mean():.2f} hours"
                ]
            })
            stats_df.to_csv(report_dir / 'trade_statistics.csv', index=False)
                
        except Exception as e:
            logger.error(f"Error generating trade analysis: {e}")
            logger.debug("Exception details:", exc_info=True)
            raise

    def plot_equity_curve(self, stats: Dict, report_dir: Path):
            """Generate equity curve plot with drawdown overlay."""
            try:
                equity_data = pd.DataFrame({
                    'Equity': stats['_equity_curve']['Equity'],
                    'DrawdownPct': stats['_equity_curve']['DrawdownPct']
                })
                
                # Create figure with secondary y-axis
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), 
                                            gridspec_kw={'height_ratios': [3, 1]})
                
                # Plot equity curve
                ax1.plot(equity_data.index, equity_data['Equity'], 
                        label='Portfolio Value', color='blue')
                ax1.set_title('Portfolio Performance and Drawdown Analysis')
                ax1.set_ylabel('Portfolio Value ($)')
                ax1.grid(True)
                ax1.legend()
                
                # Plot drawdown
                ax2.fill_between(equity_data.index, 0, -equity_data['DrawdownPct'],
                            color='red', alpha=0.3, label='Drawdown')
                ax2.set_ylabel('Drawdown (%)')
                ax2.set_xlabel('Date')
                ax2.grid(True)
                ax2.legend()
                
                plt.tight_layout()
                plt.savefig(report_dir / 'equity_curve.png')
                plt.close()
                
                # Create interactive version
                fig = go.Figure()
                
                # Add equity curve
                fig.add_trace(go.Scatter(
                    x=equity_data.index,
                    y=equity_data['Equity'],
                    name='Portfolio Value',
                    line=dict(color='blue')
                ))
                
                # Add drawdown
                fig.add_trace(go.Scatter(
                    x=equity_data.index,
                    y=-equity_data['DrawdownPct'],
                    name='Drawdown',
                    fill='tozeroy',
                    line=dict(color='red')
                ))
                
                fig.update_layout(
                    title='Interactive Equity Curve and Drawdown',
                    yaxis_title='Value',
                    hovermode='x unified'
                )
                
                fig.write_html(str(report_dir / 'interactive_equity.html'))
                
            except Exception as e:
                logger.error(f"Error plotting equity curve: {e}")
                raise

    def plot_drawdown_analysis(self, stats: Dict, report_dir: Path):
        """Generate detailed drawdown analysis plots."""
        try:
            drawdown_data = pd.DataFrame({
                'Drawdown': stats['_equity_curve']['DrawdownPct']
            })
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # Drawdown distribution
            sns.histplot(data=-drawdown_data['Drawdown'], ax=ax1, bins=50)
            ax1.set_title('Drawdown Distribution')
            ax1.set_xlabel('Drawdown (%)')
            ax1.set_ylabel('Frequency')
            
            # Drawdown duration analysis
            drawdown_periods = self.get_drawdown_periods(drawdown_data)
            if drawdown_periods:
                durations = [period['duration'].days for period in drawdown_periods]
                magnitudes = [period['max_drawdown'] for period in drawdown_periods]
                
                ax2.scatter(durations, magnitudes)
                ax2.set_title('Drawdown Duration vs Magnitude')
                ax2.set_xlabel('Duration (Days)')
                ax2.set_ylabel('Maximum Drawdown (%)')
                ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(report_dir / 'drawdown_analysis.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting drawdown analysis: {e}")
            raise

    def plot_monthly_returns(self, stats: Dict, report_dir: Path):
        """Generate monthly returns heatmap with proper error handling."""
        try:
            # Extract trade data
            trades = stats.get('_trades', [])
            
            # Check if there are any trades
            if not trades:
                logger.warning("No trades available for monthly returns analysis")
                # Create a simple placeholder plot
                plt.figure(figsize=(12, 8))
                plt.text(0.5, 0.5, 'No Trade Data Available', 
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=14)
                plt.axis('off')
                plt.savefig(report_dir / 'monthly_returns.png')
                plt.close()
                return

            trades_df = pd.DataFrame({
                'Exit Time': [t.exit_time for t in trades],
                'Return': [t.pl_pct for t in trades]
            })
            
            # Set datetime index
            trades_df['Exit Time'] = pd.to_datetime(trades_df['Exit Time'])
            trades_df = trades_df.set_index('Exit Time')
            
            # Calculate monthly returns
            monthly_returns = trades_df.resample('M')['Return'].sum()
            
            if monthly_returns.empty:
                logger.warning("No monthly returns data available")
                plt.figure(figsize=(12, 8))
                plt.text(0.5, 0.5, 'No Monthly Returns Data Available', 
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=14)
                plt.axis('off')
                plt.savefig(report_dir / 'monthly_returns.png')
                plt.close()
                return
                
            # Create pivot table with filled missing values
            monthly_returns = monthly_returns.to_frame()
            monthly_returns['Year'] = monthly_returns.index.year
            monthly_returns['Month'] = monthly_returns.index.month
            
            pivot_table = monthly_returns.pivot(
                index='Year',
                columns='Month',
                values='Return'
            ).fillna(0)  # Fill NaN with 0
            
            # Check if pivot table has data
            if pivot_table.empty or pivot_table.size == 0:
                logger.warning("Empty pivot table for monthly returns")
                plt.figure(figsize=(12, 8))
                plt.text(0.5, 0.5, 'Insufficient Data for Monthly Returns Analysis', 
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=14)
                plt.axis('off')
                plt.savefig(report_dir / 'monthly_returns.png')
                plt.close()
                return
            
            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                pivot_table,
                annot=True,
                fmt='.1f',
                cmap='RdYlGn',
                center=0,
                cbar_kws={'label': 'Return (%)'}
            )
            
            plt.title('Monthly Returns Heatmap')
            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            plt.xticks(np.arange(12) + 0.5, month_labels, rotation=0)
            
            plt.tight_layout()
            plt.savefig(report_dir / 'monthly_returns.png')
            plt.close()
            
            # Save monthly returns data
            pivot_table.to_csv(report_dir / 'monthly_returns.csv')
            
            logger.info("Monthly returns analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Error plotting monthly returns: {e}")
            # Create error notification plot
            plt.figure(figsize=(12, 8))
            plt.text(0.5, 0.5, 'Error Generating Monthly Returns Plot', 
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=14,
                    color='red')
            plt.axis('off')
            plt.savefig(report_dir / 'monthly_returns.png')
            plt.close()
            raise

    def plot_indicators_with_trades(self, df: pd.DataFrame, trades: List, 
                                  indicators: Dict, report_dir: Path):
        """Generate comprehensive indicator plot with trade points."""
        try:
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Price Action', 'RSI', 'MACD', 'Volume'),
                row_heights=[0.4, 0.2, 0.2, 0.2]
            )
            
            # Price and trades
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Add moving averages
            if 'SMA_20' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['SMA_20'],
                        name='SMA 20',
                        line=dict(color='blue', width=1)
                    ),
                    row=1, col=1
                )
            
            if 'EMA_50' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['EMA_50'],
                        name='EMA 50',
                        line=dict(color='orange', width=1)
                    ),
                    row=1, col=1
                )
            
            # Add trades
            buy_points = [(t.entry_time, t.entry_price) for t in trades if t.size > 0]
            sell_points = [(t.exit_time, t.exit_price) for t in trades if t.size > 0]
            
            if buy_points:
                buy_times, buy_prices = zip(*buy_points)
                fig.add_trace(
                    go.Scatter(
                        x=buy_times,
                        y=buy_prices,
                        mode='markers',
                        name='Buy',
                        marker=dict(color='green', size=10, symbol='triangle-up')
                    ),
                    row=1, col=1
                )
            
            if sell_points:
                sell_times, sell_prices = zip(*sell_points)
                fig.add_trace(
                    go.Scatter(
                        x=sell_times,
                        y=sell_prices,
                        mode='markers',
                        name='Sell',
                        marker=dict(color='red', size=10, symbol='triangle-down')
                    ),
                    row=1, col=1
                )
            
            # Add RSI
            if 'RSI' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['RSI'], name='RSI'),
                    row=2, col=1
                )
                fig.add_hrect(
                    y0=70, y1=100,
                    fillcolor="red", opacity=0.1,
                    layer="below", line_width=0,
                    row=2, col=1
                )
                fig.add_hrect(
                    y0=0, y1=30,
                    fillcolor="green", opacity=0.1,
                    layer="below", line_width=0,
                    row=2, col=1
                )
            
            # Add MACD
            if all(x in df.columns for x in ['MACD', 'MACD_Signal']):
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['MACD'], name='MACD'),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal'),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Hist'),
                    row=3, col=1
                )
            
            # Add Volume
            fig.add_trace(
                go.Bar(x=df.index, y=df['Volume'], name='Volume'),
                row=4, col=1
            )
            
            # Update layout
            fig.update_layout(
                title='Trading Activity with Indicators',
                xaxis_rangeslider_visible=False,
                height=1200,
                showlegend=True
            )
            
            # Save both static and interactive versions
            fig.write_html(str(report_dir / 'indicators_trades.html'))
            fig.write_image(str(report_dir / 'indicators_trades.png'))
            
        except Exception as e:
            logger.error(f"Error plotting indicators with trades: {e}")
            raise

    def generate_indicator_analysis(self, df: pd.DataFrame, trades: List):
        """Generate analysis of indicator effectiveness."""
        try:
            analysis = {
                'RSI': {'true_positives': 0, 'false_positives': 0},
                'MACD': {'true_positives': 0, 'false_positives': 0}
            }
            
            for trade in trades:
                entry_idx = df.index.get_loc(trade.entry_time)
                
                # RSI Analysis
                if 'RSI' in df.columns:
                    rsi_value = df['RSI'].iloc[entry_idx]
                    if rsi_value < 30 and trade.pl > 0:  # Oversold and profitable
                        analysis['RSI']['true_positives'] += 1
                    elif rsi_value < 30:  # Oversold but not profitable
                        analysis['RSI']['false_positives'] += 1
                
                # MACD Analysis
                if all(x in df.columns for x in ['MACD', 'MACD_Signal']):
                    macd = df['MACD'].iloc[entry_idx]
                    signal = df['MACD_Signal'].iloc[entry_idx]
                    if macd > signal and trade.pl > 0:  # Bullish crossover and profitable
                        analysis['MACD']['true_positives'] += 1
                    elif macd > signal:  # Bullish crossover but not profitable
                        analysis['MACD']['false_positives'] += 1
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing indicators: {e}")
            raise

    def cleanup_old_results(self, keep_days: int = 30):
        """Clean up old backtest results to save disk space."""
        try:
            now = datetime.now()
            results_dir = self.output_dir / 'backtest_results'
            
            if not results_dir.exists():
                return
            
            for result_dir in results_dir.iterdir():
                if not result_dir.is_dir():
                    continue
                
                mtime = datetime.fromtimestamp(result_dir.stat().st_mtime)
                if (now - mtime).days > keep_days:
                    shutil.rmtree(result_dir)
                    logger.info(f"Removed old backtest results: {result_dir}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old results: {e}")

    def get_drawdown_periods(self, drawdown_data: pd.DataFrame) -> List[Dict]:
        """Extract drawdown periods for analysis."""
        try:
            periods = []
            in_drawdown = False
            start_idx = None
            max_drawdown = 0
            
            for idx, row in drawdown_data.iterrows():
                drawdown = -row['Drawdown']
                
                if drawdown > 0:
                    if not in_drawdown:
                        in_drawdown = True
                        start_idx = idx
                    max_drawdown = max(max_drawdown, drawdown)
                elif in_drawdown:
                    periods.append({
                        'start': start_idx,
                        'end': idx,
                        'duration': idx - start_idx,
                        'max_drawdown': max_drawdown
                    })
                    in_drawdown = False
                    max_drawdown = 0
            
            return periods
            
        except Exception as e:
            logger.error(f"Error analyzing drawdown periods: {e}")
            return []    

    def generate_trade_data_csv(self, trades: List, report_dir: Path):
            """Generate detailed CSV of all trades with indicators at entry/exit."""
            try:
                trade_data = []
                for trade in trades:
                    # Basic trade information
                    trade_info = {
                        'Entry Time': trade.entry_time,
                        'Exit Time': trade.exit_time,
                        'Entry Price': trade.entry_price,
                        'Exit Price': trade.exit_price,
                        'Size': trade.size,
                        'PnL': trade.pl,
                        'Return %': trade.pl_pct,
                        'Duration': trade.exit_time - trade.entry_time,
                    }
                    
                    # Add indicator values at entry
                    for name, values in self.technical_indicators.items():
                        if len(values) > trade.entry_idx:
                            trade_info[f'Entry_{name}'] = values[trade.entry_idx]
                        if len(values) > trade.exit_idx:
                            trade_info[f'Exit_{name}'] = values[trade.exit_idx]
                    
                    trade_data.append(trade_info)
                
                # Convert to DataFrame and save
                df = pd.DataFrame(trade_data)
                df.to_csv(report_dir / 'detailed_trades.csv', index=False)
                
                # Save Excel version with multiple sheets
                with pd.ExcelWriter(report_dir / 'trade_analysis.xlsx') as writer:
                    df.to_excel(writer, sheet_name='Detailed Trades', index=False)
                    
                    # Add summary statistics
                    summary = pd.DataFrame({
                        'Metric': [
                            'Total Trades',
                            'Winning Trades',
                            'Losing Trades',
                            'Win Rate',
                            'Average Win',
                            'Average Loss',
                            'Profit Factor',
                            'Average Duration'
                        ],
                        'Value': [
                            len(df),
                            len(df[df['PnL'] > 0]),
                            len(df[df['PnL'] < 0]),
                            f"{(len(df[df['PnL'] > 0]) / len(df) * 100):.2f}%",
                            f"${df[df['PnL'] > 0]['PnL'].mean():.2f}",
                            f"${df[df['PnL'] < 0]['PnL'].mean():.2f}",
                            f"{abs(df[df['PnL'] > 0]['PnL'].sum() / df[df['PnL'] < 0]['PnL'].sum()):.2f}",
                            f"{df['Duration'].mean().total_seconds() / 3600:.2f} hours"
                        ]
                    })
                    summary.to_excel(writer, sheet_name='Summary', index=False)
                    
            except Exception as e:
                logger.error(f"Error generating trade data CSV: {e}")
                raise

    def generate_daily_summary_csv(self, trades: List, report_dir: Path):
        """Generate daily summary of trading activity."""
        try:
            # Convert trades to DataFrame
            trades_df = pd.DataFrame([{
                'Date': t.exit_time.date(),
                'PnL': t.pl,
                'Return': t.pl_pct,
                'Trade Count': 1,
                'Win': 1 if t.pl > 0 else 0,
                'Loss': 1 if t.pl < 0 else 0
            } for t in trades])
            
            # Group by date
            daily_summary = trades_df.groupby('Date').agg({
                'PnL': ['sum', 'mean', 'std'],
                'Return': ['sum', 'mean', 'std'],
                'Trade Count': 'sum',
                'Win': 'sum',
                'Loss': 'sum'
            }).round(4)
            
            # Flatten column names
            daily_summary.columns = ['_'.join(col).strip() for col in daily_summary.columns]
            
            # Calculate additional metrics
            daily_summary['Win_Rate'] = (daily_summary['Win_sum'] / 
                                       daily_summary['Trade Count_sum'] * 100).round(2)
            daily_summary['Cumulative_PnL'] = daily_summary['PnL_sum'].cumsum()
            
            # Save to CSV and Excel
            daily_summary.to_csv(report_dir / 'daily_summary.csv')
            
            # Create Excel with formatting
            with pd.ExcelWriter(report_dir / 'daily_summary.xlsx') as writer:
                daily_summary.to_excel(writer, sheet_name='Daily Summary')
                workbook = writer.book
                worksheet = writer.sheets['Daily Summary']
                
                # Add formats
                money_fmt = workbook.add_format({'num_format': '$#,##0.00'})
                pct_fmt = workbook.add_format({'num_format': '0.00%'})
                
                # Apply formats to specific columns
                for idx, col in enumerate(daily_summary.columns):
                    if 'PnL' in col or 'Return' in col:
                        worksheet.set_column(idx + 1, idx + 1, None, money_fmt)
                    elif 'Rate' in col:
                        worksheet.set_column(idx + 1, idx + 1, None, pct_fmt)
                
        except Exception as e:
            logger.error(f"Error generating daily summary: {e}")
            raise

    def save_trade_log(self, trades_log: List[Dict], report_dir: Path):
            """Save detailed trade log with additional metrics."""
            try:
                trades_df = pd.DataFrame(trades_log)
                
                # Calculate additional metrics
                if len(trades_df) > 0:
                    trades_df['trade_duration'] = trades_df['time'].diff()
                    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
                    trades_df['rolling_win_rate'] = trades_df['pnl'].rolling(
                        window=20).apply(lambda x: (x > 0).mean())
                    
                    # Add market context
                    trades_df['market_condition'] = trades_df['price'].pct_change(20).apply(
                        lambda x: 'Uptrend' if x > 0.02 else 'Downtrend' if x < -0.02 else 'Range')
                    
                    # Calculate trade-specific metrics
                    trades_df['risk_reward_ratio'] = abs(trades_df['pnl'] / 
                        trades_df['price'].diff().abs())
                    trades_df['efficiency'] = trades_df['pnl'] / trades_df['price'].diff().abs()
                    trades_df['max_favorable_excursion'] = trades_df.groupby(
                        trades_df['time'].dt.date)['price'].transform('max') - trades_df['price']
                    trades_df['max_adverse_excursion'] = trades_df['price'] - trades_df.groupby(
                        trades_df['time'].dt.date)['price'].transform('min')
                
                # Save detailed trade log to Excel with multiple sheets
                with pd.ExcelWriter(report_dir / 'detailed_trade_log.xlsx') as writer:
                    # Main trade log
                    trades_df.to_excel(writer, sheet_name='Trade Log', index=False)
                    
                    # Trade summary by market condition
                    if len(trades_df) > 0:
                        market_summary = trades_df.groupby('market_condition').agg({
                            'pnl': ['count', 'sum', 'mean'],
                            'rolling_win_rate': 'mean',
                            'risk_reward_ratio': 'mean',
                            'efficiency': 'mean'
                        }).round(4)
                        market_summary.columns = ['_'.join(col).strip() for col in market_summary.columns]
                        market_summary.to_excel(writer, sheet_name='Market Analysis')
                    
                    # Time-based analysis
                    if len(trades_df) > 0:
                        trades_df['hour'] = trades_df['time'].dt.hour
                        time_analysis = trades_df.groupby('hour').agg({
                            'pnl': ['count', 'sum', 'mean'],
                            'rolling_win_rate': 'mean'
                        }).round(4)
                        time_analysis.columns = ['_'.join(col).strip() for col in time_analysis.columns]
                        time_analysis.to_excel(writer, sheet_name='Time Analysis')
                    
                    # Trade statistics summary
                    summary_stats = pd.DataFrame({
                        'Metric': [
                            'Total Trades',
                            'Winning Trades',
                            'Losing Trades',
                            'Win Rate',
                            'Average Win',
                            'Average Loss',
                            'Largest Win',
                            'Largest Loss',
                            'Average Risk-Reward',
                            'Average Efficiency',
                            'Total PnL'
                        ],
                        'Value': [
                            len(trades_df),
                            len(trades_df[trades_df['pnl'] > 0]),
                            len(trades_df[trades_df['pnl'] < 0]),
                            f"{(len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100):.2f}%",
                            f"${trades_df[trades_df['pnl'] > 0]['pnl'].mean():.2f}",
                            f"${trades_df[trades_df['pnl'] < 0]['pnl'].mean():.2f}",
                            f"${trades_df['pnl'].max():.2f}",
                            f"${trades_df['pnl'].min():.2f}",
                            f"{trades_df['risk_reward_ratio'].mean():.2f}",
                            f"{trades_df['efficiency'].mean():.2f}",
                            f"${trades_df['pnl'].sum():.2f}"
                        ]
                    })
                    summary_stats.to_excel(writer, sheet_name='Summary', index=False)
                
                # Save CSV version for easy import
                trades_df.to_csv(report_dir / 'trade_log.csv', index=False)
                
                logger.info(f"Trade log saved with {len(trades_df)} entries")
                
            except Exception as e:
                logger.error(f"Error saving trade log: {e}")
                raise

    def generate_interactive_plot(self, df: pd.DataFrame, trades: List, 
                                indicators: Dict, report_dir: Path):
        """Generate interactive HTML plot using plotly."""
        try:
            # Create figure with subplots
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Price & Indicators', 'RSI', 'MACD', 'Volume'),
                row_heights=[0.5, 0.15, 0.15, 0.2]
            )
            
            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Add moving averages
            if 'SMA_20' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['SMA_20'],
                        name='SMA 20',
                        line=dict(color='blue', width=1)
                    ),
                    row=1, col=1
                )
            
            if 'EMA_50' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['EMA_50'],
                        name='EMA 50',
                        line=dict(color='orange', width=1)
                    ),
                    row=1, col=1
                )
            
            # Add trades
            for trade in trades:
                # Entry points
                fig.add_trace(
                    go.Scatter(
                        x=[trade.entry_time],
                        y=[trade.entry_price],
                        mode='markers',
                        name='Buy' if trade.size > 0 else 'Sell',
                        marker=dict(
                            symbol='triangle-up' if trade.size > 0 else 'triangle-down',
                            size=10,
                            color='green' if trade.size > 0 else 'red'
                        )
                    ),
                    row=1, col=1
                )
                
                # Exit points
                fig.add_trace(
                    go.Scatter(
                        x=[trade.exit_time],
                        y=[trade.exit_price],
                        mode='markers',
                        name='Exit',
                        marker=dict(
                            symbol='x',
                            size=10,
                            color='black'
                        )
                    ),
                    row=1, col=1
                )
            
            # Add RSI
            if 'RSI' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['RSI'], name='RSI'),
                    row=2, col=1
                )
                # Add RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # Add MACD
            if all(x in df.columns for x in ['MACD', 'MACD_Signal']):
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['MACD'], name='MACD'),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal'),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Hist'),
                    row=3, col=1
                )
            
            # Add Volume
            colors = ['red' if c < o else 'green' 
                     for c, o in zip(df['Close'], df['Open'])]
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color=colors
                ),
                row=4, col=1
            )
            
            # Update layout
            fig.update_layout(
                title='Trading Activity with Technical Indicators',
                xaxis_rangeslider_visible=False,
                height=1200,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Add range selector
            fig.update_xaxes(
                rangeslider_visible=False,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                row=4, col=1
            )
            
            # Save interactive plot
            fig.write_html(str(report_dir / 'interactive_chart.html'))
            
            logger.info("Generated interactive plot")
            
        except Exception as e:
            logger.error(f"Error generating interactive plot: {e}")
            raise     
         
    def generate_performance_report(self, stats: Dict, company_name: str, algo_num: int):
            """Generate comprehensive HTML performance report with detailed metrics and visualizations."""
            try:
                report_dir = self.output_dir / 'backtest_results' / f"{company_name}_algo{algo_num}"
                report_dir.mkdir(parents=True, exist_ok=True)
                
                html_content = f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Backtest Performance Report - {company_name}</title>
                    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                    <style>
                        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
                        .dashboard-card {{
                            background: #fff;
                            border-radius: 10px;
                            padding: 20px;
                            margin: 10px;
                            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                            transition: transform 0.3s ease;
                        }}
                        .dashboard-card:hover {{
                            transform: translateY(-5px);
                        }}
                        .metric {{
                            font-size: 24px;
                            font-weight: bold;
                            margin: 10px 0;
                        }}
                        .good {{ color: #28a745; }}
                        .bad {{ color: #dc3545; }}
                        .neutral {{ color: #ffc107; }}
                        .chart-container {{
                            position: relative;
                            height: 400px;
                            margin: 20px 0;
                        }}
                        .tab-content {{
                            padding: 20px;
                            background: #f8f9fa;
                            border-radius: 0 0 10px 10px;
                        }}
                        .nav-tabs {{ border-bottom: 2px solid #dee2e6; }}
                        .nav-tabs .nav-link.active {{
                            border-color: #dee2e6 #dee2e6 #fff;
                            font-weight: bold;
                        }}
                    </style>
                </head>
                <body class="bg-light">
                    <div class="container-fluid py-4">
                        <div class="row mb-4">
                            <div class="col">
                                <h1 class="display-4">Performance Report - {company_name}</h1>
                                <h3 class="text-muted">Algorithm {algo_num}</h3>
                            </div>
                        </div>

                        <!-- Key Metrics Dashboard -->
                        <div class="row mb-4">
                            <div class="col-md-3">
                                <div class="dashboard-card">
                                    <h5>Total Return</h5>
                                    <div class="metric {'good' if stats['Return [%]'] > 0 else 'bad'}">
                                        {stats['Return [%]']:.2f}%
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="dashboard-card">
                                    <h5>Sharpe Ratio</h5>
                                    <div class="metric {'good' if stats['Sharpe Ratio'] > 1 else 'neutral'}">
                                        {stats['Sharpe Ratio']:.2f}
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="dashboard-card">
                                    <h5>Max Drawdown</h5>
                                    <div class="metric bad">
                                        {stats['Max. Drawdown [%]']:.2f}%
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="dashboard-card">
                                    <h5>Win Rate</h5>
                                    <div class="metric {'good' if stats['Win Rate [%]'] > 50 else 'neutral'}">
                                        {stats['Win Rate [%]']:.2f}%
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Navigation Tabs -->
                        <ul class="nav nav-tabs" role="tablist">
                            <li class="nav-item">
                                <a class="nav-link active" data-bs-toggle="tab" href="#overview">Performance Overview</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" data-bs-toggle="tab" href="#trades">Trade Analysis</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" data-bs-toggle="tab" href="#risk">Risk Metrics</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" data-bs-toggle="tab" href="#charts">Interactive Charts</a>
                            </li>
                        </ul>

                        <!-- Tab Content -->
                        <div class="tab-content">
                            <!-- Overview Tab -->
                            <div id="overview" class="tab-pane active">
                                <div class="row">
                                    <div class="col-md-6">
                                        <div class="dashboard-card">
                                            <h4>Performance Metrics</h4>
                                            <table class="table table-striped">
                                                <tbody>
                                                    <tr>
                                                        <td>Annual Return</td>
                                                        <td class="{'good' if stats['Return (Ann.) [%]'] > 0 else 'bad'}">
                                                            {stats['Return (Ann.) [%]']:.2f}%
                                                        </td>
                                                    </tr>
                                                    <tr>
                                                        <td>Volatility (Ann.)</td>
                                                        <td>{stats['Volatility (Ann.) [%]']:.2f}%</td>
                                                    </tr>
                                                    <tr>
                                                        <td>Sortino Ratio</td>
                                                        <td>{stats['Sortino Ratio']:.2f}</td>
                                                    </tr>
                                                    <tr>
                                                        <td>Calmar Ratio</td>
                                                        <td>{stats['Calmar Ratio']:.2f}</td>
                                                    </tr>
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="dashboard-card">
                                            <h4>Equity Curve</h4>
                                            <div class="chart-container">
                                                <img src="equity_curve.png" class="img-fluid" alt="Equity Curve">
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <!-- Trades Tab -->
                            <div id="trades" class="tab-pane fade">
                                <div class="row">
                                    <div class="col-md-6">
                                        <div class="dashboard-card">
                                            <h4>Trade Statistics</h4>
                                            <table class="table table-striped">
                                                <tbody>
                                                    <tr>
                                                        <td>Total Trades</td>
                                                        <td>{stats['# Trades']}</td>
                                                    </tr>
                                                    <tr>
                                                        <td>Best Trade</td>
                                                        <td class="good">{stats['Best Trade [%]']:.2f}%</td>
                                                    </tr>
                                                    <tr>
                                                        <td>Worst Trade</td>
                                                        <td class="bad">{stats['Worst Trade [%]']:.2f}%</td>
                                                    </tr>
                                                    <tr>
                                                        <td>Average Trade</td>
                                                        <td>{stats['Avg. Trade [%]']:.2f}%</td>
                                                    </tr>
                                                    <tr>
                                                        <td>Profit Factor</td>
                                                        <td>{stats['Profit Factor']:.2f}</td>
                                                    </tr>
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="dashboard-card">
                                            <h4>Trade Distribution</h4>
                                            <img src="trade_analysis.png" class="img-fluid" alt="Trade Analysis">
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <!-- Risk Metrics Tab -->
                            <div id="risk" class="tab-pane fade">
                                <div class="row">
                                    <div class="col-md-6">
                                        <div class="dashboard-card">
                                            <h4>Drawdown Analysis</h4>
                                            <img src="drawdown_analysis.png" class="img-fluid" alt="Drawdown Analysis">
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="dashboard-card">
                                            <h4>Monthly Returns</h4>
                                            <img src="monthly_returns.png" class="img-fluid" alt="Monthly Returns">
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <!-- Charts Tab -->
                            <div id="charts" class="tab-pane fade">
                                <div class="dashboard-card">
                                    <h4>Interactive Trading Chart</h4>
                                    <iframe src="interactive_chart.html" width="100%" height="800px" frameborder="0"></iframe>
                                </div>
                            </div>
                        </div>
                    </div>

                    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
                    <script>
                        // Initialize tooltips
                        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
                        var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {{
                            return new bootstrap.Tooltip(tooltipTriggerEl)
                        }})
                    </script>
                </body>
                </html>
                """
                
                # Write HTML report
                with open(report_dir / 'performance_report.html', 'w') as f:
                    f.write(html_content)
                    
                logger.info(f"Generated performance report for {company_name} algorithm {algo_num}")
                
            except Exception as e:
                logger.error(f"Error generating performance report: {e}")
                raise

    def update_html_report(self, report_dir: Path, company_name: str, algo_num: int):
        """Update HTML report to include new visualizations and real-time updates."""
        try:
            # Read existing report
            report_path = report_dir / 'performance_report.html'
            if not report_path.exists():
                logger.warning("Performance report not found, skipping update")
                return
            
            with open(report_path, 'r') as f:
                content = f.read()
            
            # Add real-time updates section
            realtime_content = """
                <div class="dashboard-card">
                    <h4>Real-Time Strategy Performance</h4>
                    <div id="realtime-metrics">
                        <!-- Metrics will be updated via JavaScript -->
                    </div>
                </div>
            """
            
            # Add indicator analysis section
            indicator_content = """
                <div class="dashboard-card">
                    <h4>Indicator Performance Analysis</h4>
                    <div id="indicator-analysis">
                        <!-- Indicator effectiveness metrics -->
                    </div>
                </div>
            """
            
            # Update content
            updated_content = content.replace(
                '</div>\n</div>',
                f'{realtime_content}\n{indicator_content}\n</div>\n</div>'
            )
            
            # Add JavaScript for real-time updates
            js_content = """
                <script>
                    function updateMetrics() {
                        // Fetch latest metrics and update display
                        fetch('metrics.json')
                            .then(response => response.json())
                            .then(data => {
                                document.getElementById('realtime-metrics').innerHTML = 
                                    createMetricsHTML(data);
                            });
                    }

                    function createMetricsHTML(data) {
                        // Create HTML for metrics display
                        let html = '<div class="row">';
                        for (let metric in data) {
                            html += `
                                <div class="col-md-3">
                                    <div class="metric-card">
                                        <h5>${metric}</h5>
                                        <div class="metric ${data[metric].class}">
                                            ${data[metric].value}
                                        </div>
                                    </div>
                                </div>
                            `;
                        }
                        html += '</div>';
                        return html;
                    }

                    // Update metrics every minute
                    setInterval(updateMetrics, 60000);
                    updateMetrics(); // Initial update
                </script>
            """
            
            updated_content = updated_content.replace('</body>', f'{js_content}\n</body>')
            
            # Save updated report
            with open(report_path, 'w') as f:
                f.write(updated_content)
            
            logger.info(f"Updated HTML report for {company_name} algorithm {algo_num}")
            
        except Exception as e:
            logger.error(f"Error updating HTML report: {e}")
            raise
    
    def validate_data_quality(self, df: pd.DataFrame) -> bool:
        """Validate data quality before backtesting."""
        try:
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.any():
                logger.warning(f"Found missing values: {missing_values[missing_values > 0]}")
                
            # Check for zero or negative prices
            invalid_prices = ((df[['Open', 'High', 'Low', 'Close']] <= 0).sum())
            if invalid_prices.any():
                logger.warning(f"Found invalid prices: {invalid_prices[invalid_prices > 0]}")
                
            # Check for price anomalies
            high_low_invalid = (df['High'] < df['Low']).sum()
            if high_low_invalid > 0:
                logger.warning(f"Found {high_low_invalid} cases where High < Low")
                
            # Check for gaps in timestamps
            time_diff = df.index.to_series().diff()
            irregular_intervals = time_diff.value_counts()
            if len(irregular_intervals) > 1:
                logger.warning("Found irregular time intervals in data")
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating data quality: {e}")
            return False

    def optimize_strategy_parameters(self, strategy: Dict, data: pd.DataFrame) -> Dict:
        """Optimize strategy parameters using grid search."""
        try:
            best_params = {}
            best_sharpe = float('-inf')
            
            # Define parameter ranges
            param_grid = {
                'RSI': range(10, 21, 2),  # RSI periods
                'MACD': [(8,17,9), (12,26,9), (10,20,9)],  # Fast, slow, signal
                'stop_loss': [0.01, 0.02, 0.03],
                'take_profit': [0.02, 0.03, 0.04]
            }
            
            total_combinations = np.prod([len(v) for v in param_grid.values()])
            logger.info(f"Testing {total_combinations} parameter combinations")
            
            # Grid search
            for rsi_period in param_grid['RSI']:
                for macd_params in param_grid['MACD']:
                    for sl in param_grid['stop_loss']:
                        for tp in param_grid['take_profit']:
                            # Update strategy parameters
                            test_strategy = strategy.copy()
                            test_strategy['indicators'][0]['params']['length'] = rsi_period
                            test_strategy['indicators'][1]['params'].update({
                                'fast': macd_params[0],
                                'slow': macd_params[1],
                                'signal': macd_params[2]
                            })
                            test_strategy['risk_management'].update({
                                'stop_loss': sl,
                                'take_profit': tp
                            })
                            
                            # Run backtest with parameters
                            stats = self.run_backtest(test_strategy, data)
                            
                            # Update best parameters if better Sharpe ratio
                            if stats['Sharpe Ratio'] > best_sharpe:
                                best_sharpe = stats['Sharpe Ratio']
                                best_params = {
                                    'RSI_period': rsi_period,
                                    'MACD_params': macd_params,
                                    'stop_loss': sl,
                                    'take_profit': tp
                                }
                                
            logger.info(f"Best parameters found: {best_params}")
            return best_params
            
        except Exception as e:
            logger.error(f"Error optimizing strategy parameters: {e}")
            return strategy

    def perform_walk_forward_analysis(self, data: pd.DataFrame, 
                                    train_ratio: float = 0.7) -> Dict:
        """Perform walk-forward analysis to test strategy robustness."""
        try:
            results = []
            window_size = len(data) // 4  # Quarter of the data
            
            for i in range(0, len(data) - window_size, window_size // 2):
                # Split data into training and testing
                train_data = data.iloc[i:i + int(window_size * train_ratio)]
                test_data = data.iloc[i + int(window_size * train_ratio):i + window_size]
                
                # Optimize strategy on training data
                strategy = self.get_default_strategy("WalkForward")
                optimized_params = self.optimize_strategy_parameters(strategy, train_data)
                
                # Test optimized strategy
                test_stats = self.run_backtest(strategy, test_data)
                
                results.append({
                    'window_start': data.index[i],
                    'window_end': data.index[i + window_size],
                    'train_sharpe': test_stats['Sharpe Ratio'],
                    'test_return': test_stats['Return [%]'],
                    'parameters': optimized_params
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Error performing walk-forward analysis: {e}")
            return []

    def calculate_strategy_robustness(self, data: pd.DataFrame, 
                                    num_simulations: int = 100) -> Dict:
        """Calculate strategy robustness through Monte Carlo simulation."""
        try:
            simulation_results = []
            
            for _ in range(num_simulations):
                # Create modified price series with random noise
                noise = np.random.normal(0, 0.001, len(data))
                modified_data = data.copy()
                modified_data['Close'] *= (1 + noise)
                
                # Run backtest with modified data
                stats = self.run_backtest(modified_data)
                simulation_results.append({
                    'return': stats['Return [%]'],
                    'sharpe': stats['Sharpe Ratio'],
                    'max_dd': stats['Max. Drawdown [%]']
                })
            
            # Calculate robustness metrics
            robustness = {
                'return_std': np.std([r['return'] for r in simulation_results]),
                'sharpe_ratio_std': np.std([r['sharpe'] for r in simulation_results]),
                'max_dd_std': np.std([r['max_dd'] for r in simulation_results]),
                'reliability': len([r for r in simulation_results if r['return'] > 0]) / num_simulations
            }
            
            return robustness
            
        except Exception as e:
            logger.error(f"Error calculating strategy robustness: {e}")
            return {}

    def generate_strategy_report(self, stats: Dict, data: pd.DataFrame, 
                               output_dir: Path) -> None:
        """Generate comprehensive strategy performance report."""
        try:
            report = {
                'performance_metrics': {
                    'return': stats['Return [%]'],
                    'sharpe_ratio': stats['Sharpe Ratio'],
                    'max_drawdown': stats['Max. Drawdown [%]'],
                    'win_rate': stats['Win Rate [%]']
                },
                'risk_metrics': {
                    'var_95': self._calculate_var(stats['_trades'], 0.95),
                    'cvar_95': self._calculate_cvar(stats['_trades'], 0.95),
                    'beta': self._calculate_beta(data, stats['_equity_curve'])
                },
                'trade_metrics': {
                    'profit_factor': stats['Profit Factor'],
                    'avg_trade': stats['Avg. Trade [%]'],
                    'avg_duration': self._calculate_avg_trade_duration(stats['_trades'])
                }
            }
            
            # Add robustness analysis
            robustness = self.calculate_strategy_robustness(data)
            report['robustness'] = robustness
            
            # Save report
            with open(output_dir / 'strategy_report.json', 'w') as f:
                json.dump(report, f, indent=4)
                
            logger.info(f"Strategy report generated: {output_dir}/strategy_report.json")
            
        except Exception as e:
            logger.error(f"Error generating strategy report: {e}")

# Add these helper methods to calculate risk metrics

    def validate_trades_data(self, trades: List) -> bool:
        """Validate trades data before analysis."""
        try:
            # Check if trades is a list or DataFrame
            if isinstance(trades, pd.DataFrame):
                # For DataFrame format
                if trades.empty:
                    logger.warning("Empty trades DataFrame")
                    return False
                    
                required_columns = ['Entry Time', 'Exit Time', 'Entry Price', 
                                  'Exit Price', 'PnL', 'Return']
                                  
                if not all(col in trades.columns for col in required_columns):
                    logger.warning("Missing required columns in trades DataFrame")
                    return False
                    
                return True
                
            elif isinstance(trades, list):
                # For list format
                if not trades:
                    logger.warning("Empty trades list")
                    return False
                    
                # Check for required attributes in trade objects
                required_attrs = ['entry_time', 'exit_time', 'entry_price', 
                                'exit_price', 'pl', 'pl_pct']
                                
                for trade in trades:
                    if not all(hasattr(trade, attr) for attr in required_attrs):
                        logger.warning("Trade objects missing required attributes")
                        return False
                        
                return True
                
            else:
                logger.warning(f"Invalid trades data type: {type(trades)}")
                return False
                
        except Exception as e:
            logger.error(f"Error validating trades data: {e}")
            return False

    def _convert_trades_to_dataframe(self, trades: List) -> pd.DataFrame:
        """Convert trades list to DataFrame format."""
        try:
            if isinstance(trades, pd.DataFrame):
                return trades
                
            if not trades:
                return pd.DataFrame(columns=[
                    'Entry Time', 'Exit Time', 'Entry Price', 'Exit Price',
                    'Size', 'PnL', 'Return', 'Duration'
                ])
                
            trades_data = [{
                'Entry Time': t.entry_time,
                'Exit Time': t.exit_time,
                'Entry Price': t.entry_price,
                'Exit Price': t.exit_price,
                'Size': getattr(t, 'size', 0),
                'PnL': t.pl,
                'Return': t.pl_pct,
                'Duration': (t.exit_time - t.entry_time).total_seconds() / 3600
            } for t in trades]
            
            return pd.DataFrame(trades_data)
            
        except Exception as e:
            logger.error(f"Error converting trades to DataFrame: {e}")
            return pd.DataFrame()

    def analyze_trades(self, trades: List) -> Dict:
        """Analyze trades and generate performance metrics."""
        try:
            # Convert trades to DataFrame for analysis
            trades_df = self._convert_trades_to_dataframe(trades)
            
            if trades_df.empty:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'profit_factor': 0.0,
                    'avg_trade_return': 0.0,
                    'avg_trade_duration': 0.0
                }
            
            # Calculate metrics
            winning_trades = trades_df[trades_df['PnL'] > 0]
            losing_trades = trades_df[trades_df['PnL'] < 0]
            
            metrics = {
                'total_trades': len(trades_df),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0.0,
                'avg_win': winning_trades['PnL'].mean() if not winning_trades.empty else 0.0,
                'avg_loss': losing_trades['PnL'].mean() if not losing_trades.empty else 0.0,
                'profit_factor': (abs(winning_trades['PnL'].sum()) / 
                                abs(losing_trades['PnL'].sum())
                                if not losing_trades.empty and losing_trades['PnL'].sum() != 0 
                                else 0.0),
                'avg_trade_return': trades_df['Return'].mean(),
                'avg_trade_duration': trades_df['Duration'].mean()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing trades: {e}")
            return {}

    def update_basic_results(self, stats: Dict, report_dir: Path) -> None:
        """Update basic results with additional metrics."""
        try:
            # Load existing results if any
            results_file = report_dir / 'basic_results.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
            else:
                results = {}
            
            # Update with new metrics
            trades = stats.get('_trades', [])
            trade_metrics = self.analyze_trades(trades)
            
            results.update({
                'total_trades': trade_metrics.get('total_trades', 0),
                'total_return': stats.get('Return [%]', 0.0),
                'sharpe_ratio': stats.get('Sharpe Ratio', 0.0),
                'max_drawdown': stats.get('Max. Drawdown [%]', 0.0),
                'win_rate': trade_metrics.get('win_rate', 0.0) * 100,
                'profit_factor': trade_metrics.get('profit_factor', 0.0),
                'avg_trade_return': trade_metrics.get('avg_trade_return', 0.0)
            })
            
            # Save updated results
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
                
            logger.info("Basic results updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating basic results: {e}")


    def _calculate_var(self, trades: List, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        try:
            returns = [t.pl_pct for t in trades]
            return np.percentile(returns, (1 - confidence) * 100)
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0

    def _calculate_cvar(self, trades: List, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        try:
            returns = [t.pl_pct for t in trades]
            var = self._calculate_var(trades, confidence)
            return np.mean([r for r in returns if r <= var])
        except Exception as e:
            logger.error(f"Error calculating CVaR: {e}")
            return 0.0

    def _calculate_beta(self, market_data: pd.DataFrame, 
                       strategy_equity: pd.DataFrame) -> float:
        """Calculate strategy beta relative to market."""
        try:
            market_returns = market_data['Close'].pct_change().dropna()
            strategy_returns = strategy_equity['Equity'].pct_change().dropna()
            
            # Align dates
            aligned_data = pd.concat([market_returns, strategy_returns], axis=1).dropna()
            
            # Calculate beta
            covariance = aligned_data.cov().iloc[0,1]
            market_variance = aligned_data.iloc[:,0].var()
            
            return covariance / market_variance
            
        except Exception as e:
            logger.error(f"Error calculating beta: {e}")
            return 0.0

    def _calculate_avg_trade_duration(self, trades: List) -> float:
        """Calculate average trade duration in hours."""
        try:
            durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 
                        for t in trades]
            return np.mean(durations)
        except Exception as e:
            logger.error(f"Error calculating average trade duration: {e}")
            return 0.0
        
class SimplifiedTradingStrategy(Strategy):
    """Simplified trading strategy to ensure basic functionality"""
    
    def init(self):
        """Initialize strategy with basic indicators"""
        # Price data
        self.close = self.data.Close
        self.high = self.data.High
        self.low = self.data.Low
        self.volume = self.data.Volume
        
        # Calculate RSI
        self.rsi = self.I(talib.RSI, self.close, timeperiod=14)
        
        # Calculate EMAs for trend
        self.ema_20 = self.I(talib.EMA, self.close, timeperiod=20)
        self.ema_50 = self.I(talib.EMA, self.close, timeperiod=50)
        
        # Plot indicators
        self.I(lambda: self.ema_20, overlay=True, name='EMA20')
        self.I(lambda: self.ema_50, overlay=True, name='EMA50')
        self.I(lambda: self.rsi, panel='RSI', name='RSI')
        
        # Trade management
        self.last_buy_price = 0
        self.stop_loss = None
        self.take_profit = None

    def next(self):
        """Define trading logic"""
        if len(self.data) < 50:  # Wait for indicators to warm up
            return
            
        price = self.close[-1]
        rsi = self.rsi[-1]
        
        # Entry Logic - Oversold RSI with trend support
        if not self.position:
            if rsi < 30 and self.ema_20[-1] > self.ema_50[-1]:
                # Risk calculation (1% risk per trade)
                stop_price = price * 0.98  # 2% stop loss
                target_price = price * 1.04  # 4% take profit
                risk_amount = price - stop_price
                position_size = (self.equity * 0.01) / risk_amount
                
                self.buy(size=position_size)
                self.last_buy_price = price
                self.stop_loss = stop_price
                self.take_profit = target_price
                
                print(f"BUY: Price={price:.2f}, RSI={rsi:.2f}, Size={position_size:.2f}")
                
        # Exit Logic
        elif self.position:
            # Exit conditions:
            # 1. RSI overbought
            # 2. Stop loss hit
            # 3. Take profit hit
            if (rsi > 70 or 
                price <= self.stop_loss or 
                price >= self.take_profit):
                
                self.position.close()
                print(f"SELL: Price={price:.2f}, RSI={rsi:.2f}")
                self.stop_loss = None
                self.take_profit = None

def run_backtest(data: pd.DataFrame) -> dict:
    """Run backtest with the simplified strategy"""
    bt = Backtest(data, 
                  SimplifiedTradingStrategy,
                  cash=100000,
                  commission=0.002,
                  exclusive_orders=True)
    
    # Run backtest with optimization
    stats = bt.run()
    
    # Plot results
    bt.plot(show_indicators=True)
    
    return stats

# Example usage:
def test_strategy(company_name: str, start_date: str, end_date: str):
    """Test the strategy with given parameters"""
    try:
        # Initialize agent
        agent = BacktestingAgent()
        
        # Load data
        data = agent.load_historical_data(company_name, start_date, end_date)
        
        # Run backtest
        print(f"Running backtest for {company_name} from {start_date} to {end_date}")
        stats = run_backtest(data)
        
        # Print results
        print("\nBacktest Results:")
        print("-" * 50)
        print(f"Total Trades: {stats['# Trades']}")
        print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
        print(f"Total Return: {stats['Return [%]']:.2f}%")
        print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
        
        return stats
        
    except Exception as e:
        print(f"Error running backtest: {str(e)}")
        raise      


class TradeAnalyzer:
    """Analyze trade performance and generate insights with advanced metrics."""
    
    def __init__(self, trades: List, data: pd.DataFrame):
        """Initialize trade analyzer with trade data and market data."""
        try:
            # Convert trades to DataFrame
            self.trades_df = pd.DataFrame([{
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'size': t.size,
                'pnl': t.pl,
                'return': t.pl_pct,
                'entry_idx': t.entry_idx,
                'exit_idx': t.exit_idx
            } for t in trades])
            
            self.market_data = data
            
            # Add derived metrics
            if len(self.trades_df) > 0:
                self.trades_df['duration'] = (
                    self.trades_df['exit_time'] - self.trades_df['entry_time']
                )
                self.trades_df['trade_direction'] = np.where(
                    self.trades_df['size'] > 0, 'long', 'short'
                )
                self.trades_df['hour'] = self.trades_df['entry_time'].dt.hour
                self.trades_df['day_of_week'] = self.trades_df['entry_time'].dt.day_name()
                self.trades_df['is_win'] = self.trades_df['pnl'] > 0
                
                # Calculate trade-specific metrics
                self.calculate_trade_metrics()
            
            logger.info(f"Initialized TradeAnalyzer with {len(trades)} trades")
            
        except Exception as e:
            logger.error(f"Error initializing TradeAnalyzer: {e}")
            raise

    def analyze_trade_patterns(self) -> Dict:
        """Analyze patterns in successful and unsuccessful trades."""
        try:
            if len(self.trades_df) == 0:
                return {}
                
            patterns = {
                'time_based': self.analyze_time_patterns(),
                'price_patterns': self.analyze_price_patterns(),
                'market_conditions': self.analyze_market_conditions(),
                'sequential_patterns': self.analyze_sequential_patterns()
            }
            
            # Generate trade pattern report
            report = {
                'summary': patterns,
                'recommendations': self.generate_recommendations(patterns),
                'visualizations': self.generate_pattern_visualizations()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error analyzing trade patterns: {e}")
            return {}

    def analyze_price_patterns(self) -> Dict:
        """Analyze price action patterns around trades."""
        try:
            patterns = {}
            
            # Analyze price momentum before entry
            pre_trade_returns = []
            for idx, trade in self.trades_df.iterrows():
                if trade.entry_idx >= 5:  # Need at least 5 bars before entry
                    pre_trade_data = self.market_data.iloc[
                        trade.entry_idx-5:trade.entry_idx
                    ]
                    pre_trade_returns.append(
                        (pre_trade_data['Close'].pct_change().mean(), trade.is_win)
                    )
            
            if pre_trade_returns:
                returns, outcomes = zip(*pre_trade_returns)
                patterns['momentum_correlation'] = stats.pointbiserialr(
                    returns, outcomes
                ).correlation
            
            # Analyze volatility impact
            self.trades_df['pre_trade_volatility'] = self.trades_df.apply(
                lambda x: self.market_data['Close'].iloc[
                    max(0, x.entry_idx-20):x.entry_idx
                ].std() if x.entry_idx >= 20 else np.nan, 
                axis=1
            )
            
            patterns['volatility_impact'] = {
                'high_vol_win_rate': self.trades_df[
                    self.trades_df['pre_trade_volatility'] > 
                    self.trades_df['pre_trade_volatility'].median()
                ]['is_win'].mean(),
                'low_vol_win_rate': self.trades_df[
                    self.trades_df['pre_trade_volatility'] <= 
                    self.trades_df['pre_trade_volatility'].median()
                ]['is_win'].mean()
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing price patterns: {e}")
            return {}

    def analyze_market_conditions(self) -> Dict:
        """Analyze performance under different market conditions."""
        try:
            conditions = {}
            
            # Define market conditions based on trend
            self.market_data['trend'] = np.where(
                self.market_data['Close'] > self.market_data['Close'].rolling(50).mean(),
                'uptrend',
                'downtrend'
            )
            
            # Calculate volatility regimes
            self.market_data['volatility'] = self.market_data['Close'].pct_change().rolling(20).std()
            volatility_threshold = self.market_data['volatility'].median()
            self.market_data['volatility_regime'] = np.where(
                self.market_data['volatility'] > volatility_threshold,
                'high',
                'low'
            )
            
            # Analyze performance in different conditions
            for condition in ['trend', 'volatility_regime']:
                performance = {}
                for regime in self.market_data[condition].unique():
                    trades_in_regime = self.trades_df[
                        self.trades_df['entry_idx'].apply(
                            lambda x: self.market_data[condition].iloc[x] == regime
                        )
                    ]
                    if len(trades_in_regime) > 0:
                        performance[regime] = {
                            'trade_count': len(trades_in_regime),
                            'win_rate': trades_in_regime['is_win'].mean(),
                            'avg_return': trades_in_regime['return'].mean(),
                            'profit_factor': abs(
                                trades_in_regime[trades_in_regime['pnl'] > 0]['pnl'].sum() /
                                trades_in_regime[trades_in_regime['pnl'] < 0]['pnl'].sum()
                            ) if len(trades_in_regime[trades_in_regime['pnl'] < 0]) > 0 else np.inf
                        }
                conditions[condition] = performance
            
            return conditions
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return {}

    def analyze_sequential_patterns(self) -> Dict:
        """Analyze patterns in trade sequences."""
        try:
            patterns = {}
            
            # Analyze winning/losing streaks
            self.trades_df['streak'] = (
                self.trades_df['is_win'].ne(self.trades_df['is_win'].shift())
            ).cumsum()
            
            streak_stats = self.trades_df.groupby('streak')['is_win'].agg(['all', 'count'])
            max_win_streak = streak_stats[streak_stats['all']]['count'].max()
            max_loss_streak = streak_stats[~streak_stats['all']]['count'].max()
            
            patterns['streaks'] = {
                'max_win_streak': max_win_streak,
                'max_loss_streak': max_loss_streak,
                'avg_win_streak': streak_stats[streak_stats['all']]['count'].mean(),
                'avg_loss_streak': streak_stats[~streak_stats['all']]['count'].mean()
            }
            
            # Analyze trade clustering
            self.trades_df['time_between_trades'] = (
                self.trades_df['entry_time'] - self.trades_df['exit_time'].shift()
            )
            
            patterns['clustering'] = {
                'avg_time_between_trades': self.trades_df['time_between_trades'].mean(),
                'clustered_trades_win_rate': self.trades_df[
                    self.trades_df['time_between_trades'] < 
                    self.trades_df['time_between_trades'].median()
                ]['is_win'].mean()
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing sequential patterns: {e}")
            return {}

    def calculate_trade_metrics(self) -> Dict:
        """Calculate comprehensive trade metrics."""
        try:
            if len(self.trades_df) == 0:
                return {}
            
            winning_trades = self.trades_df[self.trades_df['pnl'] > 0]
            losing_trades = self.trades_df[self.trades_df['pnl'] < 0]
            
            metrics = TradeMetrics(
                total_trades=len(self.trades_df),
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                win_rate=len(winning_trades) / len(self.trades_df),
                avg_win=winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
                avg_loss=losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
                profit_factor=abs(
                    winning_trades['pnl'].sum() / losing_trades['pnl'].sum()
                ) if len(losing_trades) > 0 else np.inf,
                max_drawdown=self.calculate_max_drawdown(),
                sharpe_ratio=self.calculate_sharpe_ratio(),
                sortino_ratio=self.calculate_sortino_ratio(),
                avg_trade_duration=self.trades_df['duration'].mean().total_seconds() / 3600,
                risk_reward_ratio=abs(
                    winning_trades['pnl'].mean() / losing_trades['pnl'].mean()
                ) if len(losing_trades) > 0 else np.inf
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating trade metrics: {e}")
            return {}

    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from trade equity curve."""
        try:
            equity_curve = self.trades_df['pnl'].cumsum()
            running_max = equity_curve.expanding().max()
            drawdowns = equity_curve - running_max
            return abs(drawdowns.min())
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0

    def calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio of trades."""
        try:
            returns = self.trades_df['return']
            excess_returns = returns - 0.02/252  # Assuming 2% risk-free rate
            return np.sqrt(252) * excess_returns.mean() / excess_returns.std() if len(returns) > 1 else 0
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio of trades."""
        try:
            returns = self.trades_df['return']
            excess_returns = returns - 0.02/252  # Assuming 2% risk-free rate
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std()
            return np.sqrt(252) * excess_returns.mean() / downside_std if len(downside_returns) > 1 else 0
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0

    def analyze_time_patterns(self) -> Dict:
        """Analyze trading performance by time of day and day of week."""
        try:
            time_patterns = {}
            
            # Hour of day analysis
            hourly_stats = self.trades_df.groupby('hour').agg({
                'pnl': ['count', 'sum', 'mean'],
                'is_win': 'mean'
            })
            hourly_stats.columns = ['trade_count', 'total_pnl', 'avg_pnl', 'win_rate']
            
            time_patterns['hourly'] = hourly_stats.to_dict()
            
            # Day of week analysis
            daily_stats = self.trades_df.groupby('day_of_week').agg({
                'pnl': ['count', 'sum', 'mean'],
                'is_win': 'mean'
            })
            daily_stats.columns = ['trade_count', 'total_pnl', 'avg_pnl', 'win_rate']
            
            time_patterns['daily'] = daily_stats.to_dict()
            
            # Find best and worst trading times
            best_hour = hourly_stats['win_rate'].idxmax()
            worst_hour = hourly_stats['win_rate'].idxmin()
            
            time_patterns['optimal_times'] = {
                'best_hour': {
                    'hour': best_hour,
                    'win_rate': hourly_stats.loc[best_hour, 'win_rate'],
                    'avg_pnl': hourly_stats.loc[best_hour, 'avg_pnl']
                },
                'worst_hour': {
                    'hour': worst_hour,
                    'win_rate': hourly_stats.loc[worst_hour, 'win_rate'],
                    'avg_pnl': hourly_stats.loc[worst_hour, 'avg_pnl']
                }
            }
            
            return time_patterns
            
        except Exception as e:
            logger.error(f"Error analyzing time patterns: {e}")
            return {}
        
    def analyze_indicator_effectiveness(self) -> Dict:
        """Analyze effectiveness of different indicators."""
        try:
            indicators = {}
            
            # Common technical indicators
            for indicator, values in self.market_data.items():
                if indicator in ['RSI', 'MACD', 'MACD_Signal', 'SMA_20', 'EMA_50']:
                    effectiveness = self.analyze_single_indicator(indicator, values)
                    if effectiveness:
                        indicators[indicator] = effectiveness
            
            # Analyze indicator combinations
            if 'MACD' in self.market_data and 'MACD_Signal' in self.market_data:
                macd_effectiveness = self.analyze_macd_crossover()
                indicators['MACD_Crossover'] = macd_effectiveness
            
            if 'SMA_20' in self.market_data and 'EMA_50' in self.market_data:
                ma_effectiveness = self.analyze_ma_crossover()
                indicators['MA_Crossover'] = ma_effectiveness
            
            # Generate effectiveness report
            report = {
                'indicator_metrics': indicators,
                'combined_signals': self.analyze_combined_signals(),
                'recommendations': self.generate_indicator_recommendations(indicators)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error analyzing indicator effectiveness: {e}")
            return {}

    def analyze_single_indicator(self, indicator: str, values: pd.Series) -> Dict:
        """Analyze effectiveness of a single indicator."""
        try:
            effectiveness = {}
            
            # Analyze trades based on indicator values
            for idx, trade in self.trades_df.iterrows():
                if trade.entry_idx >= 0:
                    indicator_value = values.iloc[trade.entry_idx]
                    trade_outcome = trade.pnl > 0
                    
                    if indicator == 'RSI':
                        if indicator_value < 30:  # Oversold
                            if 'oversold' not in effectiveness:
                                effectiveness['oversold'] = {'wins': 0, 'losses': 0}
                            if trade_outcome:
                                effectiveness['oversold']['wins'] += 1
                            else:
                                effectiveness['oversold']['losses'] += 1
                        elif indicator_value > 70:  # Overbought
                            if 'overbought' not in effectiveness:
                                effectiveness['overbought'] = {'wins': 0, 'losses': 0}
                            if trade_outcome:
                                effectiveness['overbought']['wins'] += 1
                            else:
                                effectiveness['overbought']['losses'] += 1
                    
                    # Add signal to effectiveness tracking
                    signal_key = f"signal_{len(effectiveness)}"
                    if signal_key not in effectiveness:
                        effectiveness[signal_key] = {
                            'value': indicator_value,
                            'outcome': trade_outcome,
                            'pnl': trade.pnl
                        }
            
            # Calculate success rates
            for key in effectiveness:
                if isinstance(effectiveness[key], dict) and 'wins' in effectiveness[key]:
                    total = effectiveness[key]['wins'] + effectiveness[key]['losses']
                    if total > 0:
                        effectiveness[key]['success_rate'] = (
                            effectiveness[key]['wins'] / total
                        )
            
            return effectiveness
            
        except Exception as e:
            logger.error(f"Error analyzing single indicator: {e}")
            return {}

    def analyze_macd_crossover(self) -> Dict:
        """Analyze effectiveness of MACD crossover signals."""
        try:
            crossover_stats = {
                'bullish': {'wins': 0, 'losses': 0, 'avg_pnl': []},
                'bearish': {'wins': 0, 'losses': 0, 'avg_pnl': []}
            }
            
            for idx, trade in self.trades_df.iterrows():
                if trade.entry_idx >= 1:  # Need at least one prior bar
                    macd_prev = self.market_data['MACD'].iloc[trade.entry_idx - 1]
                    signal_prev = self.market_data['MACD_Signal'].iloc[trade.entry_idx - 1]
                    macd_curr = self.market_data['MACD'].iloc[trade.entry_idx]
                    signal_curr = self.market_data['MACD_Signal'].iloc[trade.entry_idx]
                    
                    # Detect crossovers
                    if macd_prev <= signal_prev and macd_curr > signal_curr:
                        key = 'bullish'
                    elif macd_prev >= signal_prev and macd_curr < signal_curr:
                        key = 'bearish'
                    else:
                        continue
                    
                    # Record outcome
                    if trade.pnl > 0:
                        crossover_stats[key]['wins'] += 1
                    else:
                        crossover_stats[key]['losses'] += 1
                    crossover_stats[key]['avg_pnl'].append(trade.pnl)
            
            # Calculate statistics
            for key in crossover_stats:
                total = crossover_stats[key]['wins'] + crossover_stats[key]['losses']
                if total > 0:
                    crossover_stats[key]['success_rate'] = (
                        crossover_stats[key]['wins'] / total
                    )
                    crossover_stats[key]['avg_pnl'] = np.mean(
                        crossover_stats[key]['avg_pnl']
                    )
            
            return crossover_stats
            
        except Exception as e:
            logger.error(f"Error analyzing MACD crossover: {e}")
            return {}

    def analyze_ma_crossover(self) -> Dict:
        """Analyze effectiveness of moving average crossover signals."""
        try:
            crossover_stats = {
                'golden_cross': {'wins': 0, 'losses': 0, 'avg_pnl': []},
                'death_cross': {'wins': 0, 'losses': 0, 'avg_pnl': []}
            }
            
            for idx, trade in self.trades_df.iterrows():
                if trade.entry_idx >= 1:
                    sma_prev = self.market_data['SMA_20'].iloc[trade.entry_idx - 1]
                    ema_prev = self.market_data['EMA_50'].iloc[trade.entry_idx - 1]
                    sma_curr = self.market_data['SMA_20'].iloc[trade.entry_idx]
                    ema_curr = self.market_data['EMA_50'].iloc[trade.entry_idx]
                    
                    # Detect crossovers
                    if sma_prev <= ema_prev and sma_curr > ema_curr:
                        key = 'golden_cross'
                    elif sma_prev >= ema_prev and sma_curr < ema_curr:
                        key = 'death_cross'
                    else:
                        continue
                    
                    # Record outcome
                    if trade.pnl > 0:
                        crossover_stats[key]['wins'] += 1
                    else:
                        crossover_stats[key]['losses'] += 1
                    crossover_stats[key]['avg_pnl'].append(trade.pnl)
            
            # Calculate statistics
            for key in crossover_stats:
                total = crossover_stats[key]['wins'] + crossover_stats[key]['losses']
                if total > 0:
                    crossover_stats[key]['success_rate'] = (
                        crossover_stats[key]['wins'] / total
                    )
                    crossover_stats[key]['avg_pnl'] = np.mean(
                        crossover_stats[key]['avg_pnl']
                    )
            
            return crossover_stats
            
        except Exception as e:
            logger.error(f"Error analyzing MA crossover: {e}")
            return {}

    def analyze_combined_signals(self) -> Dict:
        """Analyze effectiveness of combined indicator signals."""
        try:
            combined_stats = {
                'rsi_macd': {'wins': 0, 'losses': 0, 'avg_pnl': []},
                'rsi_ma': {'wins': 0, 'losses': 0, 'avg_pnl': []},
                'macd_ma': {'wins': 0, 'losses': 0, 'avg_pnl': []}
            }
            
            for idx, trade in self.trades_df.iterrows():
                if trade.entry_idx >= 1:
                    # Check RSI + MACD
                    rsi = self.market_data['RSI'].iloc[trade.entry_idx]
                    macd_cross = (
                        self.market_data['MACD'].iloc[trade.entry_idx] >
                        self.market_data['MACD_Signal'].iloc[trade.entry_idx]
                    )
                    
                    if rsi < 30 and macd_cross:
                        key = 'rsi_macd'
                        # Record outcome
                        if trade.pnl > 0:
                            combined_stats[key]['wins'] += 1
                        else:
                            combined_stats[key]['losses'] += 1
                        combined_stats[key]['avg_pnl'].append(trade.pnl)
                    
                    # Check RSI + MA Cross
                    ma_cross = (
                        self.market_data['SMA_20'].iloc[trade.entry_idx] >
                        self.market_data['EMA_50'].iloc[trade.entry_idx]
                    )
                    
                    if rsi < 30 and ma_cross:
                        key = 'rsi_ma'
                        if trade.pnl > 0:
                            combined_stats[key]['wins'] += 1
                        else:
                            combined_stats[key]['losses'] += 1
                        combined_stats[key]['avg_pnl'].append(trade.pnl)
                    
                    # Check MACD + MA Cross
                    if macd_cross and ma_cross:
                        key = 'macd_ma'
                        if trade.pnl > 0:
                            combined_stats[key]['wins'] += 1
                        else:
                            combined_stats[key]['losses'] += 1
                        combined_stats[key]['avg_pnl'].append(trade.pnl)
            
            # Calculate statistics
            for key in combined_stats:
                total = combined_stats[key]['wins'] + combined_stats[key]['losses']
                if total > 0:
                    combined_stats[key]['success_rate'] = (
                        combined_stats[key]['wins'] / total
                    )
                    combined_stats[key]['avg_pnl'] = np.mean(
                        combined_stats[key]['avg_pnl']
                    )
            
            return combined_stats
            
        except Exception as e:
            logger.error(f"Error analyzing combined signals: {e}")
            return {}

    def generate_indicator_recommendations(self, indicators: Dict) -> List[str]:
        """Generate recommendations based on indicator analysis."""
        try:
            recommendations = []
            
            # Analyze individual indicators
            for indicator, stats in indicators.items():
                if 'success_rate' in stats:
                    if stats['success_rate'] > 0.6:
                        recommendations.append(
                            f"{indicator} shows strong predictive power with "
                            f"{stats['success_rate']:.1%} success rate"
                        )
                    elif stats['success_rate'] < 0.4:
                        recommendations.append(
                            f"Consider reviewing {indicator} usage as it shows "
                            f"weak performance with {stats['success_rate']:.1%} success rate"
                        )
            
            # Analyze combined signals
            combined_stats = self.analyze_combined_signals()
            for combo, stats in combined_stats.items():
                if 'success_rate' in stats and stats['success_rate'] > 0.65:
                    recommendations.append(
                        f"Combined signal {combo} shows excellent performance with "
                        f"{stats['success_rate']:.1%} success rate"
                    )
            
            # Add general recommendations
            recommendations.append(
                "Consider using multiple confirming indicators before entry"
            )
            recommendations.append(
                "Monitor indicator effectiveness regularly and adjust parameters if needed"
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating indicator recommendations: {e}")
            return []

    def generate_pattern_visualizations(self) -> Dict:
        """Generate visualizations for trade patterns."""
        try:
            visualizations = {}
            
            # Time-based performance heatmap
            fig_time = self.create_time_heatmap()
            visualizations['time_performance'] = fig_time
            
            # Trade distribution by market condition
            fig_conditions = self.create_market_condition_chart()
            visualizations['market_conditions'] = fig_conditions
            
            # Indicator performance comparison
            fig_indicators = self.create_indicator_comparison()
            visualizations['indicator_performance'] = fig_indicators
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Error generating pattern visualizations: {e}")
            return {}

    def create_time_heatmap(self) -> go.Figure:
        """Create time-based performance heatmap."""
        try:
            # Prepare data
            hourly_stats = self.trades_df.pivot_table(
                values='pnl',
                index='hour',
                columns=self.trades_df['entry_time'].dt.dayofweek,
                aggfunc='mean'
            )
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=hourly_stats.values,
                x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
                y=hourly_stats.index,
                colorscale='RdYlGn',
                colorbar_title='Average PnL'
            ))
            
            fig.update_layout(
                title='Trading Performance by Time',
                xaxis_title='Day of Week',
                yaxis_title='Hour of Day'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating time heatmap: {e}")
            return go.Figure()

    def create_market_condition_chart(self) -> go.Figure:
        """Create chart showing performance in different market conditions."""
        try:
            # Analyze market conditions
            conditions = self.analyze_market_conditions()
            
            # Prepare data for visualization
            fig = go.Figure()
            
            for condition_type, stats in conditions.items():
                win_rates = []
                labels = []
                
                for condition, metrics in stats.items():
                    win_rates.append(metrics['win_rate'])
                    labels.append(condition)
                
                fig.add_trace(go.Bar(
                    name=condition_type,
                    x=labels,
                    y=win_rates,
                    text=[f"{x:.1%}" for x in win_rates]
                ))
            
            fig.update_layout(
                title='Win Rate by Market Condition',
                barmode='group',
                yaxis_tickformat=',.0%'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating market condition chart: {e}")
            return go.Figure()

    def create_indicator_comparison(self) -> go.Figure:
        """Create comparison chart of indicator performance."""
        try:
            # Analyze indicator effectiveness
            indicators = self.analyze_indicator_effectiveness()
            
            # Prepare data for visualization
            fig = make_subplots(rows=2, cols=1,
                              subplot_titles=('Success Rate', 'Average PnL'))
            
            success_rates = []
            avg_pnls = []
            names = []
            
            for indicator, stats in indicators['indicator_metrics'].items():
                if isinstance(stats, dict) and 'success_rate' in stats:
                    success_rates.append(stats['success_rate'])
                    avg_pnls.append(stats.get('avg_pnl', 0))
                    names.append(indicator)
            
            fig.add_trace(
                go.Bar(name='Success Rate',
                      x=names,
                      y=success_rates,
                      text=[f"{x:.1%}" for x in success_rates]),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(name='Average PnL',
                      x=names,
                      y=avg_pnls,
                      text=[f"${x:.2f}" for x in avg_pnls]),
                row=2, col=1
            )
            
            fig.update_layout(
                title='Indicator Performance Comparison',
                showlegend=False,
                height=800
            )
            
            fig.update_yaxes(title_text='Success Rate', tickformat=',.0%', row=1, col=1)
            fig.update_yaxes(title_text='Average PnL ($)', row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating indicator comparison chart: {e}")
            return go.Figure()

    def generate_recommendations(self, patterns: Dict) -> List[str]:
        """Generate trading recommendations based on pattern analysis."""
        try:
            recommendations = []
            
            # Time-based recommendations
            time_patterns = patterns.get('time_based', {})
            if time_patterns:
                best_hour = time_patterns.get('optimal_times', {}).get('best_hour', {})
                if best_hour:
                    recommendations.append(
                        f"Consider focusing trading around {best_hour['hour']}:00 "
                        f"which shows {best_hour['win_rate']:.1%} win rate"
                    )
                
                worst_hour = time_patterns.get('optimal_times', {}).get('worst_hour', {})
                if worst_hour:
                    recommendations.append(
                        f"Avoid trading around {worst_hour['hour']}:00 "
                        f"which shows poor performance"
                    )
            
            # Market condition recommendations
            market_conditions = patterns.get('market_conditions', {})
            for condition_type, stats in market_conditions.items():
                best_condition = max(stats.items(), key=lambda x: x[1]['win_rate'])
                recommendations.append(
                    f"Strategy performs best in {condition_type} {best_condition[0]} "
                    f"with {best_condition[1]['win_rate']:.1%} win rate"
                )
            
            # Sequential pattern recommendations
            sequential = patterns.get('sequential_patterns', {})
            streaks = sequential.get('streaks', {})
            if streaks:
                if streaks.get('max_loss_streak', 0) > 3:
                    recommendations.append(
                        "Consider implementing stronger risk management after "
                        "3 consecutive losses"
                    )
                if streaks.get('max_win_streak', 0) > 3:
                    recommendations.append(
                        "Consider scaling position size during winning streaks "
                        "to maximize performance"
                    )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []

    def export_analysis(self, output_dir: Path) -> None:
        """Export analysis results to files."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Export trade metrics
            metrics = self.calculate_trade_metrics()
            metrics_df = pd.DataFrame([{
                'Metric': key,
                'Value': value
            } for key, value in metrics.__dict__.items()])
            metrics_df.to_csv(output_dir / 'trade_metrics.csv', index=False)
            
            # Export time analysis
            time_patterns = self.analyze_time_patterns()
            if 'hourly' in time_patterns:
                pd.DataFrame(time_patterns['hourly']).to_csv(
                    output_dir / 'hourly_analysis.csv'
                )
            if 'daily' in time_patterns:
                pd.DataFrame(time_patterns['daily']).to_csv(
                    output_dir / 'daily_analysis.csv'
                )
            
            # Export indicator analysis
            indicator_analysis = self.analyze_indicator_effectiveness()
            with open(output_dir / 'indicator_analysis.json', 'w') as f:
                json.dump(indicator_analysis, f, indent=4)
            
            # Export visualizations
            pattern_viz = self.generate_pattern_visualizations()
            for name, fig in pattern_viz.items():
                fig.write_html(str(output_dir / f'{name}.html'))
            
            # Generate comprehensive report
            self.generate_analysis_report(output_dir)
            
            logger.info(f"Analysis results exported to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error exporting analysis: {e}")

    def generate_analysis_report(self, output_dir: Path) -> None:
        """Generate comprehensive HTML analysis report."""
        try:
            # Calculate all metrics and patterns
            metrics = self.calculate_trade_metrics()
            patterns = self.analyze_trade_patterns()
            indicator_analysis = self.analyze_indicator_effectiveness()
            time_patterns = self.analyze_time_patterns()
            
            # Generate recommendations
            recommendations = self.generate_recommendations(patterns)
            
            # Create HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Trading Strategy Analysis Report</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .metric-card {{
                        padding: 20px;
                        margin: 10px;
                        border-radius: 10px;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    }}
                    .good {{ color: green; }}
                    .bad {{ color: red; }}
                    .neutral {{ color: orange; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1 class="mb-4">Trading Strategy Analysis Report</h1>
                    
                    <!-- Performance Metrics -->
                    <div class="card mb-4">
                        <div class="card-header">
                            <h2>Performance Metrics</h2>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-3">
                                    <div class="metric-card">
                                        <h5>Win Rate</h5>
                                        <div class="metric {'good' if metrics.win_rate > 0.5 else 'bad'}">
                                            {metrics.win_rate:.1%}
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="metric-card">
                                        <h5>Profit Factor</h5>
                                        <div class="metric {'good' if metrics.profit_factor > 1.5 else 'neutral'}">
                                            {metrics.profit_factor:.2f}
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="metric-card">
                                        <h5>Sharpe Ratio</h5>
                                        <div class="metric {'good' if metrics.sharpe_ratio > 1 else 'neutral'}">
                                            {metrics.sharpe_ratio:.2f}
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="metric-card">
                                        <h5>Max Drawdown</h5>
                                        <div class="metric bad">
                                            ${metrics.max_drawdown:.2f}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Pattern Analysis -->
                    <div class="card mb-4">
                        <div class="card-header">
                            <h2>Pattern Analysis</h2>
                        </div>
                        <div class="card-body">
                            <div id="timeHeatmap" class="mb-4">
                                <iframe src="time_performance.html" width="100%" height="400px" frameborder="0"></iframe>
                            </div>
                            <div id="marketConditions" class="mb-4">
                                <iframe src="market_conditions.html" width="100%" height="400px" frameborder="0"></iframe>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Indicator Analysis -->
                    <div class="card mb-4">
                        <div class="card-header">
                            <h2>Indicator Analysis</h2>
                        </div>
                        <div class="card-body">
                            <div id="indicatorPerformance">
                                <iframe src="indicator_performance.html" width="100%" height="600px" frameborder="0"></iframe>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Recommendations -->
                    <div class="card mb-4">
                        <div class="card-header">
                            <h2>Recommendations</h2>
                        </div>
                        <div class="card-body">
                            <ul class="list-group">
                                {''.join([f'<li class="list-group-item">{r}</li>' for r in recommendations])}
                            </ul>
                        </div>
                    </div>
                </div>
                
                <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
            </body>
            </html>
            """
            
            # Save HTML report
            with open(output_dir / 'analysis_report.html', 'w') as f:
                f.write(html_content)
            
            logger.info("Analysis report generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating analysis report: {e}")


class EnhancedTradingStrategy(Strategy):
    """Enhanced trading strategy with proper indicator plotting and trade execution"""
    
    def init(self):
        # Initialize price data
        self.close = self.data.Close
        self.high = self.data.High
        self.low = self.data.Low
        self.volume = self.data.Volume
        
        # Calculate primary indicators
        self.rsi = self.I(talib.RSI, self.close, timeperiod=14)
        self.macd, self.macd_signal, self.macd_hist = self.I(talib.MACD, self.close, 
                                                            fastperiod=12, 
                                                            slowperiod=26, 
                                                            signalperiod=9)
        self.sma20 = self.I(talib.SMA, self.close, timeperiod=20)
        self.sma50 = self.I(talib.SMA, self.close, timeperiod=50)
        
        # Calculate Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(talib.BBANDS, self.close, 
                                                             timeperiod=20, 
                                                             nbdevup=2, 
                                                             nbdevdn=2)
        
        # Initialize trade variables
        self.last_buy_price = 0
        self.trailing_stop = None
        self.in_position = False
        
        # Plot indicators
        self.plot_indicators()

    def plot_indicators(self):
        # Plot primary indicators
        self.I(lambda: self.sma20, overlay=True, color='blue', name='SMA20')
        self.I(lambda: self.sma50, overlay=True, color='red', name='SMA50')
        self.I(lambda: self.bb_upper, overlay=True, color='gray', name='BB Upper')
        self.I(lambda: self.bb_lower, overlay=True, color='gray', name='BB Lower')
        
        # Plot RSI
        self.I(lambda: self.rsi, panel='RSI', color='purple', name='RSI')
        self.I(lambda: np.full_like(self.rsi, 70), panel='RSI', color='red', name='RSI Upper')
        self.I(lambda: np.full_like(self.rsi, 30), panel='RSI', color='green', name='RSI Lower')
        
        # Plot MACD
        self.I(lambda: self.macd, panel='MACD', color='blue', name='MACD')
        self.I(lambda: self.macd_signal, panel='MACD', color='orange', name='Signal')
        self.I(lambda: self.macd_hist, panel='MACD', color='gray', type='histogram', name='Histogram')

    def next(self):
        # Skip if not enough data
        if len(self.data) < 50:
            return

        # Get current price and indicator values
        price = self.close[-1]
        rsi = self.rsi[-1]
        macd = self.macd[-1]
        macd_signal = self.macd_signal[-1]
        prev_macd = self.macd[-2]
        prev_macd_signal = self.macd_signal[-2]
        
        # Entry conditions
        long_condition = (
            rsi < 30 and  # Oversold RSI
            macd > macd_signal and  # MACD bullish crossover
            prev_macd <= prev_macd_signal and
            price > self.sma20[-1]  # Price above SMA20
        )
        
        # Exit conditions
        exit_condition = (
            rsi > 70 or  # Overbought RSI
            (macd < macd_signal and prev_macd >= prev_macd_signal) or  # MACD bearish crossover
            price < self.sma50[-1]  # Price below SMA50
        )
        
        # Position management
        if not self.position:
            if long_condition:
                # Calculate position size (1% risk per trade)
                risk_pct = 0.01
                stop_loss = price * 0.98  # 2% stop loss
                risk_amount = price - stop_loss
                pos_size = (self.equity * risk_pct) / risk_amount
                
                # Enter long position
                self.buy(size=pos_size)
                self.last_buy_price = price
                self.trailing_stop = stop_loss
                self.in_position = True
                
        elif self.position and self.in_position:
            # Update trailing stop
            if price > self.last_buy_price:
                new_stop = price * 0.98
                if new_stop > self.trailing_stop:
                    self.trailing_stop = new_stop
            
            # Check exit conditions
            if exit_condition or price < self.trailing_stop:
                self.position.close()
                self.in_position = False
                self.trailing_stop = None

    def create_strategy(data):
        """Create and run the enhanced strategy"""
        from backtesting import Backtest
        
        # Initialize backtest
        bt = Backtest(data, EnhancedTradingStrategy,
                    cash=100000,
                    commission=0.002,
                    exclusive_orders=True)

        # Run backtest
        stats = bt.run()
        
        # Plot results with indicators
        bt.plot()
        
        return stats

class VisualizationEngine:
    """Handle all visualization related tasks with advanced charting capabilities."""
    
    def __init__(self, report_dir: Path):
        """Initialize visualization engine with report directory."""
        try:
            self.report_dir = report_dir
            self.report_dir.mkdir(parents=True, exist_ok=True)
            
            # Store figures for combined export
            self.figures = {}
            
            # Configure default styling
            self.chart_config = {
                'theme': 'plotly_white',
                'height': 800,
                'width': 1200,
                'template': 'plotly_white',
                'font': dict(family="Arial, sans-serif", size=12),
                'margin': dict(l=50, r=50, t=50, b=50)
            }
            
            logger.info(f"Initialized VisualizationEngine with output dir: {report_dir}")
            
        except Exception as e:
            logger.error(f"Error initializing VisualizationEngine: {e}")
            raise

    def create_candlestick_chart(self, df: pd.DataFrame, trades: List):
        """Create interactive candlestick chart with trades."""
        try:
            # Create figure with secondary y-axis
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=('Price Action', 'Volume'),
                row_heights=[0.7, 0.3]
            )
            
            # Add candlestick
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Add volume bars
            colors = ['red' if close < open else 'green' 
                     for close, open in zip(df['Close'], df['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.5
                ),
                row=2, col=1
            )
            
            # Add trades
            for trade in trades:
                # Entry points
                fig.add_trace(
                    go.Scatter(
                        x=[trade.entry_time],
                        y=[trade.entry_price],
                        mode='markers',
                        name='Entry',
                        marker=dict(
                            symbol='triangle-up' if trade.size > 0 else 'triangle-down',
                            size=12,
                            color='green' if trade.size > 0 else 'red',
                            line=dict(width=1, color='black')
                        )
                    ),
                    row=1, col=1
                )
                
                # Exit points
                fig.add_trace(
                    go.Scatter(
                        x=[trade.exit_time],
                        y=[trade.exit_price],
                        mode='markers',
                        name='Exit',
                        marker=dict(
                            symbol='x',
                            size=10,
                            color='black',
                            line=dict(width=1)
                        )
                    ),
                    row=1, col=1
                )
                
                # Add trade connection lines
                fig.add_trace(
                    go.Scatter(
                        x=[trade.entry_time, trade.exit_time],
                        y=[trade.entry_price, trade.exit_price],
                        mode='lines',
                        line=dict(
                            color='green' if trade.pl > 0 else 'red',
                            width=1,
                            dash='dot'
                        ),
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # Update layout
            fig.update_layout(
                title='Trading Activity with Volume',
                yaxis_title='Price',
                yaxis2_title='Volume',
                xaxis_rangeslider_visible=False,
                **self.chart_config
            )
            
            # Add range slider
            fig.update_xaxes(rangeslider_visible=False)
            
            # Store figure
            self.figures['candlestick'] = fig
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating candlestick chart: {e}")
            return None

    def create_indicator_overlay(self, df: pd.DataFrame, indicators: Dict):
        """Add technical indicators to chart with advanced visualization."""
        try:
            # Create figure with multiple subplots
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Price & Indicators', 'Momentum', 'Volume'),
                row_heights=[0.5, 0.25, 0.25]
            )
            
            # Add price
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Add moving averages
            if 'SMA_20' in indicators:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=indicators['SMA_20'],
                        name='SMA 20',
                        line=dict(color='blue', width=1)
                    ),
                    row=1, col=1
                )
            
            if 'EMA_50' in indicators:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=indicators['EMA_50'],
                        name='EMA 50',
                        line=dict(color='orange', width=1)
                    ),
                    row=1, col=1
                )
            
            # Add RSI
            if 'RSI' in indicators:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=indicators['RSI'],
                        name='RSI',
                        line=dict(color='purple')
                    ),
                    row=2, col=1
                )
                
                # Add RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", 
                            annotation_text="Overbought", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green",
                            annotation_text="Oversold", row=2, col=1)
            
            # Add MACD
            if all(x in indicators for x in ['MACD', 'MACD_Signal']):
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=indicators['MACD'],
                        name='MACD',
                        line=dict(color='blue')
                    ),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=indicators['MACD_Signal'],
                        name='Signal',
                        line=dict(color='orange')
                    ),
                    row=2, col=1
                )
                # Add MACD histogram
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=indicators['MACD'] - indicators['MACD_Signal'],
                        name='MACD Hist',
                        marker_color='gray'
                    ),
                    row=2, col=1
                )
            
            # Add volume
            colors = ['red' if close < open else 'green'
                     for close, open in zip(df['Close'], df['Open'])]
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color=colors
                ),
                row=3, col=1
            )
            
            # Update layout
            fig.update_layout(
                title='Technical Analysis Dashboard',
                yaxis_title='Price',
                yaxis2_title='Momentum',
                yaxis3_title='Volume',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                **self.chart_config
            )
            
            # Store figure
            self.figures['indicators'] = fig
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating indicator overlay: {e}")
            return None

    def create_volume_profile(self, df: pd.DataFrame):
        """Create volume profile visualization with advanced analytics."""
        try:
            # Calculate price levels
            price_range = df['High'].max() - df['Low'].min()
            num_levels = 50
            level_height = price_range / num_levels
            
            # Create price levels
            levels = np.linspace(df['Low'].min(), df['High'].max(), num_levels)
            volume_profile = np.zeros(len(levels) - 1)
            
            # Calculate volume at each level
            for i in range(len(levels) - 1):
                mask = (df['Low'] <= levels[i+1]) & (df['High'] >= levels[i])
                volume_profile[i] = df.loc[mask, 'Volume'].sum()
            
            # Create figure
            fig = go.Figure()
            
            # Add volume profile
            fig.add_trace(
                go.Bar(
                    x=volume_profile,
                    y=levels[:-1],
                    orientation='h',
                    name='Volume Profile',
                    marker_color='rgba(0,0,255,0.5)'
                )
            )
            
            # Add POC (Point of Control)
            poc_level = levels[np.argmax(volume_profile)]
            fig.add_hline(
                y=poc_level,
                line_dash="dash",
                annotation_text="POC",
                annotation_position="right"
            )
            
            # Add Value Area
            value_area_volume = np.sum(volume_profile) * 0.70  # 70% of total volume
            sorted_indices = np.argsort(volume_profile)[::-1]
            cumsum_volume = np.cumsum(volume_profile[sorted_indices])
            value_area_indices = sorted_indices[cumsum_volume <= value_area_volume]
            
            va_high = levels[max(value_area_indices)]
            va_low = levels[min(value_area_indices)]
            
            fig.add_hrect(
                y0=va_low,
                y1=va_high,
                fillcolor="rgba(0,255,0,0.1)",
                annotation_text="Value Area",
                annotation_position="right"
            )
            
            # Update layout
            fig.update_layout(
                title='Volume Profile Analysis',
                xaxis_title='Volume',
                yaxis_title='Price',
                showlegend=True,
                **self.chart_config
            )
            
            # Store figure
            self.figures['volume_profile'] = fig
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating volume profile: {e}")
            return None

    def create_trade_heatmap(self, trades: List):
        """Create heatmap of trade performance by time and day."""
        try:
            # Convert trades to DataFrame
            trades_df = pd.DataFrame([{
                'entry_time': t.entry_time,
                'pnl': t.pl,
                'return': t.pl_pct
            } for t in trades])
            
            # Add time components
            trades_df['hour'] = trades_df['entry_time'].dt.hour
            trades_df['day'] = trades_df['entry_time'].dt.day_name()
            
            # Create pivot table
            pivot_returns = trades_df.pivot_table(
                values='return',
                index='hour',
                columns='day',
                aggfunc='mean'
            )
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=pivot_returns.values,
                x=pivot_returns.columns,
                y=pivot_returns.index,
                colorscale='RdYlGn',
                colorbar_title='Return %'
            ))
            
            # Add trade count overlay
            pivot_counts = trades_df.pivot_table(
                values='pnl',
                index='hour',
                columns='day',
                aggfunc='count'
            )
            
            # Add trade counts as text
            for i in range(len(pivot_returns.index)):
                for j in range(len(pivot_returns.columns)):
                    fig.add_annotation(
                        text=str(int(pivot_counts.iloc[i, j])) if not np.isnan(pivot_counts.iloc[i, j]) else '',
                        x=pivot_returns.columns[j],
                        y=pivot_returns.index[i],
                        showarrow=False,
                        font=dict(color='black')
                    )
            
            # Update layout
            fig.update_layout(
                title='Trade Performance Heatmap',
                xaxis_title='Day of Week',
                yaxis_title='Hour of Day',
                **self.chart_config
            )
            
            # Store figure
            self.figures['heatmap'] = fig
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating trade heatmap: {e}")
            return None

    def export_to_html(self, filename: str):
        """Export all visualizations to HTML with interactive dashboard."""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Trading Analysis Dashboard</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                <link href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@5.15.4/css/all.min.css" rel="stylesheet">
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background-color: #f8f9fa;
                    }}
                    .dashboard-container {{
                        padding: 20px;
                    }}
                    .chart-container {{
                        background: white;
                        border-radius: 10px;
                        padding: 20px;
                        margin-bottom: 20px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    .sidebar {{
                        background: white;
                        border-radius: 10px;
                        padding: 20px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    .nav-link {{
                        color: #495057;
                        padding: 10px 15px;
                        border-radius: 5px;
                        margin-bottom: 5px;
                    }}
                    .nav-link:hover {{
                        background-color: #e9ecef;
                    }}
                    .nav-link.active {{
                        background-color: #0d6efd;
                        color: white;
                    }}
                    .chart-title {{
                        font-size: 1.2rem;
                        font-weight: 500;
                        margin-bottom: 15px;
                    }}
                    .control-panel {{
                        background: white;
                        border-radius: 10px;
                        padding: 15px;
                        margin-bottom: 20px;
                    }}
                    .btn-control {{
                        margin: 5px;
                    }}
                    .loading-overlay {{
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        background: rgba(255,255,255,0.8);
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        z-index: 9999;
                        visibility: hidden;
                    }}
                    .spinner {{
                        width: 50px;
                        height: 50px;
                    }}
                </style>
            </head>
            <body>
                <!-- Loading overlay -->
                <div id="loadingOverlay" class="loading-overlay">
                    <div class="spinner-border text-primary spinner" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                </div>
                
                <div class="container-fluid">
                    <div class="row">
                        <!-- Sidebar -->
                        <div class="col-md-2 sidebar">
                            <h4 class="mb-4">Navigation</h4>
                            <div class="nav flex-column nav-pills" role="tablist">
                                <button class="nav-link active" data-bs-toggle="pill" data-bs-target="#price">
                                    <i class="fas fa-chart-line me-2"></i>Price Action
                                </button>
                                <button class="nav-link" data-bs-toggle="pill" data-bs-target="#indicators">
                                    <i class="fas fa-chart-bar me-2"></i>Technical Indicators
                                </button>
                                <button class="nav-link" data-bs-toggle="pill" data-bs-target="#volume">
                                    <i class="fas fa-chart-area me-2"></i>Volume Analysis
                                </button>
                                <button class="nav-link" data-bs-toggle="pill" data-bs-target="#performance">
                                    <i class="fas fa-chart-pie me-2"></i>Performance
                                </button>
                            </div>
                            
                            <!-- Control Panel -->
                            <div class="control-panel mt-4">
                                <h5>Controls</h5>
                                <div class="d-grid gap-2">
                                    <button class="btn btn-sm btn-outline-primary" onclick="toggleIndicators()">
                                        <i class="fas fa-eye"></i> Toggle Indicators
                                    </button>
                                    <button class="btn btn-sm btn-outline-secondary" onclick="resetZoom()">
                                        <i class="fas fa-undo"></i> Reset Zoom
                                    </button>
                                    <button class="btn btn-sm btn-outline-success" onclick="downloadChart()">
                                        <i class="fas fa-download"></i> Download
                                    </button>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Main Content -->
                        <div class="col-md-10">
                            <div class="dashboard-container">
                                <h2 class="mb-4">Trading Analysis Dashboard</h2>
                                
                                <!-- Tab Content -->
                                <div class="tab-content">
                                    <!-- Price Action Tab -->
                                    <div class="tab-pane fade show active" id="price">
                                        <div class="chart-container">
                                            <div class="chart-title">Price Action & Trades</div>
                                            <div id="candlestick-chart"></div>
                                        </div>
                                    </div>
                                    
                                    <!-- Indicators Tab -->
                                    <div class="tab-pane fade" id="indicators">
                                        <div class="chart-container">
                                            <div class="chart-title">Technical Indicators</div>
                                            <div id="indicator-chart"></div>
                                        </div>
                                    </div>
                                    
                                    <!-- Volume Tab -->
                                    <div class="tab-pane fade" id="volume">
                                        <div class="chart-container">
                                            <div class="chart-title">Volume Profile</div>
                                            <div id="volume-profile"></div>
                                        </div>
                                    </div>
                                    
                                    <!-- Performance Tab -->
                                    <div class="tab-pane fade" id="performance">
                                        <div class="chart-container">
                                            <div class="chart-title">Trading Performance</div>
                                            <div id="heatmap-chart"></div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
                <script>
                    // Show loading overlay
                    function showLoading() {{
                        document.getElementById('loadingOverlay').style.visibility = 'visible';
                    }}
                    
                    // Hide loading overlay
                    function hideLoading() {{
                        document.getElementById('loadingOverlay').style.visibility = 'hidden';
                    }}
                    
                    // Initialize charts
                    document.addEventListener('DOMContentLoaded', function() {{
                        showLoading();
                        
                        // Add charts
                        {self._generate_chart_initialization_code()}
                        
                        hideLoading();
                        
                        // Add resize handler
                        window.addEventListener('resize', function() {{
                            var charts = document.getElementsByClassName('chart-container');
                            for (var i = 0; i < charts.length; i++) {{
                                Plotly.Plots.resize(charts[i].firstElementChild);
                            }}
                        }});
                    }});
                    
                    // Toggle indicators visibility
                    function toggleIndicators() {{
                        showLoading();
                        var charts = ['indicator-chart'];
                        charts.forEach(function(chartId) {{
                            var chart = document.getElementById(chartId);
                            if (chart) {{
                                var visibility = chart.style.display === 'none' ? 'block' : 'none';
                                chart.style.display = visibility;
                            }}
                        }});
                        hideLoading();
                    }}
                    
                    // Reset zoom levels
                    function resetZoom() {{
                        showLoading();
                        Object.keys(charts).forEach(function(chartId) {{
                            if (charts[chartId]) {{
                                Plotly.relayout(chartId, {{'xaxis.autorange': true, 'yaxis.autorange': true}});
                            }}
                        }});
                        hideLoading();
                    }}
                    
                    // Download chart as PNG
                    function downloadChart() {{
                        var activeTab = document.querySelector('.tab-pane.active');
                        var chartDiv = activeTab.querySelector('[id$="-chart"]');
                        if (chartDiv) {{
                            Plotly.downloadImage(chartDiv, {{
                                format: 'png',
                                width: 1200,
                                height: 800,
                                filename: 'trading_chart'
                            }});
                        }}
                    }}
                    
                    // Initialize tooltips
                    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
                    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {{
                        return new bootstrap.Tooltip(tooltipTriggerEl)
                    }});
                </script>
            </body>
            </html>
            """
            
            # Save HTML file
            output_path = self.report_dir / filename
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Successfully exported visualization dashboard to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting to HTML: {e}")
            raise

    def _generate_chart_initialization_code(self) -> str:
        """Generate JavaScript code for chart initialization."""
        try:
            code = []
            for name, fig in self.figures.items():
                if fig is not None:
                    chart_data = fig.to_json()
                    code.append(f"""
                        // Initialize {name} chart
                        var {name}Data = {chart_data};
                        Plotly.newPlot('{name}-chart', 
                                    {name}Data.data,
                                    {name}Data.layout,
                                    {{responsive: true, 
                                    displayModeBar: true,
                                    modeBarButtonsToAdd: ['drawline', 'drawopenpath', 'eraseshape'],
                                    modeBarButtonsToRemove: ['sendDataToCloud']}});
                    """)
            
            return "\n".join(code)
            
        except Exception as e:
            logger.error(f"Error generating chart initialization code: {e}")
            return ""

    def _export_static_images(self):
        """Export static images of all charts for documentation."""
        try:
            for name, fig in self.figures.items():
                if fig is not None:
                    # Export as PNG
                    png_path = self.report_dir / f"{name}.png"
                    fig.write_image(str(png_path))
                    
                    # Export as HTML with basic interactivity
                    html_path = self.report_dir / f"{name}.html"
                    fig.write_html(str(html_path))
            
            logger.info("Exported static images of all charts")
            
        except Exception as e:
            logger.error(f"Error exporting static images: {e}")

    def add_annotations(self, chart_type: str, annotations: List[Dict]):
        """Add custom annotations to specified chart."""
        try:
            if chart_type in self.figures:
                fig = self.figures[chart_type]
                
                for annotation in annotations:
                    fig.add_annotation(
                        x=annotation.get('x'),
                        y=annotation.get('y'),
                        text=annotation.get('text', ''),
                        showarrow=annotation.get('arrow', True),
                        arrowhead=annotation.get('arrowhead', 1),
                        ax=annotation.get('ax', 0),
                        ay=annotation.get('ay', -40)
                    )
                
                logger.info(f"Added annotations to {chart_type} chart")
                
        except Exception as e:
            logger.error(f"Error adding annotations: {e}")

    def add_shapes(self, chart_type: str, shapes: List[Dict]):
        """Add custom shapes to specified chart."""
        try:
            if chart_type in self.figures:
                fig = self.figures[chart_type]
                
                for shape in shapes:
                    fig.add_shape(
                        type=shape.get('type', 'line'),
                        x0=shape.get('x0'),
                        y0=shape.get('y0'),
                        x1=shape.get('x1'),
                        y1=shape.get('y1'),
                        line=shape.get('line', dict(color="red", width=2)),
                        fillcolor=shape.get('fillcolor'),
                        opacity=shape.get('opacity', 0.7)
                    )
                
                logger.info(f"Added shapes to {chart_type} chart")
                
        except Exception as e:
            logger.error(f"Error adding shapes: {e}")

    def update_layout(self, chart_type: str, layout_updates: Dict):
        """Update layout of specified chart."""
        try:
            if chart_type in self.figures:
                fig = self.figures[chart_type]
                fig.update_layout(**layout_updates)
                logger.info(f"Updated layout of {chart_type} chart")
                
        except Exception as e:
            logger.error(f"Error updating layout: {e}")

    def add_range_slider(self, chart_type: str):
        """Add range slider to specified chart."""
        try:
            if chart_type in self.figures:
                fig = self.figures[chart_type]
                fig.update_layout(
                    xaxis=dict(
                        rangeslider=dict(visible=True),
                        type="date"
                    )
                )
                logger.info(f"Added range slider to {chart_type} chart")
                
        except Exception as e:
            logger.error(f"Error adding range slider: {e}")

    def save_config(self):
        """Save current visualization configuration."""
        try:
            config = {
                'chart_config': self.chart_config,
                'figures': {
                    name: {
                        'layout': fig.layout if fig is not None else None
                    } for name, fig in self.figures.items()
                }
            }
            
            config_path = self.report_dir / 'viz_config.json'
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            logger.info(f"Saved visualization configuration to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")


# # ###Example usage

# Utility Functions
def get_date_ranges() -> tuple:
    """Calculate date ranges for backtesting."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
        end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')
        
        return start_str, end_str
    except Exception as e:
        logger.error(f"Error calculating date ranges: {e}")
        raise

def format_metrics(stats: Dict) -> str:
    """Format metrics for pretty printing with default values and error handling."""
    try:
        # Define default values for all metrics
        default_stats = {
            'Return [%]': 0.0,
            'Return (Ann.) [%]': 0.0,
            'Sharpe Ratio': 0.0,
            'Sortino Ratio': 0.0,
            'Max. Drawdown [%]': 0.0,
            'Win Rate [%]': 0.0,
            '# Trades': 0,
            'Profit Factor': 0.0,
            'Avg. Trade [%]': 0.0,
            'Best Trade [%]': 0.0,
            'Worst Trade [%]': 0.0
        }

        # Update default stats with actual values
        default_stats.update(stats)
        
        # Calculate annual return if missing
        if 'Return (Ann.) [%]' not in stats and 'Return [%]' in stats:
            # Assuming 252 trading days per year
            days = (pd.to_datetime(stats.get('End')) - pd.to_datetime(stats.get('Start'))).days
            if days > 0:
                annual_return = (1 + stats['Return [%]']/100)**(252/days) - 1
                default_stats['Return (Ann.) [%]'] = annual_return * 100
        
        metrics = [
            ("Performance Metrics", [
                ("Total Return", f"{default_stats['Return [%]']:.2f}%"),
                ("Annual Return", f"{default_stats['Return (Ann.) [%]']:.2f}%"),
                ("Sharpe Ratio", f"{default_stats['Sharpe Ratio']:.2f}"),
                ("Sortino Ratio", f"{default_stats['Sortino Ratio']:.2f}"),
                ("Max Drawdown", f"{default_stats['Max. Drawdown [%]']:.2f}%"),
                ("Win Rate", f"{default_stats['Win Rate [%]']:.2f}%"),
            ]),
            ("Trade Statistics", [
                ("Total Trades", str(default_stats['# Trades'])),
                ("Profit Factor", f"{default_stats['Profit Factor']:.2f}"),
                ("Average Trade", f"{default_stats['Avg. Trade [%]']:.2f}%"),
                ("Best Trade", f"{default_stats['Best Trade [%]']:.2f}%"),
                ("Worst Trade", f"{default_stats['Worst Trade [%]']:.2f}%"),
            ])
        ]
        
        output = []
        for section, items in metrics:
            output.append(f"\n{section}:")
            output.append("=" * 50)
            for key, value in items:
                output.append(f"{key:<20}: {value}")
        
        return "\n".join(output)
        
    except Exception as e:
        logger.error(f"Error formatting metrics: {e}")
        # Return basic error message if formatting fails
        return "Error formatting metrics. Check logs for details."

def analyze_results(stats: Dict, company_name: str) -> None:
    """Analyze and interpret backtest results."""
    try:
        analysis = []
        
        # Performance Analysis
        if stats['Return [%]'] > 0:
            analysis.append(f"✅ Strategy is profitable with {stats['Return [%]']:.2f}% return")
        else:
            analysis.append(f"❌ Strategy is unprofitable with {stats['Return [%]']:.2f}% loss")
            
        if stats['Sharpe Ratio'] > 1:
            analysis.append(f"✅ Good risk-adjusted returns (Sharpe: {stats['Sharpe Ratio']:.2f})")
        else:
            analysis.append(f"❌ Poor risk-adjusted returns (Sharpe: {stats['Sharpe Ratio']:.2f})")
        
        print("\nStrategy Analysis:")
        print("=" * 50)
        for point in analysis:
            print(point)
            
    except Exception as e:
        logger.error(f"Error analyzing results: {e}")
        raise

# Main execution
def main():
    """Example usage of backtesting agent."""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logger.info("Starting backtesting agent")
        
        # Initialize backtesting agent
        backtest_agent = BacktestingAgent()
        company_name = "ZOMATO"
        
        # Get date ranges automatically
        start_date, end_date = get_date_ranges()
        logger.info(f"Testing period: {start_date} to {end_date}")
        


        # Set dates
        start_date = '2023-01-01'
        end_date = '2024-11-09'

        # Run test
        stats = test_strategy('ZOMATO', start_date, end_date)


        # # Run backtest
        # stats = backtest_agent.run_backtest(
        #     company_name=company_name,
        #     start_date=start_date,
        #     end_date=end_date
        # )
        
        # Print formatted results
        print(format_metrics(stats))
        
        # Analyze results
        analyze_results(stats, company_name)
        
        logger.info("Backtesting completed successfully")
        logger.info(f"Full reports available in: {backtest_agent.output_dir}/backtest_results/")

    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()

