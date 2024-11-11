import glob
import json
import re
import traceback
import pandas as pd
import numpy as np
import talib
import talib as ta
import logging
import os
from datetime import datetime
from backtesting import Backtest, Strategy
from pathlib import Path
import matplotlib.pyplot as plt
# plt.style.use('seaborn')  # Optional: for better looking plots
import seaborn as sns
sns.set_style("darkgrid")
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


# Configure style without depending on seaborn
plt.style.use('ggplot')  # Using a built-in style that's similar to seaborn

# Create logs directory if it doesn't exist
log_dir = Path(os.getcwd()) / "logs"
log_dir.mkdir(exist_ok=True)

# Create a unique log filename with timestamp
log_filename = log_dir / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure logging with both file and console handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # This will still print to console
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Starting backtest, logging to {log_filename}")

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)


# Create a unique log filename with timestamp and company name
def setup_logging(company_name):
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create logs directory using pathlib
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    # Create unique log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = log_dir / f"backtest_{company_name}_{timestamp}.log"

    # Create file handler
    file_handler = logging.FileHandler(filename=log_filename, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Get the logger
    logger = logging.getLogger('backtester')
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    logger.handlers = []

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_filename}")
    
    return logger


def format_stats(stats: dict) -> None:
    """Format and print backtest statistics"""
    try:
        print("\nBacktest Results:")
        print("=" * 50)
        
        # Core metrics - with safe access using .get()
        print(f"Total Return: {stats.get('Return [%]', 0):.2f}%")
        print(f"Sharpe Ratio: {stats.get('Sharpe Ratio', 0):.2f}")
        print(f"Max Drawdown: {stats.get('Max. Drawdown [%]', 0):.2f}%")
        print(f"Win Rate: {stats.get('Win Rate [%]', 0):.2f}%")
        print(f"Total Trades: {stats.get('# Trades', 0)}")
        
        print(f"\nRisk Metrics:")
        print("-" * 50)
        print(f"Exposure Time: {stats.get('Exposure Time [%]', 0):.2f}%")
        print(f"Final Equity: ${stats.get('Equity', stats.get('_equity', 100000)):.2f}")
        print(f"Buy & Hold Return: {stats.get('Buy & Hold Return [%]', 0):.2f}%")
        print(f"Volatility (Ann.) [%]: {stats.get('Volatility (Ann.) [%]', 0):.2f}%")
        
        print(f"\nTrade Analysis:")
        print("-" * 50)
        print(f"Average Trade: {stats.get('Avg. Trade [%]', 0):.2f}%")
        print(f"Best Trade: {stats.get('Best Trade [%]', 0):.2f}%")
        print(f"Worst Trade: {stats.get('Worst Trade [%]', 0):.2f}%")
        print(f"Profit Factor: {stats.get('Profit Factor', 0):.2f}")
        print(f"SQN: {stats.get('SQN', 0):.2f}")
        
        # Display strategy parameters
        print(f"\nStrategy Parameters:")
        print("-" * 50)
        print(f"\nPosition Sizing: Fixed 100 shares per trade")
        print(f"Commission Rate: 0.2%")
        
    except Exception as e:
        logger.error(f"Error formatting statistics: {e}")

class NumpyEncoder(json.JSONEncoder):
    """Enhanced JSON encoder for numpy and pandas types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        elif isinstance(obj, pd.Timedelta):
            return str(obj)
        elif pd.isna(obj):
            return None
        elif isinstance(obj, datetime):
            return str(obj)
        try:
            return super(NumpyEncoder, self).default(obj)
        except TypeError:
            return str(obj)  # Convert any other unserializable objects to string
    
    
class AlgoTradingStrategy(Strategy):
    trade_log = []
    
    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        AlgoTradingStrategy.trade_log = []
        self.indicators = {}
        self.indicator_values = {}
        self.timeframe_data = {}
        self.last_processed_time = {}
        self.logger = logging.getLogger('backtester')
        self.load_strategy_config()
    

    def resample_data(self, timeframe):
        """Resample data to required timeframe"""
        try:
            # Create DataFrame with all required columns
            df = pd.DataFrame({
                'Open': self.data.Open,
                'High': self.data.High,
                'Low': self.data.Low,
                'Close': self.data.Close,
                'Volume': self.data.Volume
            }, index=self.data.index)
            
            # Convert timeframe string to pandas offset
            timeframe_map = {
                '1m': '1T',
                '5m': '5T',
                '15m': '15T',
                '30m': '30T',
                '1h': '60T'
            }
            offset = timeframe_map.get(timeframe)
            
            if offset:
                # Resample using proper OHLCV aggregation
                resampled = df.resample(offset).agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).ffill()
                
                self.logger.info(f"Resampled {timeframe} data: {len(resampled)} rows")
                return resampled
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error in resample_data for {timeframe}: {e}")
            self.logger.error(traceback.format_exc())
            return None


    def update_indicators(self, current_time):
        """Update all indicators for the current tick"""
        try:
            # Update data and indicators for each timeframe
            for timeframe in set(cond['timeframe'] for cond in self.entry_conditions + self.exit_conditions):
                # Get updated data for this timeframe
                resampled_data = self.resample_data(timeframe)
                
                if resampled_data is not None:
                    self.logger.info(f"\nUpdating indicators for {timeframe} timeframe:")
                    # Get required indicators for this timeframe
                    required_indicators = {cond['indicator'] for cond in self.entry_conditions + self.exit_conditions 
                                        if cond['timeframe'] == timeframe}
                    
                    # Calculate each required indicator
                    for indicator in required_indicators:
                        key = f"{indicator}_{timeframe}"
                        self.indicators[key] = self.calculate_indicator(indicator, timeframe, resampled_data)
                        if self.indicators[key] is not None and len(self.indicators[key]) > 0:
                            self.logger.info(f"Updated {key} = {float(self.indicators[key][-1]):.2f}")
                        else:
                            self.logger.warning(f"Failed to update {key}")
                else:
                    self.logger.error(f"Failed to update {timeframe} timeframe data")
                    
        except Exception as e:
            self.logger.error(f"Error updating indicators: {e}")
            self.logger.error(traceback.format_exc())


    def load_strategy_config(self):
        """Load the latest algorithm configuration"""
        try:
            base_path = Path(os.getcwd())
            algo_dir = base_path / "output" / "algo"
            pattern = "*_algorithm-*.json"
            
            algo_files = list(algo_dir.glob(pattern))
            if not algo_files:
                raise FileNotFoundError("No algorithm files found")
            
            # Debug print available files
            self.logger.info("Available algorithm files:")
            for file in algo_files:
                self.logger.info(f"- {file}")
            
            latest_file = max(algo_files, key=lambda x: 
                int(re.search(r'-(\d+)\.json$', x.name).group(1)))
            
            self.logger.info(f"Loading strategy from: {latest_file}")
            
            with open(latest_file, 'r') as f:
                self.strategy_config = json.load(f)
            
            # Extract conditions
            self.entry_conditions = self.strategy_config.get('entry_conditions', [])
            self.exit_conditions = self.strategy_config.get('exit_conditions', [])
            self.risk_management = self.strategy_config.get('risk_management', {
                'max_position_size': 0.1,
                'stop_loss': 0.02,
                'take_profit': 0.03
            })
            self.trading_hours = self.strategy_config.get('trading_hours', {
                'start': '09:15',
                'end': '15:20'
            })
            
            # Debug print loaded configuration
            self.logger.info("\nLoaded strategy configuration:")
            self.logger.info(f"Entry conditions: {json.dumps(self.entry_conditions, indent=2)}")
            self.logger.info(f"Exit conditions: {json.dumps(self.exit_conditions, indent=2)}")
            self.logger.info(f"Risk management: {json.dumps(self.risk_management, indent=2)}")
            self.logger.info(f"Trading hours: {json.dumps(self.trading_hours, indent=2)}")
            
            # Verify indicators required
            all_conditions = self.entry_conditions + self.exit_conditions
            required_indicators = {cond['indicator'] for cond in all_conditions}
            required_timeframes = {cond['timeframe'] for cond in all_conditions}
            
            self.logger.info(f"\nRequired indicators: {required_indicators}")
            self.logger.info(f"Required timeframes: {required_timeframes}")
            
        except Exception as e:
            self.logger.error(f"Error loading strategy config: {e}")
            self.logger.error(traceback.format_exc())
            raise


    def init(self):
        """Initialize indicators based on strategy configuration"""
        try:
            # Collect all unique indicators and timeframes
            required_indicators = set()
            timeframes = set()
            
            for condition in self.entry_conditions + self.exit_conditions:
                required_indicators.add(condition['indicator'])
                timeframes.add(condition['timeframe'])
            
            self.logger.info(f"Required indicators: {required_indicators}")
            self.logger.info(f"Required timeframes: {timeframes}")
            
            # Initialize data and indicators for each timeframe
            for timeframe in timeframes:
                # Initialize timeframe data
                self.timeframe_data[timeframe] = self.resample_data(timeframe)
                if self.timeframe_data[timeframe] is not None:
                    self.logger.info(f"Initialized {timeframe} timeframe data with {len(self.timeframe_data[timeframe])} rows")
                    
                    # Calculate initial indicators for this timeframe
                    for indicator in required_indicators:
                        key = f"{indicator}_{timeframe}"
                        self.indicators[key] = self.calculate_indicator(indicator, timeframe)
                        if self.indicators[key] is not None:
                            self.logger.info(f"Initialized {key} with {len(self.indicators[key])} values")
                            self.logger.info(f"Latest {key}: {self.indicators[key][-1]:.2f}")
                        else:
                            self.logger.error(f"Failed to initialize {key}")
                else:
                    self.logger.error(f"Failed to initialize {timeframe} timeframe data")
            
        except Exception as e:
            self.logger.error(f"Error in init(): {e}")
            self.logger.error(traceback.format_exc())
            raise


    def calculate_indicator(self, indicator_name, timeframe, data=None):
        """Calculate specific indicator for given timeframe"""
        try:
            if data is None:
                data = self.resample_data(timeframe)
                
            if data is None or len(data) < 2:
                return None
                
            # Get the data arrays
            close_prices = np.array(data['Close'], dtype=float)
            high_prices = np.array(data['High'], dtype=float)
            low_prices = np.array(data['Low'], dtype=float)
            volume = np.array(data['Volume'], dtype=float)
            
            self.logger.info(f"Calculating {indicator_name} for {timeframe}")
            self.logger.info(f"Data points available: {len(data)}")
            
            if indicator_name == "RSI":
                values = talib.RSI(close_prices, timeperiod=14)
                self.logger.info(f"RSI values: latest={values[-1]:.2f}, prev={values[-2]:.2f}")
                return values
                
            elif indicator_name == "MACD":
                macd, signal, _ = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
                self.logger.info(f"MACD values: latest={macd[-1]:.2f}, prev={macd[-2]:.2f}")
                return macd
                
            elif indicator_name == "SMA":
                values = talib.SMA(close_prices, timeperiod=20)
                self.logger.info(f"SMA values: latest={values[-1]:.2f}, prev={values[-2]:.2f}")
                return values
                
            elif indicator_name == "EMA":
                values = talib.EMA(close_prices, timeperiod=50)
                self.logger.info(f"EMA values: latest={values[-1]:.2f}, prev={values[-2]:.2f}")
                return values
                
            elif indicator_name == "WMA":
                values = talib.WMA(close_prices, timeperiod=20)
                self.logger.info(f"WMA values: latest={values[-1]:.2f}, prev={values[-2]:.2f}")
                return values
                
            elif indicator_name == "VWAP":
                typical_price = (high_prices + low_prices + close_prices) / 3
                values = (typical_price * volume).cumsum() / volume.cumsum()
                self.logger.info(f"VWAP values: latest={values[-1]:.2f}, prev={values[-2]:.2f}")
                return values
                
            elif indicator_name == "Stochastic_K":
                k, _ = talib.STOCH(high_prices, low_prices, close_prices, fastk_period=14, slowk_period=3, slowd_period=3)
                self.logger.info(f"Stochastic K values: latest={k[-1]:.2f}, prev={k[-2]:.2f}")
                return k
                
            elif indicator_name == "Stochastic_D":
                _, d = talib.STOCH(high_prices, low_prices, close_prices, fastk_period=14, slowk_period=3, slowd_period=3)
                self.logger.info(f"Stochastic D values: latest={d[-1]:.2f}, prev={d[-2]:.2f}")
                return d
                
            elif indicator_name == "WILLR":
                values = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)
                self.logger.info(f"WILLR values: latest={values[-1]:.2f}, prev={values[-2]:.2f}")
                return values
                
            elif indicator_name == "BB_Upper":
                upper, _, _ = talib.BBANDS(close_prices, timeperiod=20)
                self.logger.info(f"BB Upper values: latest={upper[-1]:.2f}, prev={upper[-2]:.2f}")
                return upper
                
            elif indicator_name == "BB_Middle":
                _, middle, _ = talib.BBANDS(close_prices, timeperiod=20)
                self.logger.info(f"BB Middle values: latest={middle[-1]:.2f}, prev={middle[-2]:.2f}")
                return middle
                
            elif indicator_name == "BB_Lower":
                _, _, lower = talib.BBANDS(close_prices, timeperiod=20)
                self.logger.info(f"BB Lower values: latest={lower[-1]:.2f}, prev={lower[-2]:.2f}")
                return lower
                
            elif indicator_name == "ATR":
                values = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
                self.logger.info(f"ATR values: latest={values[-1]:.2f}, prev={values[-2]:.2f}")
                return values
                
            elif indicator_name == "KC_Upper":
                typical_price = (high_prices + low_prices + close_prices) / 3
                middle = talib.EMA(typical_price, timeperiod=20)
                atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
                values = middle + (2 * atr)
                self.logger.info(f"KC Upper values: latest={values[-1]:.2f}, prev={values[-2]:.2f}")
                return values
                
            elif indicator_name == "KC_Lower":
                typical_price = (high_prices + low_prices + close_prices) / 3
                middle = talib.EMA(typical_price, timeperiod=20)
                atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
                values = middle - (2 * atr)
                self.logger.info(f"KC Lower values: latest={values[-1]:.2f}, prev={values[-2]:.2f}")
                return values
                
            elif indicator_name == "OBV":
                values = talib.OBV(close_prices, volume)
                self.logger.info(f"OBV values: latest={values[-1]:.2f}, prev={values[-2]:.2f}")
                return values
                
            elif indicator_name == "MFI":
                values = talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=14)
                self.logger.info(f"MFI values: latest={values[-1]:.2f}, prev={values[-2]:.2f}")
                return values
                
            elif indicator_name == "ADX":
                values = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
                self.logger.info(f"ADX values: latest={values[-1]:.2f}, prev={values[-2]:.2f}")
                return values
                
            elif indicator_name == "CCI":
                values = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
                self.logger.info(f"CCI values: latest={values[-1]:.2f}, prev={values[-2]:.2f}")
                return values
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error calculating {indicator_name} for {timeframe}: {e}")
            self.logger.error(traceback.format_exc())
            return None

    def check_condition(self, condition: dict) -> bool:
        """Check if a specific condition is met"""
        try:
            indicator = condition['indicator']
            condition_type = condition['condition']
            value = condition['value']
            timeframe = condition['timeframe']
            
            key = f"{indicator}_{timeframe}"
            if key not in self.indicators:
                return False
            
            indicator_data = self.indicators[key]
            if len(indicator_data) < 2:
                return False
            
            current_val = indicator_data[-1]
            prev_val = indicator_data[-2]
            
            if condition_type == "above":
                return current_val > value
            elif condition_type == "below":
                return current_val < value
            elif condition_type == "crossover":
                return prev_val <= value and current_val > value
            elif condition_type == "crossunder":
                return prev_val >= value and current_val < value
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking condition: {e}")
            return False


    def next(self):
        """Execute trading logic for each tick"""
        try:
            # Skip if not enough data
            if len(self.data) < 50:
                return
                
            # Get current time and price
            current_time = pd.Timestamp(self.data.index[-1])
            current_time_str = current_time.strftime('%H:%M')
            current_price = self.data.Close[-1]
            
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Processing tick at {current_time}")
            self.logger.info(f"Current price: {current_price}")
            
            # Update all indicators
            self.update_indicators(current_time)
            
            # Check trading hours
            if not (self.trading_hours['start'] <= current_time_str <= self.trading_hours['end']):
                self.logger.info(f"Outside trading hours: {current_time_str}")
                return
            
            # Entry logic - no position
            if not self.position:
                self.logger.info("\nChecking entry conditions:")
                entry_signals = []
                
                # Check each entry condition
                for i, condition in enumerate(self.entry_conditions, 1):
                    key = f"{condition['indicator']}_{condition['timeframe']}"
                    if key in self.indicators and self.indicators[key] is not None:
                        signal = self.check_condition(condition)
                        current_value = float(self.indicators[key][-1])
                        self.logger.info(f"Entry condition {i}: {condition}")
                        self.logger.info(f"Current value: {current_value:.2f}, Target: {condition['value']}, Signal: {signal}")
                        entry_signals.append(signal)
                    else:
                        self.logger.warning(f"Missing indicator data for {key}")
                        entry_signals.append(False)
            
            # Entry logic - no position
            if not self.position:
                self.logger.info("\nChecking entry conditions:")
                entry_signals = []
                
                # Check each entry condition
                for i, condition in enumerate(self.entry_conditions, 1):
                    key = f"{condition['indicator']}_{condition['timeframe']}"
                    if key in self.indicators and self.indicators[key] is not None:
                        signal = self.check_condition(condition)
                        self.logger.info(f"Entry condition {i}: {condition}")
                        self.logger.info(f"Current value: {float(self.indicators[key][-1]):.2f}, Target: {condition['value']}, Signal: {signal}")
                        entry_signals.append(signal)
                    else:
                        self.logger.warning(f"Missing indicator data for {key}")
                        entry_signals.append(False)
                
                # Execute entry if all conditions met
                if all(entry_signals):
                    self.logger.info("All entry conditions met! Calculating position size...")
                    
                    # Calculate position size
                    cash = self.equity * self.risk_management['max_position_size']
                    size = int(cash / current_price)
                    
                    self.logger.info(f"Available cash: {cash:.2f}")
                    self.logger.info(f"Calculated position size: {size}")
                    
                    if size > 0:
                        # Execute buy order
                        self.buy(size=size)
                        self.entry_price = current_price
                        
                        # Log trade
                        trade = {
                            'date': self.data.index[-1],
                            'action': 'BUY',
                            'price': current_price,
                            'size': size,
                            'cost': size * current_price,
                            'indicators': {key: float(self.indicators[key][-1]) for key in self.indicators if self.indicators[key] is not None}
                        }
                        AlgoTradingStrategy.trade_log.append(trade)
                        self.logger.info(f"BUY EXECUTED: {trade}")
                    else:
                        self.logger.info("Position size too small, skipping trade")
                else:
                    self.logger.info("Not all entry conditions met, no trade executed")
            
            # Exit logic - have position
            elif self.position and self.entry_price is not None:
                self.logger.info("\nChecking exit conditions for existing position:")
                
                # Calculate current return
                current_return = (current_price - self.entry_price) / self.entry_price
                self.logger.info(f"Current return: {current_return:.2%}")
                
                # Check stop loss and take profit
                if (current_return <= -self.risk_management['stop_loss'] or 
                    current_return >= self.risk_management['take_profit']):
                    
                    # Log which threshold was hit
                    if current_return <= -self.risk_management['stop_loss']:
                        self.logger.info("Stop loss triggered!")
                    else:
                        self.logger.info("Take profit triggered!")
                    
                    # Close position
                    self.position.close()
                    
                    # Log trade
                    trade = {
                        'date': self.data.index[-1],
                        'action': 'SELL',
                        'price': current_price,
                        'size': self.position.size,
                        'profit': (current_price - self.entry_price) * self.position.size,
                        'indicators': {key: float(self.indicators[key][-1]) for key in self.indicators if self.indicators[key] is not None}
                    }
                    AlgoTradingStrategy.trade_log.append(trade)
                    self.logger.info(f"SELL EXECUTED (SL/TP): {trade}")
                    self.entry_price = None
                    return
                
                # Check exit conditions
                self.logger.info("Checking technical exit conditions:")
                exit_signals = []
                
                # Check each exit condition
                for i, condition in enumerate(self.exit_conditions, 1):
                    key = f"{condition['indicator']}_{condition['timeframe']}"
                    if key in self.indicators and self.indicators[key] is not None:
                        signal = self.check_condition(condition)
                        self.logger.info(f"Exit condition {i}: {condition}")
                        self.logger.info(f"Current value: {float(self.indicators[key][-1]):.2f}, Target: {condition['value']}, Signal: {signal}")
                        exit_signals.append(signal)
                    else:
                        self.logger.warning(f"Missing indicator data for {key}")
                        exit_signals.append(False)
                
                # Execute exit if any condition met
                if any(exit_signals):
                    self.logger.info("Exit conditions met! Closing position...")
                    self.position.close()
                    
                    # Log trade
                    trade = {
                        'date': self.data.index[-1],
                        'action': 'SELL',
                        'price': current_price,
                        'size': self.position.size,
                        'profit': (current_price - self.entry_price) * self.position.size,
                        'indicators': {key: float(self.indicators[key][-1]) for key in self.indicators if self.indicators[key] is not None}
                    }
                    AlgoTradingStrategy.trade_log.append(trade)
                    self.logger.info(f"SELL EXECUTED (Technical): {trade}")
                    self.entry_price = None
                else:
                    self.logger.info("No exit conditions met, holding position")
                    
        except Exception as e:
            self.logger.error(f"Error in next(): {e}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            return

    def _log_indicator_values(self):
        """Log indicator values to verify calculations"""
        try:
            logger.info("\nCurrent Indicator Values:")
            for name, values in self.indicators.items():
                if len(values) > 0:
                    # Get both current and previous values to verify updates
                    logger.info(f"{name}: Current = {values[-1]}, Previous = {values[-2]}")
                else:
                    logger.warning(f"{name}: No values calculated")
                    
            # Also log current price
            logger.info(f"Current Close: {self.close_series[-1]}")
            logger.info(f"Previous Close: {self.close_series[-2]}")
            
        except Exception as e:
            logger.error(f"Error logging indicator values: {e}")

    @classmethod
    def get_trade_log(cls):
        """Return the trade log as a DataFrame"""
        if not cls.trade_log:
            return pd.DataFrame()
        
        df = pd.DataFrame(cls.trade_log)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].round(2)
        return df
    

class BacktestRunner:
    def __init__(self):
        # Set up paths
        self.base_path = Path(os.getcwd())
        # Input data directory (where your CSV files are)
        self.input_data_dir = self.base_path / "agents" / "backtesting_agent" / "historical_data"
        # Output directory (where results will be saved)
        self.output_dir = self.base_path / "output" / "backtest_results"
        # Algorithm directory (where algorithm configurations are saved)
        self.algo_dir = self.base_path / "output" / "algo"  
        
        # logger.info(f"Input data directory: {self.input_data_dir}")
        # logger.info(f"Output directory: {self.output_dir}")

    def get_latest_algorithm(self, company_name: str) -> dict:
        """Get the latest algorithm configuration for the company"""
        try:
            algo_dir = self.base_path / "output" / "algo"
            pattern = f"{company_name}_algorithm-*.json"
            algo_files = list(algo_dir.glob(pattern))
            
            if not algo_files:
                raise FileNotFoundError(f"No algorithm files found for {company_name}")
            
            # Extract numbers from filenames and find the latest
            latest_file = max(algo_files, key=lambda x: 
                int(re.search(r'-(\d+)\.json$', x.name).group(1)))
            
            logger.info(f"Loading algorithm from: {latest_file}")
            
            with open(latest_file, 'r') as f:
                algo_config = json.load(f)
                
            # Validate required fields
            required_fields = ['indicators', 'entry_conditions', 'exit_conditions', 
                            'risk_management', 'trading_hours']
            
            missing_fields = [field for field in required_fields 
                            if field not in algo_config]
            
            if missing_fields:
                raise ValueError(f"Missing required fields in algorithm config: {missing_fields}")
                
            return algo_config
                
        except Exception as e:
            logger.error(f"Error loading algorithm configuration: {e}")
            raise

    def load_data(self, company_name: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load and prepare historical data with proper timezone handling"""
        try:
            # Load data
            file_path = self.input_data_dir / f"{company_name}_minute.csv"
            logger.info(f"Loading data from: {file_path}")
            
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            # Read CSV with datetime parsing
            df = pd.read_csv(file_path)
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df = df.set_index('Datetime')
            
            # Filter date range if provided
            if start_date:
                start_dt = pd.to_datetime(start_date)
                if not start_dt.tzinfo:
                    start_dt = start_dt.tz_localize('Asia/Kolkata')
                df = df[df.index >= start_dt]
                
            if end_date:
                end_dt = pd.to_datetime(end_date)
                if not end_dt.tzinfo:
                    end_dt = end_dt.tz_localize('Asia/Kolkata')
                df = df[df.index <= end_dt]
            
            # Sort index
            df = df.sort_index()
            
            # Ensure numeric columns
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Forward fill any missing values
            df = df.ffill()
            
            # Log sample data
            logger.info(f"Loaded {len(df)} records from {df.index[0]} to {df.index[-1]}")
            logger.info(f"Sample data at start:\n{df.head().to_string()}")
            logger.info(f"Sample data at end:\n{df.tail().to_string()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def run_backtest(self, company_name: str, start_date: str = None, end_date: str = None) -> dict:
        """Run backtest with proper trade log handling"""
        try:
            # Setup logging for this backtest run
            self.logger = setup_logging(company_name)
            self.logger.info(f"Starting backtest for {company_name}")
            self.logger.info(f"Input data directory: {self.input_data_dir}")
            self.logger.info(f"Output directory: {self.output_dir}")

            # Setup directories
            self.setup_result_directories(company_name)
            
            # Load and validate data
            data = self.load_data(company_name, start_date, end_date)
            if len(data) < 100:
                raise ValueError(f"Insufficient data points: {len(data)}")
            
            # Initialize and run backtest
            bt = Backtest(
                data,
                AlgoTradingStrategy,
                cash=100000,
                commission=0.002,
                exclusive_orders=True
            )
            
            # Run backtest
            self.logger.info("Starting backtest execution...")
            stats = bt.run()
            self.logger.info("Backtest execution completed")
            
            # Get trade log using class method
            trades_df = AlgoTradingStrategy.get_trade_log()
            
            # Plot results
            try:
                # Create interactive plot
                self.create_interactive_plot(data, trades_df, company_name)
                
                # Create backtesting plot
                plot_file = self.plots_dir / f"{company_name}_backtest.html"
                bt.plot(filename=str(plot_file), open_browser=False)
                
                self.logger.info("Plots created successfully")
            except Exception as plot_error:
                self.logger.error(f"Error creating plots: {plot_error}")
            
            # Save results
            self.save_results(stats, trades_df, company_name)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
            raise


    def create_interactive_plot(self, df: pd.DataFrame, trades_df: pd.DataFrame, company_name: str):
            """Create an interactive HTML plot using Plotly"""
            try:
                # Create figure with secondary y-axis
                fig = make_subplots(rows=4, cols=1, 
                                shared_xaxes=True,
                                vertical_spacing=0.02,
                                row_heights=[0.5, 0.2, 0.15, 0.15])

                # Add candlestick chart
                fig.add_trace(go.Candlestick(x=df.index,
                                            open=df['Open'],
                                            high=df['High'],
                                            low=df['Low'],
                                            close=df['Close'],
                                            name='Price'),
                            row=1, col=1)

                # Get unique indicators from conditions
                indicators = set()
                algo_config = self.get_latest_algorithm(company_name)
                for condition in algo_config['entry_conditions'] + algo_config['exit_conditions']:
                    indicators.add(condition['indicator'])

                # Plot indicators
                colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
                color_idx = 0
                
                for indicator in indicators:
                    if indicator == 'SMA':
                        sma = talib.SMA(df['Close'], timeperiod=20)
                        fig.add_trace(go.Scatter(
                            x=df.index, y=sma,
                            name='SMA(20)',
                            line=dict(color=colors[color_idx % len(colors)])
                        ), row=1, col=1)
                        color_idx += 1
                    
                    elif indicator == 'RSI':
                        rsi = talib.RSI(df['Close'], timeperiod=14)
                        fig.add_trace(go.Scatter(
                            x=df.index, y=rsi,
                            name='RSI(14)',
                            line=dict(color=colors[color_idx % len(colors)])
                        ), row=2, col=1)
                        # Add RSI levels
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2)
                        color_idx += 1
                    
                    elif indicator == 'WILLR':
                        willr = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
                        fig.add_trace(go.Scatter(
                            x=df.index, y=willr,
                            name='Williams %R(14)',
                            line=dict(color=colors[color_idx % len(colors)])
                        ), row=3, col=1)
                        # Add Williams %R levels
                        fig.add_hline(y=-20, line_dash="dash", line_color="red", row=3)
                        fig.add_hline(y=-80, line_dash="dash", line_color="green", row=3)
                        color_idx += 1

                    elif indicator == 'MACD':
                        macd, signal, hist = talib.MACD(df['Close'],
                                                    fastperiod=12,
                                                    slowperiod=26,
                                                    signalperiod=9)
                        fig.add_trace(go.Scatter(
                            x=df.index, y=macd,
                            name='MACD',
                            line=dict(color=colors[color_idx % len(colors)])
                        ), row=4, col=1)
                        fig.add_trace(go.Scatter(
                            x=df.index, y=signal,
                            name='Signal',
                            line=dict(color=colors[(color_idx + 1) % len(colors)])
                        ), row=4, col=1)
                        # Add MACD histogram
                        fig.add_trace(go.Bar(
                            x=df.index, y=hist,
                            name='MACD Histogram',
                            marker_color=np.where(hist > 0, 'green', 'red')
                        ), row=4, col=1)
                        color_idx += 2

                # Add volume bars
                colors = ['red' if row['Open'] > row['Close'] else 'green' 
                        for idx, row in df.iterrows()]
                fig.add_trace(go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color=colors
                ), row=2, col=1)

                # Add trades if available
                if not trades_df.empty:
                    # Add buy signals
                    buy_trades = trades_df[trades_df['action'] == 'BUY']
                    if not buy_trades.empty:
                        fig.add_trace(go.Scatter(
                            x=buy_trades['date'],
                            y=[df.loc[date, 'Low'] * 0.99 for date in buy_trades['date']],
                            mode='markers',
                            name='Buy',
                            marker=dict(
                                symbol='triangle-up',
                                size=10,
                                color='green',
                            )
                        ), row=1, col=1)

                    # Add sell signals
                    sell_trades = trades_df[trades_df['action'] == 'SELL']
                    if not sell_trades.empty:
                        fig.add_trace(go.Scatter(
                            x=sell_trades['date'],
                            y=[df.loc[date, 'High'] * 1.01 for date in sell_trades['date']],
                            mode='markers',
                            name='Sell',
                            marker=dict(
                                symbol='triangle-down',
                                size=10,
                                color='red',
                            )
                        ), row=1, col=1)

                # Update layout
                fig.update_layout(
                    title=f'{company_name} Trading Chart',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark',
                    height=1200,
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )

                # Update axes labels
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="Volume", row=2, col=1)
                
                if 'RSI' in indicators:
                    fig.update_yaxes(title_text="RSI", row=2, col=1)
                if 'MACD' in indicators:
                    fig.update_yaxes(title_text="MACD", row=4, col=1)
                if 'WILLR' in indicators:
                    fig.update_yaxes(title_text="Williams %R", row=3, col=1)

                # Save the figure
                plot_path = self.plots_dir / f"{company_name}_interactive_chart.html"
                fig.write_html(str(plot_path))
                logger.info(f"Saved interactive plot to {plot_path}")

            except Exception as e:
                logger.error(f"Error creating interactive plot: {e}")
                raise

    
    def get_next_result_number(self, company_name: str) -> int:
        """Get the next available result number"""
        pattern = str(self.output_dir / f"{company_name}_result-*")
        existing_dirs = glob.glob(pattern)
        if not existing_dirs:
            return 1
        
        numbers = []
        for dir_path in existing_dirs:
            match = re.search(rf"{company_name}_result-(\d+)$", dir_path)
            if match:
                numbers.append(int(match.group(1)))
        
        return max(numbers) + 1 if numbers else 1

    def setup_result_directories(self, company_name: str) -> None:
        """Setup directory structure for results"""
        # Get next result number
        result_num = self.get_next_result_number(company_name)
        
        # Create main result directory
        self.run_dir = self.output_dir / f"{company_name}_result-{result_num}"
        
        # Create subdirectories
        self.plots_dir = self.run_dir / "plots"
        self.data_dir = self.run_dir / "data"
        self.stats_dir = self.run_dir / "stats"
        
        # Create all directories
        for directory in [self.run_dir, self.plots_dir, self.data_dir, self.stats_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created result directory: {self.run_dir}")

    def save_results(self, stats: dict, trades_df: pd.DataFrame, company_name: str):
        """Save all results to organized directories with enhanced trade logging"""
        try:
            # Ensure all directories exist
            for dir_path in [self.stats_dir, self.data_dir, self.plots_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Save statistics
            stats_dict = {}
            for key, value in stats.items():
                try:
                    if isinstance(value, (pd.Series, pd.DataFrame)):
                        continue
                    elif isinstance(value, (np.integer, np.floating, np.bool_)):
                        stats_dict[key] = value.item()
                    elif isinstance(value, pd.Timedelta):
                        stats_dict[key] = str(value)
                    else:
                        stats_dict[key] = value
                except Exception as e:
                    logger.warning(f"Skipping stat {key} due to: {str(e)}")
            
            # Save statistics
            stats_file = self.stats_dir / f"{company_name}_stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_dict, f, indent=4, cls=NumpyEncoder)
            logger.info(f"Saved statistics to: {stats_file}")

            # Save trade log with enhanced error handling
            if isinstance(trades_df, pd.DataFrame):  # Remove the empty check
                try:
                    # Create a copy to avoid modifying the original
                    trades_df_copy = trades_df.copy()
                    
                    # Ensure date column is properly formatted
                    if 'date' in trades_df_copy.columns:
                        trades_df_copy['date'] = pd.to_datetime(trades_df_copy['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Round numeric columns
                    numeric_columns = trades_df_copy.select_dtypes(include=[np.number]).columns
                    trades_df_copy[numeric_columns] = trades_df_copy[numeric_columns].round(2)
                    
                    # Save to CSV
                    trades_file = self.data_dir / f"{company_name}_trades.csv"
                    trades_df_copy.to_csv(trades_file, index=False)
                    logger.info(f"Successfully saved trade log to: {trades_file}")
                    
                    # Also save as JSON for backup
                    trades_json = self.data_dir / f"{company_name}_trades.json"
                    trades_df_copy.to_json(trades_json, orient='records', indent=4, date_format='iso')
                    logger.info(f"Successfully saved trade log backup to: {trades_json}")
                
                except Exception as e:
                    logger.error(f"Error saving trade log: {e}")
                    # Attempt basic save as fallback
                    trades_file = self.data_dir / f"{company_name}_trades.csv"
                    trades_df.to_csv(trades_file, index=False)
            else:
                logger.warning("Invalid trades DataFrame")

            # Save parameters
            params = {
                "entry_conditions": {
                    "rsi_threshold": 30,
                    "macd_condition": "crossover",
                    "trend_condition": "price > SMA20"
                },
                "exit_conditions": {
                    "rsi_threshold": 70,
                    "macd_condition": "crossunder",
                    "trend_condition": "price < EMA50"
                },
                "position_sizing": "fixed_100_shares",
                "commission_rate": 0.002,
                "performance_summary": {
                    "total_return": float(stats_dict.get('Return [%]', 0)),
                    "sharpe_ratio": float(stats_dict.get('Sharpe Ratio', 0)),
                    "max_drawdown": float(stats_dict.get('Max. Drawdown [%]', 0)),
                    "win_rate": float(stats_dict.get('Win Rate [%]', 0)),
                    "total_trades": int(stats_dict.get('# Trades', 0))
                }
            }
            
            params_file = self.stats_dir / f"{company_name}_parameters.json"
            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=4)
            logger.info(f"Saved parameters to: {params_file}")

            # Generate and save summary report
            report = self.generate_summary_report(stats_dict, trades_df, company_name)
            report_file = self.run_dir / f"{company_name}_summary.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Saved summary report to: {report_file}")

        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
            
    def generate_summary_report(self, stats: dict, trades_df: pd.DataFrame, company_name: str) -> str:
        """Generate a comprehensive summary report"""
        report = []
        report.append(f"Backtest Summary Report - {company_name}")
        report.append("=" * 50)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Performance Summary
        report.append("Performance Summary")
        report.append("-" * 30)
        report.append(f"Total Return: {stats.get('Return [%]', 0):.2f}%")
        report.append(f"Sharpe Ratio: {stats.get('Sharpe Ratio', 0):.2f}")
        report.append(f"Max Drawdown: {stats.get('Max. Drawdown [%]', 0):.2f}%")
        report.append(f"Win Rate: {stats.get('Win Rate [%]', 0):.2f}%")
        report.append(f"Total Trades: {stats.get('# Trades', 0)}")
        report.append("")
        
        # Trade Statistics
        report.append("Trade Statistics")
        report.append("-" * 30)
        report.append(f"Average Trade: {stats.get('Avg. Trade [%]', 0):.2f}%")
        report.append(f"Best Trade: {stats.get('Best Trade [%]', 0):.2f}%")
        report.append(f"Worst Trade: {stats.get('Worst Trade [%]', 0):.2f}%")
        report.append(f"Profit Factor: {stats.get('Profit Factor', 0):.2f}")
        
        # Trade Analysis
        if len(trades_df) > 0:
            report.append("\nTrade Analysis")
            report.append("-" * 30)
            report.append(f"Total Number of Trades: {len(trades_df)}")
            buys = trades_df[trades_df['action'] == 'BUY']
            sells = trades_df[trades_df['action'] == 'SELL']
            report.append(f"Number of Buy Trades: {len(buys)}")
            report.append(f"Number of Sell Trades: {len(sells)}")
        
        # Directory Structure
        report.append("\nResults Directory Structure")
        report.append("-" * 30)
        report.append(f"Main Directory: {self.run_dir}")
        report.append(f"├── plots/")
        report.append(f"│   └── {company_name}_backtest.html")
        report.append(f"├── data/")
        report.append(f"│   └── {company_name}_trades.csv")
        report.append(f"├── stats/")
        report.append(f"│   ├── {company_name}_stats.json")
        report.append(f"│   └── {company_name}_parameters.json")
        report.append(f"└── {company_name}_summary.txt")
        
        return "\n".join(report)


    def plot_results(self, df: pd.DataFrame, stats: dict, company_name: str):
        """Create manual plots using matplotlib"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            # Create figure
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            
            # Calculate indicators
            sma20 = talib.SMA(df['Close'], timeperiod=20)
            ema50 = talib.EMA(df['Close'], timeperiod=50)
            rsi = talib.RSI(df['Close'], timeperiod=14)
            macd, signal, hist = talib.MACD(df['Close'], 
                                          fastperiod=12, 
                                          slowperiod=26, 
                                          signalperiod=9)

            # Plot price and moving averages
            ax1.plot(df.index, df['Close'], label='Close', color='blue', alpha=0.7)
            ax1.plot(df.index, sma20, label='SMA20', color='orange', alpha=0.7)
            ax1.plot(df.index, ema50, label='EMA50', color='red', alpha=0.7)
            ax1.set_title(f'{company_name} - Price Action')
            ax1.legend()
            ax1.grid(True)
            
            # Plot RSI
            ax2.plot(df.index, rsi, label='RSI', color='purple')
            ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax2.fill_between(df.index, 70, 100, color='r', alpha=0.1)
            ax2.fill_between(df.index, 0, 30, color='g', alpha=0.1)
            ax2.set_title('RSI')
            ax2.set_ylim(0, 100)
            ax2.grid(True)
            
            # Plot MACD
            ax3.plot(df.index, macd, label='MACD', color='blue')
            ax3.plot(df.index, signal, label='Signal', color='red')
            ax3.bar(df.index, hist, label='Histogram',
                   color=['red' if h < 0 else 'green' for h in hist], alpha=0.3)
            ax3.set_title('MACD')
            ax3.grid(True)
            ax3.legend()

            # Format x-axis
            for ax in [ax1, ax2, ax3]:
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot
            plot_path = self.plots_dir / f"{company_name}_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved analysis plot to {plot_path}")
            
        except Exception as e:
            logger.warning(f"Error creating manual plots: {e}")



if __name__ == "__main__":
    try:
        # Debug logging setup
        import os
        from pathlib import Path
        
        # Get absolute path
        root_path = Path(os.getcwd())
        log_dir = root_path / "logs"
        
        # Create logs directory if it doesn't exist
        log_dir.mkdir(exist_ok=True)
        
        print(f"Root path: {root_path}")
        print(f"Log directory: {log_dir}")
        print(f"Log directory exists: {log_dir.exists()}")
        print(f"Log directory contents: {list(log_dir.glob('*'))}")
        
        # Company configuration
        COMPANY_NAME = "ZOMATO"
        
        # Initialize and run agent
        runner = BacktestRunner()
        
        # Use proper datetime format matching your CSV
        start_date = "2023-01-01 09:15:00+05:30"
        end_date = "2024-11-09 15:30:00+05:30"
        
        stats = runner.run_backtest(COMPANY_NAME, start_date, end_date)
        format_stats(stats)
        
    except Exception as e:
        print(f"Error: {e}")
        raise