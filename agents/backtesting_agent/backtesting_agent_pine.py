import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from backtesting import Backtest, Strategy
import logging
from datetime import datetime
from pathlib import Path
import talib
import glob
import re
from typing import Dict, List, Optional, Union
from typing import Tuple
from agents.backtesting_agent.technical_indicators import TechnicalIndicators


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PineScriptStrategy(Strategy):
    """Strategy class that implements Pine Script logic"""

    # Add class-level attribute
    pine_params = None
    def init(self):
        """Initialize strategy with Pine Script parameters and indicators"""
        if self.pine_params is None:
            raise ValueError("Pine Script parameters not set")
        
        # Validate data
        if isinstance(self.data.index, pd.DatetimeIndex):
            if self.data.index.isnull().any():
                logger.warning("Found NaT values in datetime index")
        
        # Check for NaN values in OHLCV data
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if pd.isna(self.data[col]).any():
                logger.warning(f"Found NaN values in {col} data")
        
        self.indicators = {}
        self.pine_variables = {}
        self.last_trade_date = None
        self.daily_trades = []
        
        # Parse and initialize Pine Script
        self.initialize_pine_script()

    def initialize_pine_script(self):
            """Initialize indicators and variables from Pine Script"""
            try:
                if not hasattr(self, 'pine_params'):
                    raise ValueError("No Pine Script parameters defined")
                
                if not isinstance(self.pine_params, dict):
                    raise ValueError("Invalid Pine Script parameters format")
                
                required_keys = ['inputs', 'indicators', 'long_entry', 'long_exit', 
                            'short_entry', 'short_exit', 'risk_params']
                
                missing_keys = [key for key in required_keys if key not in self.pine_params]
                if missing_keys:
                    raise ValueError(f"Missing required parameters: {missing_keys}")
                
                # Extract input parameters
                for param in self.pine_params.get('inputs', []):
                    if not all(key in param for key in ['name', 'default']):
                        raise ValueError(f"Invalid input parameter format: {param}")
                    setattr(self, param['name'], param['default'])
                
                # Initialize indicators
                for indicator in self.pine_params.get('indicators', []):
                    if not all(key in indicator for key in ['id', 'type']):
                        raise ValueError(f"Invalid indicator format: {indicator}")
                    self.initialize_indicator(indicator)
                    
            except Exception as e:
                logger.error(f"Error initializing Pine Script: {e}")
                raise

    def initialize_indicator(self, indicator: Dict):
        """Initialize a technical indicator from Pine Script definition"""
        try:
            close = self.data.Close.astype(float)
            high = self.data.High.astype(float)
            low = self.data.Low.astype(float)
            volume = self.data.Volume.astype(float)
            
            indicator_type = indicator['type'].lower()
            params = indicator.get('params', {})
            
            # Extended indicator mappings including new indicators
            indicator_mappings = {
                'sma': lambda: self.I(talib.SMA, close, timeperiod=params.get('length', 20)),
                'ema': lambda: self.I(talib.EMA, close, timeperiod=params.get('length', 20)),
                'vwap': lambda: self.I(TechnicalIndicators.vwap, high, low, close, volume),
                'rsi': lambda: self.I(talib.RSI, close, timeperiod=params.get('length', 14)),
                'macd': lambda: self._initialize_macd(indicator['id'], close, params),
                'stoch': lambda: self._initialize_stochastic(indicator['id'], high, low, close, params),
                'bb': lambda: self._initialize_bbands(indicator['id'], close, params),
                'atr': lambda: self.I(talib.ATR, high, low, close, timeperiod=params.get('length', 14)),
                'obv': lambda: self.I(talib.OBV, close, volume),
                'mfi': lambda: self.I(talib.MFI, high, low, close, volume, timeperiod=params.get('length', 14)),
                'adx': lambda: self.I(talib.ADX, high, low, close, timeperiod=params.get('length', 14)),
                'supertrend': lambda: self._initialize_supertrend(indicator['id'], high, low, close, params),
            }

            
            if indicator_type in indicator_mappings:
                self.indicators[indicator['id']] = indicator_mappings[indicator_type]()
            else:
                logger.warning(f"Unsupported indicator type: {indicator_type}")
                
        except Exception as e:
            logger.error(f"Error initializing indicator {indicator['type']}: {e}")
            raise

    def _initialize_macd(self, indicator_id: str, close: pd.Series, params: dict):
        """Helper method for MACD initialization"""
        macd, signal, hist = talib.MACD(close, 
                                    fastperiod=params.get('fast_length', 12),
                                    slowperiod=params.get('slow_length', 26),
                                    signalperiod=params.get('signal_length', 9))
        self.indicators[f"{indicator_id}_line"] = self.I(lambda: macd)
        self.indicators[f"{indicator_id}_signal"] = self.I(lambda: signal)
        self.indicators[f"{indicator_id}_hist"] = self.I(lambda: hist)
        return macd

    def _initialize_bbands(self, indicator_id: str, close: pd.Series, params: dict):
        """Helper method for Bollinger Bands initialization"""
        upper, middle, lower = talib.BBANDS(close, 
                                        timeperiod=params.get('length', 20),
                                        nbdevup=params.get('mult', 2),
                                        nbdevdn=params.get('mult', 2))
        self.indicators[f"{indicator_id}_upper"] = self.I(lambda: upper)
        self.indicators[f"{indicator_id}_middle"] = self.I(lambda: middle)
        self.indicators[f"{indicator_id}_lower"] = self.I(lambda: lower)
        return middle

    def _initialize_stochastic(self, indicator_id: str, high: pd.Series, low: pd.Series, 
                            close: pd.Series, params: dict):
        """Helper method for Stochastic initialization"""
        k, d = TechnicalIndicators.stochastic(
            high, low, close,
            k_period=params.get('k_period', 14),
            d_period=params.get('d_period', 3),
            slowing=params.get('slowing', 3)
        )
        self.indicators[f"{indicator_id}_k"] = self.I(lambda: k)
        self.indicators[f"{indicator_id}_d"] = self.I(lambda: d)
        return k

    def _initialize_supertrend(self, indicator_id: str, high: pd.Series, low: pd.Series, 
                            close: pd.Series, params: dict):
        """Helper method for SuperTrend initialization"""
        supertrend, direction = TechnicalIndicators.supertrend(
            high, low, close,
            period=params.get('period', 10),
            multiplier=params.get('multiplier', 3)
        )
        self.indicators[f"{indicator_id}_direction"] = self.I(lambda: direction)
        return self.I(lambda: supertrend)

    def _initialize_vwap(self, high, low, close, volume):
        """Helper method for VWAP calculation"""
        typical_price = (high + low + close) / 3
        return self.I(lambda: (typical_price * volume).cumsum() / volume.cumsum())

    def evaluate_pine_condition(self, condition: Dict) -> bool:
        """Evaluate a Pine Script condition"""
        try:
            condition_type = condition['type']
            
            if condition_type == 'comparison':
                left = self.get_value(condition['left'])
                right = self.get_value(condition['right'])
                operator = condition['operator']
                
                if operator == '>':
                    return left > right
                elif operator == '<':
                    return left < right
                elif operator == '>=':
                    return left >= right
                elif operator == '<=':
                    return left <= right
                elif operator == '==':
                    return left == right
            
            elif condition_type == 'and':
                return all(self.evaluate_pine_condition(cond) for cond in condition['conditions'])
            
            elif condition_type == 'or':
                return any(self.evaluate_pine_condition(cond) for cond in condition['conditions'])
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating condition: {e}")
            return False

    def get_value(self, source: Union[Dict, float]) -> float:
        """Get value from various Pine Script sources"""
        try:
            if isinstance(source, (int, float)):
                return float(source)
            
            source_type = source['type']
            
            if source_type == 'indicator':
                return self.indicators[source['id']][-1]
            elif source_type == 'price':
                if source['field'] == 'close':
                    return self.data.Close[-1]
                elif source['field'] == 'open':
                    return self.data.Open[-1]
                elif source['field'] == 'high':
                    return self.data.High[-1]
                elif source['field'] == 'low':
                    return self.data.Low[-1]
            elif source_type == 'variable':
                return self.pine_variables.get(source['name'], 0)
            
            return 0
            
        except Exception as e:
            logger.error(f"Error getting value: {e}")
            return 0

    def next(self):
        """Execute trading logic based on Pine Script conditions"""
        try:
            current_datetime = self.data.index[-1]
            
            # Skip if current_datetime is NaT
            if pd.isna(current_datetime):
                return
                
            current_date = current_datetime.date()
            
            # Initialize new day tracking
            if self.last_trade_date != current_date:
                self.handle_day_start(current_date)
            
            # Get current position status
            current_position = self.position.size if self.position else 0
            
            # Evaluate entry/exit conditions
            long_entry = self.evaluate_pine_condition(self.pine_params['long_entry'])
            long_exit = self.evaluate_pine_condition(self.pine_params['long_exit'])
            short_entry = self.evaluate_pine_condition(self.pine_params['short_entry'])
            short_exit = self.evaluate_pine_condition(self.pine_params['short_exit'])
            
            # Execute trades
            self.execute_trades(current_position, long_entry, long_exit, short_entry, short_exit)
            
            # Handle end of day - check if it's a valid datetime first
            try:
                if not pd.isna(current_datetime) and current_datetime.time().strftime('%H:%M') == '15:29':
                    self.handle_day_end()
            except AttributeError:
                logger.warning(f"Invalid datetime value encountered: {current_datetime}")
                
        except Exception as e:
            logger.error(f"Error in next(): {e}")
            raise

    def handle_day_start(self, current_date):
        """Handle start of new trading day"""
        if self.position:
            if self.position.size > 0:
                if self.daily_trades:
                    self.daily_trades[-1]['buy_trades'] += 1
            elif self.position.size < 0:
                if self.daily_trades:
                    self.daily_trades[-1]['sell_trades'] += 1
            self.position.close()
        
        self.daily_trades.append({
            'date': current_date,
            'trades': 0,
            'buy_trades': 0,
            'sell_trades': 0,
            'equity': self.equity,
            'start_equity': self.equity,
            'final_equity': self.equity
        })
        self.last_trade_date = current_date

    def execute_trades(self, current_position, long_entry, long_exit, short_entry, short_exit):
        """Execute trades based on signals"""
        # No position - check entries
        if current_position == 0:
            if long_entry:
                self.buy()
                self.daily_trades[-1]['trades'] += 1
                logger.info(f"{self.data.index[-1]}: Opening long position")
            elif short_entry:
                self.sell()
                self.daily_trades[-1]['trades'] += 1
                logger.info(f"{self.data.index[-1]}: Opening short position")
        
        # Long position - check exit
        elif current_position > 0 and long_exit:
            self.position.close()
            self.daily_trades[-1]['buy_trades'] += 1
            logger.info(f"{self.data.index[-1]}: Closing long position")
        
        # Short position - check exit
        elif current_position < 0 and short_exit:
            self.position.close()
            self.daily_trades[-1]['sell_trades'] += 1
            logger.info(f"{self.data.index[-1]}: Closing short position")

    def handle_day_end(self):
        """Handle end of trading day"""
        if self.position:
            if self.position.size > 0:
                self.daily_trades[-1]['buy_trades'] += 1
            elif self.position.size < 0:
                self.daily_trades[-1]['sell_trades'] += 1
            self.position.close()
        
        if self.daily_trades:
            daily_pnl = self.equity - self.daily_trades[-1]['start_equity']
            daily_return = (daily_pnl / self.daily_trades[-1]['start_equity']) * 100
            self.daily_trades[-1]['profit'] = daily_return
            self.daily_trades[-1]['final_equity'] = self.equity

class BacktestingAgent:
    def __init__(self):
        self.base_path = Path(os.getcwd())
        self.output_dir = self.base_path / 'output'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'backtest_results').mkdir(exist_ok=True)

    def get_latest_algorithm(self, company_name: str) -> Tuple[str, int]:
        """Find the latest Pine Script algorithm file."""
        pattern = str(self.output_dir / 'algo' / f"{company_name}_algorithm-*.pine")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No Pine Script files found for {company_name}")
        
        latest_file = max(files, key=lambda x: int(x.split('-')[-1].split('.')[0]))
        algo_num = int(latest_file.split('-')[-1].split('.')[0])
        
        logger.info(f"Using algorithm {algo_num} for {company_name}")
        return latest_file, algo_num

    def extract_inputs(self, pine_script: str) -> List[Dict]:
        """Extract input parameters from Pine Script"""
        inputs = []
        try:
            # Match input.int, input.float, input.bool, etc.
            input_pattern = r'input\.(float|int|bool|string)\s*\(\s*["\']([^"\']+)["\']\s*,\s*["\']([^"\']+)["\']\s*,\s*([^,\)]+)'
            matches = re.finditer(input_pattern, pine_script)
            
            for match in matches:
                input_type, name, title, default = match.groups()
                inputs.append({
                    'type': input_type,
                    'name': name,
                    'title': title,
                    'default': self.parse_value(default, input_type)
                })
            
            return inputs
            
        except Exception as e:
            logger.error(f"Error extracting inputs: {e}")
            return []

    def parse_value(self, value: str, value_type: str):
        """Parse string value to appropriate type"""
        try:
            value = value.strip()
            if value_type == 'float':
                return float(value)
            elif value_type == 'int':
                return int(value)
            elif value_type == 'bool':
                return value.lower() == 'true'
            return value
        except Exception as e:
            logger.error(f"Error parsing value {value} as {value_type}: {e}")
            return None

    def extract_indicators(self, pine_script: str) -> List[Dict]:
        """Extract technical indicators from Pine Script"""
        indicators = []
        try:
            # Extended indicator patterns
            indicator_patterns = {
                'sma': r'ta\.sma\(([^,]+),\s*(\d+)\)',
                'ema': r'ta\.ema\(([^,]+),\s*(\d+)\)',
                'vwap': r'ta\.vwap\(([^,]+)\)',  # Add VWAP pattern
                'rsi': r'ta\.rsi\(([^,]+),\s*(\d+)\)',
                'macd': r'ta\.macd\(([^,]+),\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)',
                'stoch': r'ta\.stoch(?:astic)?\(([^,]+),\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)',  # Add Stochastic pattern
                'bb': r'ta\.bb(?:ands)?\(([^,]+),\s*(\d+)\s*,\s*(\d+)\)',
                'atr': r'ta\.atr\(([^,]+),\s*(\d+)\)',  # Add ATR pattern
                'obv': r'ta\.obv\(([^,]+)\)',  # Add OBV pattern
                'mfi': r'ta\.mfi\(([^,]+),\s*(\d+)\)',  # Add MFI pattern
                'adx': r'ta\.adx\(([^,]+),\s*(\d+)\)',  # Add ADX pattern
                'supertrend': r'ta\.supertrend\(([^,]+),\s*(\d+)\s*,\s*(\d+)\)'  # Add Supertrend pattern
            }
            
            for ind_type, pattern in indicator_patterns.items():
                matches = re.finditer(pattern, pine_script)
                for i, match in enumerate(matches):
                    indicator = {
                        'id': f"{ind_type}_{i}",
                        'type': ind_type,
                        'params': {}
                    }
                    
                    if ind_type == 'stoch':
                        source, k_period, d_period, slowing = match.groups()
                        indicator['params'] = {
                            'source': source.strip(),
                            'k_period': int(k_period),
                            'd_period': int(d_period),
                            'slowing': int(slowing)
                        }
                    elif ind_type == 'supertrend':
                        source, period, multiplier = match.groups()
                        indicator['params'] = {
                            'source': source.strip(),
                            'period': int(period),
                            'multiplier': float(multiplier)
                        }
                    elif ind_type in ['vwap', 'obv']:  # For indicators without parameters
                        source = match.groups()[0]
                        indicator['params'] = {
                            'source': source.strip()
                        }
                    indicators.append(indicator)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error extracting indicators: {e}")
            return []

    def extract_conditions(self, pine_script: str, trigger: str) -> Dict:
        """Extract trading conditions from Pine Script"""
        try:
            # Fix the escape sequence and make pattern more robust
            pattern = rf'{re.escape(trigger)}[^)]+?when\s*=\s*([^,\)]+)'
            match = re.search(pattern, pine_script)
            
            if not match:
                logger.warning(f"No conditions found for trigger: {trigger}")
                return {'type': 'and', 'conditions': []}
            
            condition_str = match.group(1)
            return self.parse_condition(condition_str)
            
        except Exception as e:
            logger.error(f"Error extracting conditions for {trigger}: {e}")
            return {'type': 'and', 'conditions': []}
    

    def parse_condition(self, condition_str: str) -> Dict:
        """Parse a Pine Script condition string into a condition object"""
        try:
            condition_str = condition_str.strip()
            
            # Handle more Pine Script operators
            operators = {
                '==': 'equals',
                '!=': 'not_equals',
                '>': 'greater',
                '>=': 'greater_equals',
                '<': 'less',
                '<=': 'less_equals',
                'crosses above': 'crossover',
                'crosses below': 'crossunder',
                'crosses': 'cross'
            }
            
            for op_str, op_type in operators.items():
                if op_str in condition_str:
                    parts = condition_str.split(op_str)
                    if len(parts) == 2:
                        return {
                            'type': 'comparison',
                            'operator': op_type,
                            'left': self.parse_operand(parts[0].strip()),
                            'right': self.parse_operand(parts[1].strip())
                        }
            
            # Handle compound conditions
            if ' and ' in condition_str.lower():
                parts = condition_str.lower().split(' and ')
                return {
                    'type': 'and',
                    'conditions': [self.parse_condition(p) for p in parts]
                }
            
            if ' or ' in condition_str.lower():
                parts = condition_str.lower().split(' or ')
                return {
                    'type': 'or',
                    'conditions': [self.parse_condition(p) for p in parts]
                }
                
            return {'type': 'literal', 'value': condition_str.strip()}
            
        except Exception as e:
            logger.error(f"Error parsing condition {condition_str}: {e}")
            return {'type': 'literal', 'value': 'true'}

    def parse_operand(self, operand: str) -> Union[Dict, float]:
        """Parse a Pine Script operand into a value or reference"""
        try:
            # Try to parse as number
            try:
                return float(operand)
            except ValueError:
                pass
            
            # Check for indicator reference
            if operand.startswith('ta.'):
                return {
                    'type': 'indicator',
                    'id': operand.replace('ta.', '').lower()
                }
            
            # Check for price reference
            if operand in ['close', 'open', 'high', 'low']:
                return {
                    'type': 'price',
                    'field': operand
                }
            
            # Assume it's a variable
            return {
                'type': 'variable',
                'name': operand
            }
            
        except Exception as e:
            logger.error(f"Error parsing operand {operand}: {e}")
            return 0.0

    def parse_pine_script(self, pine_file: str) -> Dict:
        """Parse Pine Script file into strategy parameters"""
        try:
            with open(pine_file, 'r') as f:
                pine_script = f.read()
            
            # Parse strategy parameters
            pine_params = {
                'inputs': self.extract_inputs(pine_script),
                'indicators': self.extract_indicators(pine_script),
                'long_entry': self.extract_conditions(pine_script, 'strategy.entry("Long"') or 
                            self.extract_conditions(pine_script, 'strategy.entry("long"'),
                'long_exit': self.extract_conditions(pine_script, 'strategy.close("Long"') or 
                            self.extract_conditions(pine_script, 'strategy.close("long"'),
                'short_entry': self.extract_conditions(pine_script, 'strategy.entry("Short"') or 
                            self.extract_conditions(pine_script, 'strategy.entry("short"'),
                'short_exit': self.extract_conditions(pine_script, 'strategy.close("Short"') or 
                            self.extract_conditions(pine_script, 'strategy.close("short"'),
                'risk_params': self.extract_risk_params(pine_script)
            }
            
            # Validate parsed parameters
            if not any([pine_params['indicators'], 
                    pine_params['long_entry']['conditions'],
                    pine_params['short_entry']['conditions']]):
                logger.warning("No trading conditions or indicators found in Pine Script")
            
            # Log parsed parameters
            logger.info(f"Parsed Pine Script parameters: {json.dumps(pine_params, indent=2)}")
            
            return pine_params
            
        except Exception as e:
            logger.error(f"Error parsing Pine Script: {e}")
            raise

    def extract_risk_params(self, pine_script: str) -> Dict:
        """Extract risk management parameters from Pine Script"""
        risk_params = {}
        try:
            # Extract stop loss
            sl_pattern = r'stop\s*=\s*(\d+(\.\d+)?)'
            sl_match = re.search(sl_pattern, pine_script)
            if sl_match:
                risk_params['stop_loss'] = float(sl_match.group(1))
            
            # Extract take profit
            tp_pattern = r'limit\s*=\s*(\d+(\.\d+)?)'
            tp_match = re.search(tp_pattern, pine_script)
            if tp_match:
                risk_params['take_profit'] = float(tp_match.group(1))
            
            # Extract position size
            size_pattern = r'qty\s*=\s*(\d+(\.\d+)?)'
            size_match = re.search(size_pattern, pine_script)
            if size_match:
                risk_params['position_size'] = float(size_match.group(1))
            
            return risk_params
            
        except Exception as e:
            logger.error(f"Error extracting risk parameters: {e}")
            return {}


    def format_daily_stats(self, trades_df, equity_curve=None, initial_cash=100000):
        """
        Format daily trading statistics with improved calculations and tracking.
        """
        daily_stats = {}
        current_capital = initial_cash
        cumulative_trades = 0
        
        # Convert trades_df to DataFrame if it's not already
        trades_df = pd.DataFrame(trades_df)
        trades_df.index = pd.to_datetime(trades_df.index)
        
        # Calculate running total of profits and trades
        for date, day_trades in trades_df.groupby(trades_df.index.date):
            # Calculate daily metrics
            day_pnl = day_trades['PnL'].sum()
            current_capital += day_pnl
            day_return = (day_pnl / initial_cash) * 100
            
            buy_trades = len(day_trades[day_trades['Size'] > 0])
            sell_trades = len(day_trades[day_trades['Size'] < 0])
            cumulative_trades += (buy_trades + sell_trades)
            
            # Calculate high-water mark
            high_water_mark = max(current_capital, initial_cash)
            drawdown = ((high_water_mark - current_capital) / high_water_mark) * 100
            
            daily_stats[date] = {
                'profit': day_return,
                'total_trades': buy_trades + sell_trades,
                'buy_trades': buy_trades,
                'sell_trades': sell_trades,
                'cumulative_trades': cumulative_trades,
                'pnl': day_pnl,
                'current_capital': current_capital,
                'drawdown': drawdown,
                'win_rate': (len(day_trades[day_trades['PnL'] > 0]) / len(day_trades) * 100) if len(day_trades) > 0 else 0
            }
        
        return daily_stats

    def plot_profits(self, stats, company_name, algo_num):
        """Create and save profit/loss graph with cumulative trades."""
        try:
            # Convert daily_trades list to a proper format for plotting
            daily_data = []
            for trade_day in stats._strategy.daily_trades:
                daily_data.append({
                    'date': trade_day['date'],
                    'profit': trade_day.get('profit', 0),
                    'trades': trade_day['trades'],
                    'buy_trades': trade_day['buy_trades'],
                    'sell_trades': trade_day['sell_trades']
                })
            
            # Convert to DataFrame for easier plotting
            df = pd.DataFrame(daily_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Create figure with two y-axes
            fig, ax1 = plt.subplots(figsize=(15, 8))
            ax2 = ax1.twinx()
            
            # Plot daily profits
            ax1.plot(df['date'], df['profit'], 'b-', marker='o', linewidth=2, label='Daily Profit/Loss')
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Profit/Loss (%)', color='b', fontsize=12)
            
            # Color the profit/loss areas
            ax1.fill_between(df['date'], df['profit'], 0, 
                            where=(df['profit'] >= 0), 
                            color='green', alpha=0.3, label='Profit Area')
            ax1.fill_between(df['date'], df['profit'], 0, 
                            where=(df['profit'] < 0), 
                            color='red', alpha=0.3, label='Loss Area')
            
            # Plot cumulative trades
            df['cumulative_trades'] = df['trades'].cumsum()
            ax2.plot(df['date'], df['cumulative_trades'], 'r--', 
                    linewidth=1.5, label='Cumulative Trades')
            ax2.set_ylabel('Number of Trades', color='r', fontsize=12)
            
            # Set title and adjust layout
            plt.title(f'Daily Trading Performance for {company_name} (Algorithm {algo_num})', 
                    fontsize=14, pad=20)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Format x-axis
            ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            # Add legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Adjust layout and save
            plt.tight_layout()
            output_file = self.output_dir / 'backtest_results' / f"{company_name}_algorithm-{algo_num}_performance.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error in plot_profits: {str(e)}")
            raise


    def save_results(self, stats, company_name, algo_num):
            """Save backtest results with daily breakdown."""
            output_dir = self.output_dir / 'backtest_results'
            output_file = output_dir / f"{company_name}_algorithm-{algo_num}_results.txt"
            
            # Get the Pine Script file content
            pine_file = self.output_dir / 'algo' / f"{company_name}_algorithm-{algo_num}.pine"
            
            with open(output_file, 'w') as f:
                # Write algorithm details
                f.write("Algorithm -\n")
                f.write("Pine Script Strategy:\n")
                with open(pine_file, 'r') as pine:
                    f.write(pine.read())
                f.write("\n\n")
                
                # Write daily results
                f.write("Daily Results:\n")
                daily_stats = {}
                
                # Process trades to get daily statistics
                for trade_day in stats._strategy.daily_trades:
                    date = trade_day['date']
                    if date not in daily_stats:
                        daily_stats[date] = {
                            'profit': trade_day.get('profit', 0),
                            'total_trades': trade_day['trades'],
                            'buy_trades': trade_day['buy_trades'],
                            'sell_trades': trade_day['sell_trades']
                        }
                
                # Write daily results
                for date in sorted(daily_stats.keys(), reverse=True):
                    day_stat = daily_stats[date]
                    f.write(f"\t{date} Profit: {day_stat['profit']:.2f}% ; "
                        f"total trade: {day_stat['total_trades']} ; "
                        f"buy_trade - {day_stat['buy_trades']} ; "
                        f"sell_trade - {day_stat['sell_trades']}\n")
                
                # Write statistics
                f.write("\nOverall Statistics:\n")
                for key, value in stats.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.2f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            
            logger.info(f"Results saved to {output_file}")

    def run_backtest(self, company_name: str, algo_num: int = None) -> Dict:
        """Run backtest using Pine Script strategy"""
        try:
            # Load historical data
            data_file = self.base_path / 'agents' / 'backtesting_agent' / 'historical_data' / f"{company_name}_minute.csv"
            if not data_file.exists():
                raise FileNotFoundError(f"Historical data not found for {company_name}")
            
            df = pd.read_csv(data_file)
            
            # Ensure proper datetime handling and clean data
            df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
            df = df.dropna(subset=['Datetime'])
            
            # Ensure OHLCV columns exist and are numeric
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            for col in required_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort and set index
            df.sort_values('Datetime', inplace=True)
            df.set_index('Datetime', inplace=True)
            
            # Remove any remaining NaN values
            df = df.dropna()
            
            if len(df) == 0:
                
                raise ValueError("No valid data points after cleaning")
            
            
            # Load Pine Script strategy
            logger.info(f"Loaded {len(df)} minute bars for backtesting")
            
            # Load and parse Pine Script
            try:
                pine_file, algo_num = self.get_latest_algorithm(company_name)
                pine_params = self.parse_pine_script(pine_file)
            except Exception as e:
                logger.error(f"Failed to parse Pine Script: {e}")
                raise ValueError("Invalid Pine Script strategy")
            
            if not pine_params:
                raise ValueError("No valid strategy parameters found in Pine Script")
            
            # Create strategy class with Pine Script parameters using class factory
            def strategy_factory(params):
                class CurrentStrategy(PineScriptStrategy):
                    pass
                CurrentStrategy.pine_params = params
                return CurrentStrategy
            
            # Create strategy instance with parameters
            StrategyClass = strategy_factory(pine_params)
            
            # Create and run backtest
            backtest = Backtest(
                df,
                StrategyClass,  # Use the created strategy class
                cash=100000,
                commission=.002,
                exclusive_orders=True,
                hedging=False
            )
            
            stats = backtest.run()
            
            # Save results and plot
            self.save_results(stats, company_name, algo_num)
            self.plot_profits(stats, company_name, algo_num)
            
            return stats
            
        except FileNotFoundError as e:
            logger.error(f"Data file error: {e}")
            raise
        except ValueError as e:
            logger.error(f"Strategy error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during backtesting: {e}")
            raise

if __name__ == "__main__":
    try:
        logger.info("Starting backtesting agent")
        backtest_agent = BacktestingAgent()
        company_name = "ZOMATO"  # or get from config
        stats = backtest_agent.run_backtest(company_name)
        logger.info("Backtesting completed successfully")
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise