
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from backtesting import Backtest, Strategy
import talib
import logging
import json
from pathlib import Path
import glob
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.dates as mdates


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Technical indicator calculations"""
    
    @staticmethod
    def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, 
                      volume: pd.Series) -> pd.Series:
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()

    @staticmethod
    def calculate_supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                           period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
        atr = talib.ATR(high, low, close, timeperiod=period)
        
        # Calculate basic upper and lower bands
        basic_ub = (high + low) / 2 + multiplier * atr
        basic_lb = (high + low) / 2 - multiplier * atr
        
        # Initialize final upper and lower bands
        final_ub = basic_ub.copy()
        final_lb = basic_lb.copy()
        
        # Initialize SuperTrend
        supertrend = pd.Series(0.0, index=close.index)
        direction = pd.Series(1, index=close.index)
        
        for i in range(period, len(close)):
            if close[i] > final_ub[i-1]:
                supertrend[i] = final_lb[i]
                direction[i] = 1
            elif close[i] < final_lb[i-1]:
                supertrend[i] = final_ub[i]
                direction[i] = -1
            else:
                supertrend[i] = supertrend[i-1]
                direction[i] = direction[i-1]
                
                if direction[i] == 1 and final_lb[i] < supertrend[i-1]:
                    supertrend[i] = final_lb[i]
                elif direction[i] == -1 and final_ub[i] > supertrend[i-1]:
                    supertrend[i] = final_ub[i]
        
        return supertrend, direction

class JSONStrategy(Strategy):
    """Strategy implementation based on JSON configuration"""
    
    def init(self):
        """Initialize strategy with debug logging"""
        if not hasattr(self, 'json_config'):
            raise ValueError("Strategy configuration not provided")
        
        logger.info("Initializing strategy...")
        logger.info(f"Strategy config: {json.dumps(self.json_config, indent=2)}")
        
        self.indicators = {}
        self.trades_log = []
        self.daily_stats = {}
        
        # Initialize indicators
        for indicator in self.json_config.get('indicators', []):
            self.initialize_indicator(indicator)
        
        # Log initialized indicators
        logger.info("Initialized indicators:")
        for name, indicator in self.indicators.items():
            if isinstance(indicator, tuple):
                logger.info(f"{name}: tuple of length {len(indicator)}")
            else:
                logger.info(f"{name}: array of shape {np.array(indicator).shape}")
        
        # Initialize position tracking
        self.current_position = 0
        self.entry_price = 0
        self.entry_time = None
        self.stop_loss = None
        self.take_profit = None

    def is_trading_time(self) -> bool:
        """Check if current bar is within trading hours."""
        try:
            trading_hours = self.json_config.get('trading_hours', {
                'start': '09:15',
                'end': '15:20'
            })
            current_time = pd.to_datetime(self.data.index[-1]).time()
            start_time = datetime.strptime(trading_hours['start'], '%H:%M').time()
            end_time = datetime.strptime(trading_hours['end'], '%H:%M').time()
            return start_time <= current_time <= end_time
        except Exception as e:
            logger.error(f"Error checking trading time: {e}")
            return False


    def initialize_indicator(self, indicator_config: Dict):
        """Initialize technical indicators"""
        try:
            indicator_type = indicator_config['type'].lower()
            params = indicator_config.get('params', {})
            name = indicator_config['name']

            # Add basic price data
            if not hasattr(self, 'base_data_added'):
                self.indicators['close'] = self.data.Close
                self.indicators['high'] = self.data.High
                self.indicators['low'] = self.data.Low
                self.indicators['volume'] = self.data.Volume
                self.base_data_added = True

            # MACD special handling
            if indicator_type == 'macd':
                macd, signal, hist = self.I(talib.MACD, self.data.Close,
                                        fastperiod=params.get('fast', 12),
                                        slowperiod=params.get('slow', 26),
                                        signalperiod=params.get('signal', 9))
                self.indicators[name] = macd
                self.indicators[f"{name}_signal"] = signal
                self.indicators[f"{name}_hist"] = hist
                logger.info(f"Initialized MACD indicators: {name}, {name}_signal, {name}_hist")
                return

            # Other indicators
            if indicator_type == 'rsi':
                self.indicators[name] = self.I(talib.RSI, self.data.Close, 
                                            timeperiod=params.get('length', 14))
            elif indicator_type == 'sma':
                self.indicators[name] = self.I(talib.SMA, self.data.Close, 
                                            timeperiod=params.get('length', 20))
            elif indicator_type == 'ema':
                self.indicators[name] = self.I(talib.EMA, self.data.Close, 
                                            timeperiod=params.get('length', 20))

            logger.info(f"Initialized indicator: {name}")

        except Exception as e:
            logger.error(f"Error initializing indicator {indicator_config['type']}: {e}")
            raise

    def initialize_macd(self, params: Dict) -> Tuple:
        """Initialize MACD indicator."""
        macd, signal, hist = talib.MACD(
            self.data.Close,
            fastperiod=params.get('fast', 12),
            slowperiod=params.get('slow', 26),
            signalperiod=params.get('signal', 9)
        )
        return (macd, signal, hist)

    def initialize_bbands(self, params: Dict) -> Tuple:
        """Initialize Bollinger Bands"""
        upper, middle, lower = talib.BBANDS(
            self.data.Close,
            timeperiod=params.get('length', 20),
            nbdevup=params.get('mult', 2),
            nbdevdn=params.get('mult', 2)
        )
        return (upper, middle, lower)

    def initialize_supertrend(self, params: Dict) -> Tuple:
        """Initialize SuperTrend indicator"""
        supertrend, direction = TechnicalIndicators.calculate_supertrend(
            self.data.High,
            self.data.Low,
            self.data.Close,
            period=params.get('period', 10),
            multiplier=params.get('multiplier', 3.0)
        )
        return (supertrend, direction)

    def is_trading_hours(self) -> bool:
        """Check if current time is within trading hours"""
        current_time = self.data.index[-1].time()
        start_time = datetime.strptime(self.trading_hours['start'], '%H:%M').time()
        end_time = datetime.strptime(self.trading_hours['end'], '%H:%M').time()
        return start_time <= current_time <= end_time

    def check_conditions(self, conditions: List[Dict]) -> bool:
        """Evaluate trading conditions with detailed logging."""
        try:
            for condition in conditions:
                ind1_name = condition['indicator1']
                ind2_name = condition['indicator2']
                
                logger.debug(f"Checking condition: {condition}")
                logger.debug(f"Available indicators: {list(self.indicators.keys())}")
                
                # Get first indicator
                if ind1_name not in self.indicators:
                    logger.warning(f"Indicator1 '{ind1_name}' not found")
                    continue
                ind1_values = self.indicators[ind1_name]
                
                # Get second indicator/value
                if ind2_name.isdigit():
                    ind2_values = np.full_like(ind1_values, float(ind2_name))
                elif ind2_name == 'signal' and f"{ind1_name}_signal" in self.indicators:
                    ind2_values = self.indicators[f"{ind1_name}_signal"]
                elif ind2_name in self.indicators:
                    ind2_values = self.indicators[ind2_name]
                else:
                    logger.warning(f"Indicator2 '{ind2_name}' not found")
                    continue
                
                # Get recent values for comparison
                if len(ind1_values) < 2 or len(ind2_values) < 2:
                    logger.warning("Not enough data points for comparison")
                    continue
                    
                ind1_curr, ind1_prev = ind1_values[-1], ind1_values[-2]
                ind2_curr, ind2_prev = ind2_values[-1], ind2_values[-2]
                
                # Check for NaN values
                if np.isnan([ind1_curr, ind1_prev, ind2_curr, ind2_prev]).any():
                    continue
                
                # Log values for debugging
                logger.debug(f"Values for {ind1_name}: {ind1_prev:.2f} -> {ind1_curr:.2f}")
                logger.debug(f"Values for {ind2_name}: {ind2_prev:.2f} -> {ind2_curr:.2f}")
                
                # Check conditions
                if condition['condition'] == 'crossover':
                    if ind1_prev <= ind2_prev and ind1_curr > ind2_curr:
                        logger.info(f"Crossover detected: {ind1_name}({ind1_curr:.2f}) crossed above {ind2_name}({ind2_curr:.2f})")
                        return True
                elif condition['condition'] == 'crossunder':
                    if ind1_prev >= ind2_prev and ind1_curr < ind2_curr:
                        logger.info(f"Crossunder detected: {ind1_name}({ind1_curr:.2f}) crossed below {ind2_name}({ind2_curr:.2f})")
                        return True

            return False
            
        except Exception as e:
            logger.error(f"Error checking conditions: {str(e)}")
            logger.debug("Exception details:", exc_info=True)
            return False

    def next(self):
        """Execute trading logic."""
        try:
            if not self.is_trading_time():
                return

            current_price = self.data.Close[-1]

            # Check entry conditions if not in position
            if not self.position:
                for condition in self.json_config['entry_conditions']:
                    if self.check_conditions([condition]):
                        size = condition.get('size', 0.95)
                        position_size = (self.equity * size) / current_price
                        logger.info(f"Entry signal at {self.data.index[-1]}")
                        logger.info(f"Price: {current_price:.2f}, Size: {position_size:.2f}, Equity: {self.equity:.2f}")
                        
                        if condition['action'] == 'buy':
                            self.buy(size=position_size)
                            self.entry_price = current_price
                            self.entry_time = self.data.index[-1]
                            
                            # Set stop loss and take profit
                            stop_loss_pct = self.json_config['risk_management']['stop_loss']
                            take_profit_pct = self.json_config['risk_management']['take_profit']
                            self.stop_loss = current_price * (1 - stop_loss_pct)
                            self.take_profit = current_price * (1 + take_profit_pct)
                            
                            logger.info(f"Stop Loss: {self.stop_loss:.2f}, Take Profit: {self.take_profit:.2f}")
                        break

            # Check exit conditions if in position
            elif self.position:
                # Check stop loss and take profit
                if self.position.is_long:
                    if current_price <= self.stop_loss or current_price >= self.take_profit:
                        logger.info(f"SL/TP triggered - Price: {current_price:.2f}, SL: {self.stop_loss:.2f}, TP: {self.take_profit:.2f}")
                        self.position.close()
                        return
                        
                # Check strategy exit conditions
                for condition in self.json_config['exit_conditions']:
                    if self.check_conditions([condition]):
                        logger.info(f"Exit signal at {self.data.index[-1]}, Price: {current_price:.2f}")
                        self.position.close()
                        break

        except Exception as e:
            logger.error(f"Error in next(): {str(e)}")
            logger.debug("Exception details:", exc_info=True)

    def log_trade(self, action: str, size: float):
        """Enhanced trade logging"""
        trade = {
            'time': self.data.index[-1],
            'action': action,
            'price': self.data.Close[-1],
            'size': size,
            'equity': self.equity,
            'pnl': self.position.pnl if self.position else 0
        }
        self.trades_log.append(trade)
        
        # Debug logging
        logger.info(f"Trade executed: {action} | Price: {trade['price']} | Size: {size} | Equity: {trade['equity']}")

class BacktestingAgent:
    """Main backtesting agent class"""
    
    def __init__(self):
        """Initialize backtesting agent with proper directory structure"""
        self.base_path = Path.cwd()
        self.output_dir = self.base_path / 'output'
        self.validate_directory_structure()
        self.technical_indicators = {}


    def validate_strategy(self, strategy: Dict, company_name: str) -> Dict:
        """Enhanced strategy validation to handle algo_agent.py format"""
        try:
            required_fields = [
                'indicators', 'entry_conditions', 'exit_conditions',
                'trading_hours', 'initial_capital', 'commission', 'risk_management'
            ]
            
            default_strategy = self.get_default_strategy(company_name)
            validated_strategy = default_strategy.copy()

            # Update with provided strategy
            for field in required_fields:
                if field in strategy:
                    validated_strategy[field] = strategy[field]
            
            # Validate and clean conditions
            for cond_type in ['entry_conditions', 'exit_conditions']:
                cleaned_conditions = []
                for cond in validated_strategy[cond_type]:
                    if all(k in cond for k in ['indicator1', 'indicator2', 'condition', 'action']):
                        cleaned_conditions.append(cond)
                validated_strategy[cond_type] = cleaned_conditions

            # Add missing indicators
            existing_indicators = {ind['name']: ind for ind in validated_strategy['indicators']}
            needed_indicators = set()
            
            # Collect needed indicators from conditions
            for cond_type in ['entry_conditions', 'exit_conditions']:
                for cond in validated_strategy[cond_type]:
                    needed_indicators.add(cond['indicator1'])
                    if not cond['indicator2'].isdigit():
                        needed_indicators.add(cond['indicator2'])

            # Add missing indicators
            for ind_name in needed_indicators:
                if ind_name not in existing_indicators:
                    if 'macd' in ind_name:
                        validated_strategy['indicators'].append({
                            "type": "MACD",
                            "name": "macd",
                            "params": {"fast": 12, "slow": 26, "signal": 9}
                        })
                    elif 'rsi' in ind_name:
                        validated_strategy['indicators'].append({
                            "type": "RSI",
                            "name": "rsi",
                            "params": {"length": 14}
                        })

            logger.info(f"Validated strategy: {json.dumps(validated_strategy, indent=2)}")
            return validated_strategy

        except Exception as e:
            logger.error(f"Strategy validation failed: {e}")
            return self.get_default_strategy(company_name)
        

    def _validate_conditions(self, conditions: List[Dict], indicator_names: List[str]) -> List[Dict]:
        """Validate trading conditions against available indicators."""
        valid_conditions = []
        
        for condition in conditions:
            if isinstance(condition, dict):
                # Check if condition has the required fields
                if all(k in condition for k in ['indicator1', 'indicator2', 'condition', 'action']):
                    ind1 = condition['indicator1']
                    ind2 = condition['indicator2']
                    
                    try:
                        # Handle numeric indicator2 values (like "30" for RSI)
                        if isinstance(ind2, str) and ind2.isdigit():
                            valid_conditions.append(condition)
                            continue
                            
                        # Check if both indicators exist
                        if ind1 in indicator_names and ind2 in indicator_names:
                            valid_conditions.append(condition)
                        else:
                            logger.warning(f"Invalid indicators in condition: {ind1} or {ind2}")
                    except Exception as e:
                        logger.error(f"Error validating condition: {e}")
                        
        return valid_conditions


    # def initialize_indicator(self, indicator_config: Dict):
    #     """Initialize technical indicators with comprehensive validation and error handling"""
    #         # Add price series as basic indicators
    #     self.indicators['close'] = self.data.Close
    #     self.indicators['high'] = self.data.High
    #     self.indicators['low'] = self.data.Low

    #     try:
    #         indicator_type = indicator_config['type'].lower()
    #         params = indicator_config.get('params', {})
    #         name = indicator_config['name']
            
    #         logger.info(f"Initializing indicator: {name} (Type: {indicator_type})")
            
    #         # Get price data and validate
    #         close = self.data.Close
    #         high = self.data.High
    #         low = self.data.Low
    #         volume = self.data.Volume
            
    #         # Basic data validation
    #         if len(close) == 0 or len(high) == 0 or len(low) == 0 or len(volume) == 0:
    #             raise ValueError("Missing required price data")
                
    #         # Handle pre-calculated indicators from CSV
    #         if name in self.technical_indicators:
    #             logger.info(f"Using pre-calculated values for {name}")
    #             self.indicators[name] = self.technical_indicators[name]
    #             return

    #         # Define all supported indicators
    #         indicator_functions = {
    #             'sma': lambda: self.I(talib.SMA, close, 
    #                                 timeperiod=params.get('length', 20)),
                
    #             'ema': lambda: self.I(talib.EMA, close, 
    #                                 timeperiod=params.get('length', 20)),
                
    #             'rsi': lambda: self.I(talib.RSI, close, 
    #                                 timeperiod=params.get('length', 14)),
                
    #             'macd': lambda: talib.MACD(
    #                 close,
    #                 fastperiod=params.get('fast_length', 12),
    #                 slowperiod=params.get('slow_length', 26),
    #                 signalperiod=params.get('signal_length', 9)
    #             ),
                
    #             'bbands': lambda: talib.BBANDS(
    #                 close,
    #                 timeperiod=params.get('length', 20),
    #                 nbdevup=params.get('mult', 2),
    #                 nbdevdn=params.get('mult', 2)
    #             ),
                
    #             'vwap': lambda: self.I(TechnicalIndicators.calculate_vwap, 
    #                                 high, low, close, volume),
                
    #             'supertrend': lambda: TechnicalIndicators.calculate_supertrend(
    #                 high, low, close,
    #                 period=params.get('period', 10),
    #                 multiplier=params.get('multiplier', 3.0)
    #             ),
                
    #             'obv': lambda: self.I(talib.OBV, close, volume),
                
    #             'atr': lambda: self.I(talib.ATR, high, low, close,
    #                                 timeperiod=params.get('length', 14)),
                
    #             'stoch': lambda: talib.STOCH(
    #                 high, low, close,
    #                 fastk_period=params.get('k_period', 14),
    #                 slowk_period=params.get('k_smooth', 3),
    #                 slowk_matype=0,
    #                 slowd_period=params.get('d_period', 3),
    #                 slowd_matype=0
    #             ),
                
    #             'adx': lambda: self.I(talib.ADX, high, low, close,
    #                                 timeperiod=params.get('length', 14)),
                
    #             'cci': lambda: self.I(talib.CCI, high, low, close,
    #                                 timeperiod=params.get('length', 14)),
                
    #             'roc': lambda: self.I(talib.ROC, close,
    #                                 timeperiod=params.get('length', 10)),
                
    #             'mom': lambda: self.I(talib.MOM, close,
    #                                 timeperiod=params.get('length', 10)),
                
    #             # Price references
    #             'close': lambda: close,
    #             'high': lambda: high,
    #             'low': lambda: low,
    #             'volume': lambda: volume,
    #         }
            
    #         # Calculate indicator
    #         if indicator_type in indicator_functions:
    #             try:
    #                 indicator_value = indicator_functions[indicator_type]()
                    
    #                 # Validate indicator values
    #                 if indicator_value is not None:
    #                     # Handle tuple indicators (like MACD, Bollinger Bands)
    #                     if isinstance(indicator_value, tuple):
    #                         # Validate each component
    #                         for i, component in enumerate(indicator_value):
    #                             if not isinstance(component, np.ndarray):
    #                                 component = np.array(component)
    #                             if np.all(np.isnan(component)):
    #                                 raise ValueError(f"All values are NaN for {name} component {i}")
    #                             logger.debug(f"{name} component {i} range: {np.nanmin(component):.2f} to {np.nanmax(component):.2f}")
    #                     else:
    #                         # Single array indicator
    #                         if not isinstance(indicator_value, np.ndarray):
    #                             indicator_value = np.array(indicator_value)
    #                         if np.all(np.isnan(indicator_value)):
    #                             raise ValueError(f"All values are NaN for {name}")
    #                         logger.debug(f"{name} range: {np.nanmin(indicator_value):.2f} to {np.nanmax(indicator_value):.2f}")
                        
    #                     # Store the indicator
    #                     self.indicators[name] = indicator_value
    #                     logger.info(f"Successfully initialized {name} indicator")
                        
    #                     # Log some statistics about the indicator
    #                     if isinstance(indicator_value, tuple):
    #                         logger.info(f"{name} initialized as tuple with {len(indicator_value)} components")
    #                     else:
    #                         non_nan = np.count_nonzero(~np.isnan(indicator_value))
    #                         logger.info(f"{name} initialized with {non_nan} valid values out of {len(indicator_value)}")
    #                 else:
    #                     raise ValueError(f"Indicator calculation returned None for {name}")
                        
    #             except Exception as e:
    #                 logger.error(f"Error calculating {name}: {str(e)}")
    #                 raise
    #         else:
    #             logger.warning(f"Unsupported indicator type: {indicator_type}")
    #             logger.info(f"Using price series as fallback for {name}")
    #             self.indicators[name] = close
                
    #     except Exception as e:
    #         logger.error(f"Error initializing indicator {name}: {str(e)}")
    #         # Instead of raising, we'll use price series as fallback
    #         logger.warning(f"Using price series as fallback for {name}")
    #         self.indicators[name] = self.data.Close
            
    #     finally:
    #         # Verify indicator was stored
    #         if name not in self.indicators:
    #             logger.error(f"Failed to initialize {name}, using price series as fallback")
    #             self.indicators[name] = self.data.Close


    def initialize_indicator(self, indicator_config: Dict):
        """Initialize technical indicators with comprehensive validation."""
        try:
            indicator_type = indicator_config['type'].lower()
            params = indicator_config.get('params', {})
            name = indicator_config['name']

            # Basic price data
            self.indicators['close'] = self.data.Close
            self.indicators['high'] = self.data.High
            self.indicators['low'] = self.data.Low
            self.indicators['volume'] = self.data.Volume

            # Extended indicator mapping
            indicators = {
                'sma': lambda: self.I(talib.SMA, self.data.Close, timeperiod=params.get('length', 20)),
                'ema': lambda: self.I(talib.EMA, self.data.Close, timeperiod=params.get('length', 20)),
                'rsi': lambda: self.I(talib.RSI, self.data.Close, timeperiod=params.get('length', 14)),
                'macd': lambda: self.initialize_macd(params),
                'bbands': lambda: self.initialize_bbands(params),
                'obv': lambda: self.I(talib.OBV, self.data.Close, self.data.Volume)
            }
            
            if indicator_type in indicators:
                self.indicators[name] = indicators[indicator_type]()
                if indicator_type == 'macd':
                    # For MACD, also store signal line as separate indicator
                    macd_values = self.indicators[name]
                    if isinstance(macd_values, tuple) and len(macd_values) >= 2:
                        self.indicators[f"{name}_signal"] = macd_values[1]
                    
            logger.info(f"Initialized indicator: {name}")
                
        except Exception as e:
            logger.error(f"Error initializing indicator {name}: {e}")
            raise



    def initialize_alma(self, params: Dict) -> np.ndarray:
        """Initialize ALMA (Arnaud Legoux Moving Average) indicator"""
        try:
            length = params.get('length', 20)
            offset = params.get('offset', 0.85)
            sigma = params.get('sigma', 6)
            source = self.data[params.get('source', 'Close')].values
            
            def calculate_alma(data, window_size, offset, sigma):
                m = offset * (window_size - 1)
                s = window_size / sigma
                weights = np.zeros(window_size)
                
                for i in range(window_size):
                    weights[i] = np.exp(-((i - m) ** 2) / (2 * s * s))
                    
                weights = weights / np.sum(weights)
                return np.convolve(data, weights[::-1], mode='valid')
            
            alma = calculate_alma(source, length, offset, sigma)
            # Pad the beginning to match input length
            padding = np.full(length - 1, np.nan)
            return np.concatenate([padding, alma])
            
        except Exception as e:
            logger.error(f"Error calculating ALMA: {e}")
            return np.full_like(self.data.Close.values, np.nan)

    def generate_performance_report(self, stats: Dict, company_name: str, algo_num: int) -> None:
        """Generate comprehensive HTML performance report"""
        try:
            report_dir = self.output_dir / 'backtest_results' / f"{company_name}_algo{algo_num}"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # Create HTML report
            html_content = f"""
            <html>
            <head>
                <title>Backtest Performance Report - {company_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .metric {{ margin: 10px 0; }}
                    .chart {{ margin: 20px 0; }}
                    .good {{ color: green; }}
                    .bad {{ color: red; }}
                </style>
            </head>
            <body>
                <h1>Backtest Performance Report - {company_name}</h1>
                <h2>Summary Statistics</h2>
                <div class="metric">
                    <h3>Returns</h3>
                    <p>Total Return: <span class="{'good' if stats['Return [%]'] > 0 else 'bad'}">{stats['Return [%]']:.2f}%</span></p>
                    <p>Annual Return: <span class="{'good' if stats['Return (Ann.) [%]'] > 0 else 'bad'}">{stats['Return (Ann.) [%]']:.2f}%</span></p>
                </div>
                
                <div class="metric">
                    <h3>Risk Metrics</h3>
                    <p>Sharpe Ratio: {stats['Sharpe Ratio']:.2f}</p>
                    <p>Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%</p>
                    <p>Volatility (Ann.): {stats['Volatility (Ann.) [%]']:.2f}%</p>
                </div>
                
                <div class="metric">
                    <h3>Trade Statistics</h3>
                    <p>Total Trades: {stats['# Trades']}</p>
                    <p>Win Rate: {stats['Win Rate [%]']:.2f}%</p>
                    <p>Best Trade: {stats['Best Trade [%]']:.2f}%</p>
                    <p>Worst Trade: {stats['Worst Trade [%]']:.2f}%</p>
                </div>
                
                <div class="chart">
                    <h2>Performance Charts</h2>
                    <img src="equity_curve.png" alt="Equity Curve">
                    <img src="drawdown_analysis.png" alt="Drawdown Analysis">
                    <img src="monthly_returns.png" alt="Monthly Returns">
                </div>
            </body>
            </html>
            """
            
            with open(report_dir / 'performance_report.html', 'w') as f:
                f.write(html_content)
                
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            raise


    def load_latest_strategy(self, company_name: str) -> Tuple[Dict, int]:
        """Load the latest JSON strategy file"""
        try:
            pattern = str(self.output_dir / 'algo' / f"{company_name}_algorithm-*.json")
            files = glob.glob(pattern)
            
            if not files:
                logger.warning(f"No strategy files found for {company_name}")
                return self.get_default_strategy(company_name), 1
            
            latest_file = max(files, key=lambda x: int(x.split('-')[-1].split('.')[0]))
            algo_num = int(latest_file.split('-')[-1].split('.')[0])
            
            with open(latest_file, 'r') as f:
                strategy = json.load(f)
            
            # Validate and update strategy with any missing fields
            strategy = self.validate_strategy(strategy, company_name)
            
            logger.info(f"Loaded strategy {algo_num} for {company_name}")
            return strategy, algo_num
            
        except Exception as e:
            logger.error(f"Error loading strategy: {e}")
            return self.get_default_strategy(company_name), 1
    


    def get_default_strategy(self, company_name: str) -> Dict:
        """Return a default strategy if generation/loading fails"""
        default_strategy = {
            "indicators": [
                {
                    "type": "SMA",
                    "name": "sma_fast",
                    "params": {
                        "length": 10,
                        "source": "Close"
                    }
                },
                {
                    "type": "SMA",
                    "name": "sma_slow",
                    "params": {
                        "length": 20,
                        "source": "Close"
                    }
                },
                {
                    "type": "RSI",
                    "name": "rsi",
                    "params": {
                        "length": 14,
                        "source": "Close"
                    }
                }
            ],
            "entry_conditions": [
                {
                    "indicator1": "sma_fast",
                    "indicator2": "sma_slow",
                    "condition": "crossover",
                    "action": "buy",
                    "size": 0.95
                }
            ],
            "exit_conditions": [
                {
                    "indicator1": "sma_fast",
                    "indicator2": "sma_slow",
                    "condition": "crossunder",
                    "action": "exit_long"
                }
            ],
            "short_conditions": [
                {
                    "indicator1": "sma_fast",
                    "indicator2": "sma_slow",
                    "condition": "crossunder",
                    "action": "sell",
                    "size": 0.95
                }
            ],
            "short_exit_conditions": [
                {
                    "indicator1": "sma_fast",
                    "indicator2": "sma_slow",
                    "condition": "crossover",
                    "action": "exit_short"
                }
            ],
            "trading_hours": {
                "start": "09:15",
                "end": "15:30"
            },
            "initial_capital": 100000,
            "commission": 0.002
        }
        
        # Save default strategy
        output_dir = self.output_dir / 'algo'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        strategy_file = output_dir / f"{company_name}_algorithm-1.json"
        with open(strategy_file, 'w') as f:
            json.dump(default_strategy, f, indent=4)
        
        logger.info(f"Created default strategy for {company_name}")
        return default_strategy



    def validate_strategy(self, strategy: Dict, company_name: str) -> Dict:
            """Enhanced strategy validation to handle algo_agent.py format"""
            try:
                # Convert moving_averages to indicators format
                if 'moving_averages' in strategy:
                    indicators = []
                    for ma in strategy['moving_averages']:
                        indicator = {
                            'type': ma['type'],
                            'name': ma['name'],
                            'params': {
                                'length': ma['length'],
                                'offset': ma.get('offset', 0.85),
                                'sigma': ma.get('sigma', 5.0),
                                'source': ma['source']
                            }
                        }
                        indicators.append(indicator)
                    strategy['indicators'] = indicators
                    del strategy['moving_averages']

                # Ensure all required fields exist
                required_fields = [
                    'indicators', 'entry_conditions', 'exit_conditions',
                    'trading_hours', 'initial_capital', 'commission'
                ]
                
                for field in required_fields:
                    if field not in strategy:
                        logger.warning(f"Missing {field} in strategy, using default")
                        if field == 'indicators':
                            strategy[field] = self.get_default_strategy(company_name)[field]
                        else:
                            strategy[field] = self.get_default_strategy(company_name)[field]

                # Validate indicator references
                indicator_names = [ind['name'] for ind in strategy['indicators']]
                
                self._validate_conditions(strategy['entry_conditions'], indicator_names)
                self._validate_conditions(strategy['exit_conditions'], indicator_names)
                
                if 'short_conditions' in strategy:
                    self._validate_conditions(strategy['short_conditions'], indicator_names)
                if 'short_exit_conditions' in strategy:
                    self._validate_conditions(strategy['short_exit_conditions'], indicator_names)

                return strategy
                
            except Exception as e:
                logger.error(f"Strategy validation failed: {e}")
                return self.get_default_strategy(company_name)
    

    def validate_directory_structure(self):
        """Validate and create required directory structure"""
        try:
            required_dirs = [
                self.base_path / 'agents' / 'backtesting_agent' / 'historical_data',
                self.output_dir / 'backtest_results',
                self.output_dir / 'algo'
            ]
            
            # Create directories if they don't exist
            for directory in required_dirs:
                directory.mkdir(parents=True, exist_ok=True)
                
            logger.info("Directory structure validated and created")
            
        except Exception as e:
            logger.error(f"Error validating directory structure: {e}")
            raise


    def run_backtest(self, company_name: str, start_date: str = None, 
                    end_date: str = None) -> Dict:
        """Run backtest for the given company"""
        try:
            # Load and validate strategy
            strategy, algo_num = self.load_latest_strategy(company_name)
            strategy = self.validate_strategy(strategy, company_name)
            
            # Load and sort data
            data = self.load_historical_data(company_name, start_date, end_date)
            data = data.sort_index()  # Ensure data is sorted
            
            # Initialize strategy
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
            
            # Run backtest and generate reports
            stats = backtest.run()
            self.generate_reports(stats, company_name, algo_num, data)
            
            # Generate HTML performance report
            self.generate_performance_report(stats, company_name, algo_num)
            
            return stats
            
        except Exception as e:
            logger.error(f"Backtesting error: {e}")
            raise



    def load_historical_data(self, company_name: str, start_date: str = None, 
                            end_date: str = None) -> pd.DataFrame:
        """Load and preprocess historical data with extended columns"""
        try:
            data_file = self.base_path / 'agents' / 'backtesting_agent' / 'historical_data' / f"{company_name}_minute.csv"
            
            if not data_file.exists():
                raise FileNotFoundError(f"Historical data file not found: {data_file}")
            
            # Read CSV with all columns
            df = pd.read_csv(data_file)
            
            # Handle timezone-aware datetime with UTC+05:30
            df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_localize(None)
            df = df.set_index('Datetime')
            
            # Convert date strings to timezone-naive datetime
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
            
            # Handle zero or missing volume
            df['Volume'] = df['Volume'].replace(0, np.nan)
            df['Volume'] = df['Volume'].fillna(df['Volume'].mean())
            
            # Store technical indicators if present
            indicator_cols = {
                'SMA_50': 'sma_50',
                'RSI': 'rsi',
                'MACD': 'macd',
                'MACD_Signal': 'macd_signal',
                'MACD_Hist': 'macd_hist',
                'BB_Upper': 'bb_upper',
                'BB_Middle': 'bb_middle',
                'BB_Lower': 'bb_lower',
                'OBV': 'obv'
            }
            
            self.technical_indicators = {}
            for csv_col, indicator_name in indicator_cols.items():
                if csv_col in df.columns:
                    self.technical_indicators[indicator_name] = pd.to_numeric(
                        df[csv_col], errors='coerce'
                    ).values
            
            # Drop rows with NaN in required columns
            df = df.dropna(subset=required_cols)
            
            if len(df) == 0:
                raise ValueError("No valid data points after preprocessing")
            
            logger.info(f"Loaded {len(df)} records from {df.index[0]} to {df.index[-1]}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            raise



    def generate_reports(self, stats: Dict, company_name: str, algo_num: int,
                        data: pd.DataFrame):
        """Generate comprehensive backtest reports"""
        try:
            # Create output directory
            report_dir = self.output_dir / 'backtest_results' / f"{company_name}_algo{algo_num}"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate reports in order with proper error handling
            self.save_summary_stats(stats, report_dir)
            
            try:
                self.plot_equity_curve(stats, report_dir)
            except Exception as e:
                logger.error(f"Error generating equity curve: {e}")
                
            try:
                self.plot_drawdown_analysis(stats, report_dir)
            except Exception as e:
                logger.error(f"Error generating drawdown analysis: {e}")
                
            try:
                self.generate_trade_analysis(stats, report_dir)
            except Exception as e:
                logger.error(f"Error generating trade analysis: {e}")
                
            try:
                self.plot_monthly_returns(stats, report_dir)
            except Exception as e:
                logger.error(f"Error generating monthly returns: {e}")
            
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
            raise


    def save_summary_stats(self, stats: Dict, report_dir: Path):
        """Save detailed summary statistics"""
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
            f.write(f"Average Trade Duration: {stats['Avg. Trade Duration']}\n\n")
            
            # Daily Trading Summary
            if hasattr(stats['_strategy'], 'daily_stats'):
                f.write("\nDaily Trading Summary:\n")
                for date, stats in sorted(stats['_strategy'].daily_stats.items(), reverse=True):
                    f.write(f"{date} Profit: {stats['profit']:.2f}% ; ")
                    f.write(f"total trade: {stats['total_trades']} ; ")
                    f.write(f"buy_trade - {stats['buy_trades']} ; ")
                    f.write(f"sell_trade - {stats['sell_trades']}\n")

    def plot_equity_curve(self, stats: Dict, report_dir: Path):
        """Generate equity curve plot with drawdown overlay"""
        equity_data = pd.DataFrame({
            'Equity': stats['_equity_curve']['Equity'],
            'DrawdownPct': stats['_equity_curve']['DrawdownPct']
        })
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        ax1.plot(equity_data.index, equity_data['Equity'], label='Portfolio Value', color='blue')
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

    def plot_drawdown_analysis(self, stats: Dict, report_dir: Path):
        """Generate detailed drawdown analysis plots"""
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

    def get_drawdown_periods(self, drawdown_data: pd.DataFrame) -> List[Dict]:
        """Extract drawdown periods for analysis"""
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

    def generate_trade_analysis(self, stats: Dict, report_dir: Path):
        """Generate detailed trade analysis with proper error handling"""
        try:
            trades = stats.get('_trades', [])
            if not trades or len(trades) == 0:
                logger.warning("No trades found for analysis")
                return
                
            trades_data = []
            for t in trades:
                if hasattr(t, 'entry_time') and hasattr(t, 'exit_time'):
                    trades_data.append({
                        'Entry Time': t.entry_time,
                        'Exit Time': t.exit_time,
                        'Size': t.size,
                        'Entry Price': t.entry_price,
                        'Exit Price': t.exit_price,
                        'PnL': t.pl,
                        'Return': t.pl_pct,
                        'Duration': (t.exit_time - t.entry_time).total_seconds() / 3600,
                        'Type': 'Long' if t.size > 0 else 'Short'
                    })
            
            if not trades_data:
                logger.warning("No valid trades for analysis")
                return
                
            trades_df = pd.DataFrame(trades_data)

            # Generate trade analysis plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
            
            # Plot 1: PnL Distribution
            sns.histplot(data=trades_df['PnL'], ax=ax1, bins=20)
            ax1.set_title('Trade PnL Distribution')
            ax1.set_xlabel('PnL ($)')
            
            # Plot 2: Return vs Duration
            ax2.scatter(trades_df['Duration'], trades_df['Return'])
            ax2.set_title('Trade Return vs Duration')
            ax2.set_xlabel('Duration (hours)')
            ax2.set_ylabel('Return (%)')
            
            # Plot 3: Cumulative PnL
            cumulative_pnl = trades_df['PnL'].cumsum()
            ax3.plot(range(len(cumulative_pnl)), cumulative_pnl)
            ax3.set_title('Cumulative PnL')
            ax3.set_xlabel('Trade Number')
            ax3.set_ylabel('Cumulative PnL ($)')
            
            # Plot 4: Returns by Trade Type
            sns.boxplot(data=trades_df, x='Type', y='Return', ax=ax4)
            ax4.set_title('Returns by Trade Type')
            ax4.set_xlabel('Trade Type')
            ax4.set_ylabel('Return (%)')
            
            plt.tight_layout()
            plt.savefig(report_dir / 'trade_analysis.png')
            plt.close()
            
            # Save trade log
            trades_df.to_csv(report_dir / 'trades.csv', index=False)
            
        except Exception as e:
            logger.error(f"Error generating trade analysis: {str(e)}")
            logger.debug("Exception details:", exc_info=True)

    def plot_monthly_returns(self, stats: Dict, report_dir: Path):
        """Generate monthly returns heatmap"""
        try:
            # Extract trade data from stats
            trades = stats.get('_trades', [])
            if not trades:
                logger.warning("No trades found for monthly analysis")
                return
                
            # Create DataFrame with relevant trade information
            trades_df = pd.DataFrame({
                'Exit Time': [t.exit_time for t in trades],
                'PnL': [t.pl for t in trades],
                'Return': [t.pl_pct for t in trades]
            })
            
            # Set datetime index
            trades_df['Exit Time'] = pd.to_datetime(trades_df['Exit Time'])
            trades_df = trades_df.set_index('Exit Time')
            
            # Calculate monthly returns
            monthly_returns = trades_df.resample('M')['PnL'].sum()
            monthly_returns = monthly_returns.to_frame()
            monthly_returns['Year'] = monthly_returns.index.year
            monthly_returns['Month'] = monthly_returns.index.month
            
            # Create pivot table for heatmap
            pivot_table = monthly_returns.pivot(
                index='Year', 
                columns='Month', 
                values='PnL'
            )
            
            # Plot heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                pivot_table, 
                annot=True, 
                fmt='.0f', 
                cmap='RdYlGn', 
                center=0,
                cbar_kws={'label': 'PnL ($)'}
            )
            plt.title('Monthly Returns Heatmap')
            plt.ylabel('Year')
            plt.xlabel('Month')
            
            # Format month labels
            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            plt.xticks(np.arange(12) + 0.5, month_labels, rotation=0)
            
            plt.tight_layout()
            plt.savefig(report_dir / 'monthly_returns.png')
            plt.close()
            
            # Save monthly returns data
            monthly_returns_flat = pivot_table.reset_index().melt(
                id_vars=['Year'],
                var_name='Month',
                value_name='PnL'
            )
            monthly_returns_flat.to_csv(report_dir / 'monthly_returns.csv', index=False)
            
        except Exception as e:
            logger.error(f"Error generating monthly returns plot: {e}")
            logger.info("Skipping monthly returns analysis due to error")


    def save_trade_log(self, trades_log: List[Dict], report_dir: Path):
        """Save detailed trade log with additional metrics"""
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
        
        # Save detailed trade log
        trades_df.to_excel(report_dir / 'detailed_trade_log.xlsx', index=False)
        
        # Generate trade summary
        with open(report_dir / 'trade_summary.txt', 'w') as f:
            f.write("Trade Summary Statistics\n")
            f.write("=" * 50 + "\n\n")
            
            if len(trades_df) > 0:
                f.write(f"Total Trades: {len(trades_df)}\n")
                f.write(f"Profitable Trades: {(trades_df['pnl'] > 0).sum()}\n")
                f.write(f"Loss-Making Trades: {(trades_df['pnl'] < 0).sum()}\n")
                f.write(f"Average Profit per Trade: ${trades_df['pnl'].mean():.2f}\n")
                f.write(f"Largest Profit: ${trades_df['pnl'].max():.2f}\n")
                f.write(f"Largest Loss: ${trades_df['pnl'].min():.2f}\n")
                f.write(f"Average Trade Duration: {trades_df['trade_duration'].mean()}\n\n")
                
                # Market condition analysis
                f.write("Performance by Market Condition:\n")
                market_stats = trades_df.groupby('market_condition')['pnl'].agg([
                    'count', 'mean', 'sum'
                ])
                f.write(market_stats.to_string())



# # ###Example usage


def main():
    """Example usage of backtesting agent with specified company."""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logger.info("Starting backtesting agent")
        
        # Initialize backtesting agent
        backtest_agent = BacktestingAgent()
        
        # Specify company name here
        company_name = "ZOMATO"  # <-- Change this to test different companies
        
        # Log strategy loading
        logger.info(f"Loading strategy for {company_name}")
        
        # Run backtest
        stats = backtest_agent.run_backtest(
            company_name=company_name,
            start_date="2024-01-01 00:00:00",
            end_date="2024-10-31 23:59:59"
        )
        
        # Print comprehensive results
        print("\nBacktest Results Summary")
        print("=" * 50)
        print(f"Company: {company_name}")
        print(f"\nPerformance Metrics:")
        print(f"Total Return: {stats['Return [%]']:.2f}%")
        print(f"Annual Return: {stats['Return (Ann.) [%]']:.2f}%")
        print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
        print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
        print(f"Total Trades: {stats['# Trades']}")
        
        print(f"\nRisk Metrics:")
        print(f"Volatility (Ann.): {stats['Volatility (Ann.) [%]']:.2f}%")
        print(f"Sortino Ratio: {stats['Sortino Ratio']:.2f}")
        print(f"Calmar Ratio: {stats['Calmar Ratio']:.2f}")
        
        logger.info("Backtesting completed successfully")
        logger.info(f"Full reports available in: {backtest_agent.output_dir}/backtest_results/")

    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()

