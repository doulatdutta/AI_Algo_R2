import talib
import vectorbt as vbt
import pandas as pd
import numpy as np
import json
import glob
import re
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
log_dir = Path(os.getcwd()) / "logs"
log_dir.mkdir(exist_ok=True)
log_filename = log_dir / f"vectorbt_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class VectorBTBacktester:
    def __init__(self):
        self.base_path = Path(os.getcwd())
        self.input_data_dir = self.base_path / "agents" / "backtesting_agent" / "historical_data"
        self.output_dir = self.base_path / "output" / "backtest_results"
        self.algo_dir = self.base_path / "output" / "algo"
        
        # Create necessary directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized VectorBTBacktester with data dir: {self.input_data_dir}")
        logger.info(f"Results will be saved to: {self.output_dir}")

    def get_latest_algorithm(self, company_name: str) -> dict:
        """Get the latest algorithm configuration for the company."""
        try:
            pattern = f"{company_name}_algorithm-*.json"
            algo_files = list(self.algo_dir.glob(pattern))
            
            if not algo_files:
                raise FileNotFoundError(f"No algorithm files found for {company_name}")
            
            latest_file = max(algo_files, key=lambda x: 
                int(re.search(r'-(\d+)\.json$', x.name).group(1)))
            
            logger.info(f"Loading algorithm from: {latest_file}")
            
            with open(latest_file, 'r') as f:
                algo_config = json.load(f)
            
            return algo_config
            
        except Exception as e:
            logger.error(f"Error loading algorithm: {e}")
            raise

    def load_data(self, company_name: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
            """Load and prepare historical data."""
            try:
                file_path = self.input_data_dir / f"{company_name}_minute.csv"
                logger.info(f"Loading data from: {file_path}")
                
                df = pd.read_csv(file_path)
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                df = df.set_index('Datetime')
                
                if start_date:
                    df = df[df.index >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df.index <= pd.to_datetime(end_date)]
                
                df = df.sort_index()
                
                # Ensure all required columns exist and are numeric
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Forward fill missing values
                df = df.ffill()
                
                logger.info(f"Loaded {len(df)} records from {df.index[0]} to {df.index[-1]}")
                return df
                
            except Exception as e:
                logger.error(f"Error loading data: {e}")
                raise

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, pd.Series], 
                        algo_config: dict) -> Tuple[pd.Series, pd.Series]:
        """Generate entry and exit signals based on conditions."""
        try:
            # Align DataFrame index with indicators
            common_index = indicators[list(indicators.keys())[0]].index
            df = df.reindex(common_index)
            
            # Initialize signal series
            entry_signals = pd.Series(False, index=common_index)
            exit_signals = pd.Series(False, index=common_index)
            
            # Process conditions
            for condition in algo_config['entry_conditions']:
                indicator = condition['indicator']
                value = float(condition['value'])
                
                logger.info(f"\nProcessing entry condition: {condition}")
                logger.info(f"Indicator {indicator} range: {indicators[indicator].min():.2f} to {indicators[indicator].max():.2f}")
                
                if condition['condition'] == 'above':
                    signal = indicators[indicator] > value
                elif condition['condition'] == 'below':
                    signal = indicators[indicator] < value
                elif condition['condition'] == 'crossover':
                    signal = self.detect_crossover(indicators[indicator], pd.Series(value, index=common_index))
                elif condition['condition'] == 'crossunder':
                    signal = self.detect_crossunder(indicators[indicator], pd.Series(value, index=common_index))
                
                entry_signals |= signal
                logger.info(f"Condition generated {signal.sum()} signals")
            
            # Similar process for exit conditions
            for condition in algo_config['exit_conditions']:
                indicator = condition['indicator']
                value = float(condition['value'])
                
                logger.info(f"\nProcessing exit condition: {condition}")
                
                if condition['condition'] == 'above':
                    signal = indicators[indicator] > value
                elif condition['condition'] == 'below':
                    signal = indicators[indicator] < value
                elif condition['condition'] == 'crossover':
                    signal = self.detect_crossover(indicators[indicator], pd.Series(value, index=common_index))
                elif condition['condition'] == 'crossunder':
                    signal = self.detect_crossunder(indicators[indicator], pd.Series(value, index=common_index))
                
                exit_signals |= signal
            
            # Apply trading hours filter
            trading_hours = algo_config['trading_hours']
            time_mask = pd.Series(common_index.strftime('%H:%M')).between(
                trading_hours['start'],
                trading_hours['end']
            )
            
            entry_signals &= time_mask
            exit_signals &= time_mask
            
            logger.info(f"\nFinal signal count:")
            logger.info(f"Entry signals: {entry_signals.sum()}")
            logger.info(f"Exit signals: {exit_signals.sum()}")
            
            return entry_signals, exit_signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            raise

    def detect_crossover(self, series1: pd.Series, value: float) -> pd.Series:
            """Detect when series crosses above a value."""
            series1 = pd.to_numeric(series1, errors='coerce')
            current = series1 > value
            previous = series1.shift(1) <= value
            return current & previous

    def detect_crossunder(self, series1: pd.Series, value: float) -> pd.Series:
        """Detect when series crosses below a value."""
        series1 = pd.to_numeric(series1, errors='coerce')
        current = series1 < value
        previous = series1.shift(1) >= value
        return current & previous

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, pd.Series], 
                        algo_config: dict) -> Tuple[pd.Series, pd.Series]:
        """Generate entry and exit signals based on conditions."""
        try:
            # Get common index
            common_index = indicators[list(indicators.keys())[0]].index
            
            # Initialize signal series
            entry_signals = pd.Series(False, index=common_index)
            exit_signals = pd.Series(False, index=common_index)
            
            # Debug logs
            logger.info("\nStarting signal generation")
            logger.info(f"Data points to process: {len(common_index)}")
            logger.info("\nIndicator Statistics:")
            for name, values in indicators.items():
                logger.info(f"{name} - Min: {values.min():.2f}, Max: {values.max():.2f}, Mean: {values.mean():.2f}")
            
            # Process entry conditions
            entry_conditions_met = []
            for condition in algo_config['entry_conditions']:
                indicator = condition['indicator']
                value = float(condition['value'])
                
                logger.info(f"\nChecking entry condition: {condition}")
                logger.info(f"Indicator {indicator} current value: {indicators[indicator].iloc[-1]:.2f}")
                logger.info(f"Condition value: {value}")
                
                condition_met = pd.Series(False, index=common_index)
                
                if condition['condition'] == 'above':
                    condition_met = indicators[indicator] > value
                    logger.info(f"Above condition matched {condition_met.sum()} times")
                    
                elif condition['condition'] == 'below':
                    condition_met = indicators[indicator] < value
                    logger.info(f"Below condition matched {condition_met.sum()} times")
                    
                elif condition['condition'] == 'crossover':
                    condition_met = self.detect_crossover(indicators[indicator], value)
                    logger.info(f"Crossover condition matched {condition_met.sum()} times")
                    
                elif condition['condition'] == 'crossunder':
                    condition_met = self.detect_crossunder(indicators[indicator], value)
                    logger.info(f"Crossunder condition matched {condition_met.sum()} times")
                
                entry_conditions_met.append(condition_met)
            
            # Process exit conditions
            exit_conditions_met = []
            for condition in algo_config['exit_conditions']:
                indicator = condition['indicator']
                value = float(condition['value'])
                
                logger.info(f"\nChecking exit condition: {condition}")
                logger.info(f"Indicator {indicator} current value: {indicators[indicator].iloc[-1]:.2f}")
                logger.info(f"Condition value: {value}")
                
                condition_met = pd.Series(False, index=common_index)
                
                if condition['condition'] == 'above':
                    condition_met = indicators[indicator] > value
                    logger.info(f"Above condition matched {condition_met.sum()} times")
                    
                elif condition['condition'] == 'below':
                    condition_met = indicators[indicator] < value
                    logger.info(f"Below condition matched {condition_met.sum()} times")
                    
                elif condition['condition'] == 'crossover':
                    condition_met = self.detect_crossover(indicators[indicator], value)
                    logger.info(f"Crossover condition matched {condition_met.sum()} times")
                    
                elif condition['condition'] == 'crossunder':
                    condition_met = self.detect_crossunder(indicators[indicator], value)
                    logger.info(f"Crossunder condition matched {condition_met.sum()} times")
                
                exit_conditions_met.append(condition_met)
            
            # Combine conditions
            if entry_conditions_met:
                entry_signals = pd.concat(entry_conditions_met, axis=1).all(axis=1)
            if exit_conditions_met:
                exit_signals = pd.concat(exit_conditions_met, axis=1).all(axis=1)
            
            # Apply trading hours filter
            trading_hours = algo_config['trading_hours']
            time_mask = pd.Series(common_index.strftime('%H:%M')).between(
                trading_hours['start'],
                trading_hours['end']
            ).values
            
            entry_signals = entry_signals & time_mask
            exit_signals = exit_signals & time_mask
            
            # Log results
            logger.info("\nSignal Generation Results:")
            logger.info(f"Total entry signals: {entry_signals.sum()}")
            logger.info(f"Total exit signals: {exit_signals.sum()}")
            
            if entry_signals.sum() == 0:
                logger.warning("\nNo entry signals generated. Analysis:")
                for i, condition in enumerate(algo_config['entry_conditions']):
                    logger.warning(f"Entry condition {i+1}: {condition}")
                    logger.warning(f"Matches: {entry_conditions_met[i].sum()} times")
            
            if exit_signals.sum() == 0:
                logger.warning("\nNo exit signals generated. Analysis:")
                for i, condition in enumerate(algo_config['exit_conditions']):
                    logger.warning(f"Exit condition {i+1}: {condition}")
                    logger.warning(f"Matches: {exit_conditions_met[i].sum()} times")
            
            # Sample signal points
            if entry_signals.sum() > 0:
                entry_points = common_index[entry_signals]
                logger.info("\nSample Entry Points:")
                logger.info(entry_points[:5].tolist())
            
            if exit_signals.sum() > 0:
                exit_points = common_index[exit_signals]
                logger.info("\nSample Exit Points:")
                logger.info(exit_points[:5].tolist())
            
            return entry_signals, exit_signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            raise


    def run_backtest(self, company_name: str, start_date: str = None, end_date: str = None) -> dict:
            """Run the backtest using vectorbt."""
            try:
                # Load algorithm and data
                algo_config = self.get_latest_algorithm(company_name)
                df = self.load_data(company_name, start_date, end_date)
                
                # Calculate indicators
                indicators = self.calculate_indicators(df, algo_config)
                
                # Generate signals
                entry_signals, exit_signals = self.generate_signals(df, indicators, algo_config)
                
                # Handle case where no signals are generated
                if entry_signals.sum() == 0 and exit_signals.sum() == 0:
                    logger.warning("No trading signals generated. Check your algorithm conditions.")
                    return {
                        'Total Return': 0.0,
                        'Sharpe Ratio': 0.0,
                        'Max Drawdown': 0.0,
                        'Total Trades': 0,
                        'Win Rate': 0.0
                    }
                
                # Convert signals to numpy arrays
                entries = entry_signals.to_numpy()
                exits = exit_signals.to_numpy()
                
                # Set size to fixed value based on max_position_size
                size = int(algo_config['initial_capital'] * algo_config['risk_management']['max_position_size'] / df['Close'].iloc[0])
                
                # Run backtest using vectorbt Portfolio
                portfolio = vbt.Portfolio.from_signals(
                    close=df['Close'].to_numpy(),  # Convert to numpy array
                    entries=entries,
                    exits=exits,
                    init_cash=algo_config['initial_capital'],
                    fees=algo_config['commission'],
                    fixed_size=size,  # Use fixed size instead of lambda
                    freq='1T'  # 1-minute frequency
                )
                
                # Save results
                self.save_results(portfolio, df, indicators, company_name)
                
                # Create performance metrics
                self.create_performance_metrics(portfolio, self.output_dir)
                
                # Create trade analysis
                self.create_trade_analysis(portfolio, self.output_dir)
                
                # Calculate risk metrics
                self.calculate_risk_metrics(portfolio, self.output_dir)
                
                return portfolio.stats()
                
            except Exception as e:
                logger.error(f"Error running backtest: {e}")
                raise

    def save_results(self, portfolio: vbt.Portfolio, data: vbt.Data, 
                    indicators: Dict[str, pd.Series], company_name: str):
        """Save backtest results and generate visualizations."""
        try:
            # Create result directory
            result_num = len(list(self.output_dir.glob(f"{company_name}_result-*"))) + 1
            result_dir = self.output_dir / f"{company_name}_result-{result_num}"
            result_dir.mkdir(parents=True, exist_ok=True)
            
            # Save portfolio stats
            stats = portfolio.stats()
            stats_file = result_dir / "stats.json"
            stats_dict = {k: str(v) if isinstance(v, (np.integer, np.floating)) else v 
                         for k, v in stats.items()}
            
            with open(stats_file, 'w') as f:
                json.dump(stats_dict, f, indent=4)
            
            # Save trade history
            trades = portfolio.trades
            trades_file = result_dir / "trades.csv"
            trades.records_readable.to_csv(trades_file)
            
            # Create and save visualizations
            self.create_plots(portfolio, data, indicators, result_dir, company_name)
            
            logger.info(f"Results saved to: {result_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

    def create_plots(self, portfolio: vbt.Portfolio, data: vbt.Data, 
                    indicators: Dict[str, pd.Series], result_dir: Path, company_name: str):
        """Create comprehensive visualization of the backtest results."""
        try:
            # Create main figure with subplots
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.4, 0.2, 0.2, 0.2]
            )
            
            # Plot price and portfolio value
            fig.add_trace(
                go.Scatter(x=data.index, y=data.get('Close'),
                          name='Price', line=dict(color='blue')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=portfolio.value.index, y=portfolio.value,
                          name='Portfolio Value', line=dict(color='green')),
                row=1, col=1
            )
            
            # Plot entry/exit points
            entries = portfolio.entries
            exits = portfolio.exits
            
            fig.add_trace(
                go.Scatter(x=data.index[entries], 
                          y=data.get('Close')[entries],
                          mode='markers',
                          name='Entries',
                          marker=dict(color='green', size=10, symbol='triangle-up')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=data.index[exits],
                          y=data.get('Close')[exits],
                          mode='markers',
                          name='Exits',
                          marker=dict(color='red', size=10, symbol='triangle-down')),
                row=1, col=1
            )
            
            # Plot indicators
            row = 2
            for indicator_name, indicator_values in indicators.items():
                if row > 4:  # Limit to available subplots
                    break
                    
                fig.add_trace(
                    go.Scatter(x=data.index, y=indicator_values,
                              name=indicator_name),
                    row=row, col=1
                )
                row += 1
            
            # Update layout
            fig.update_layout(
                title=f'{company_name} Backtest Results',
                height=1200,
                showlegend=True
            )
            
            # Save plots
            fig.write_html(str(result_dir / 'backtest_analysis.html'))
            
            # Create additional analysis plots
            portfolio.plot().write_html(str(result_dir / 'portfolio_analysis.html'))
            portfolio.plot_trades().write_html(str(result_dir / 'trades_analysis.html'))
            portfolio.plot_drawdown().write_html(str(result_dir / 'drawdown_analysis.html'))
            
            logger.info(f"Created and saved plots to {result_dir}")
            
        except Exception as e:
            logger.error(f"Error creating plots: {e}")
            raise

    def create_performance_metrics(self, portfolio: vbt.Portfolio, result_dir: Path):
        """Create detailed performance metrics report"""
        try:
            metrics = {
                "Returns": {
                    "Total Return (%)": portfolio.total_return * 100,
                    "Annual Return (%)": portfolio.annual_return * 100,
                    "Sharpe Ratio": portfolio.sharpe_ratio,
                    "Sortino Ratio": portfolio.sortino_ratio,
                    "Calmar Ratio": portfolio.calmar_ratio,
                },
                "Risk Metrics": {
                    "Max Drawdown (%)": portfolio.max_drawdown * 100,
                    "Volatility (%)": portfolio.volatility * 100,
                    "Value at Risk (%)": portfolio.var * 100,
                    "Conditional VaR (%)": portfolio.cvar * 100,
                },
                "Trade Statistics": {
                    "Total Trades": len(portfolio.trades),
                    "Win Rate (%)": portfolio.win_rate * 100,
                    "Best Trade (%)": portfolio.trades.returns.max() * 100,
                    "Worst Trade (%)": portfolio.trades.returns.min() * 100,
                    "Avg Win (%)": portfolio.trades.returns[portfolio.trades.returns > 0].mean() * 100,
                    "Avg Loss (%)": portfolio.trades.returns[portfolio.trades.returns < 0].mean() * 100,
                    "Profit Factor": portfolio.profit_factor,
                    "Recovery Factor": portfolio.recovery_factor,
                },
                "Time Analysis": {
                    "Avg Hold Time": portfolio.trades.duration.mean(),
                    "Max Hold Time": portfolio.trades.duration.max(),
                    "Min Hold Time": portfolio.trades.duration.min(),
                    "Time in Market (%)": portfolio.time_in_market * 100,
                }
            }
            
            # Convert to DataFrame for better formatting
            metrics_df = pd.DataFrame({
                (category, metric): value
                for category, category_metrics in metrics.items()
                for metric, value in category_metrics.items()
            }).unstack()
            
            # Save metrics
            metrics_file = result_dir / 'performance_metrics.csv'
            metrics_df.to_csv(metrics_file)
            
            # Create HTML report
            html_report = """
            <html>
            <head>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    table { border-collapse: collapse; width: 100%; }
                    th, td { padding: 8px; text-align: left; border: 1px solid #ddd; }
                    th { background-color: #f2f2f2; }
                    .category { background-color: #e6e6e6; font-weight: bold; }
                </style>
            </head>
            <body>
            <h1>Performance Metrics Report</h1>
            """
            
            for category, category_metrics in metrics.items():
                html_report += f"<h2>{category}</h2>"
                html_report += "<table>"
                html_report += "<tr><th>Metric</th><th>Value</th></tr>"
                
                for metric, value in category_metrics.items():
                    formatted_value = f"{value:.2f}" if isinstance(value, float) else str(value)
                    html_report += f"<tr><td>{metric}</td><td>{formatted_value}</td></tr>"
                
                html_report += "</table>"
            
            html_report += "</body></html>"
            
            # Save HTML report
            report_file = result_dir / 'performance_report.html'
            with open(report_file, 'w') as f:
                f.write(html_report)
            
            logger.info(f"Created performance metrics report at {report_file}")
            
        except Exception as e:
            logger.error(f"Error creating performance metrics: {e}")
            raise

    def calculate_indicators(self, df: pd.DataFrame, algo_config: dict) -> Dict[str, pd.Series]:
            """
            Calculate all required technical indicators based on algorithm configuration.
            
            Args:
                df (pd.DataFrame): DataFrame with OHLCV data
                algo_config (dict): Algorithm configuration containing indicator requirements
                
            Returns:
                Dict[str, pd.Series]: Dictionary of calculated indicators
            """
            indicators = {}
            
            try:
                # Extract unique indicators from conditions
                required_indicators = set()
                for condition in algo_config['entry_conditions'] + algo_config['exit_conditions']:
                    required_indicators.add(condition['indicator'])
                
                logger.info(f"Required indicators: {required_indicators}")
                
                # Add buffer at the start to ensure enough data for calculations
                buffer_size = 100  # Larger buffer to ensure enough data
                df_extended = df.copy()
                
                # Calculate each required indicator
                for indicator in required_indicators:
                    try:
                        if indicator == "SMA":
                            sma = talib.SMA(df_extended['Close'], timeperiod=20)
                            # Remove NaN values at the start
                            valid_sma = sma[buffer_size:]
                            indicators['SMA'] = pd.Series(
                                valid_sma.values,
                                index=df.index[buffer_size:],
                                name='SMA'
                            )
                            
                        elif indicator == "EMA":
                            ema = talib.EMA(df_extended['Close'], timeperiod=50)
                            # Remove NaN values at the start
                            valid_ema = ema[buffer_size:]
                            indicators['EMA'] = pd.Series(
                                valid_ema.values,
                                index=df.index[buffer_size:],
                                name='EMA'
                            )
                            
                        elif indicator == "MACD":
                            macd, signal, hist = talib.MACD(
                                df_extended['Close'],
                                fastperiod=12,
                                slowperiod=26,
                                signalperiod=9
                            )
                            # Remove NaN values at the start
                            valid_index = df.index[buffer_size:]
                            indicators['MACD'] = pd.Series(
                                macd[buffer_size:].values,
                                index=valid_index,
                                name='MACD'
                            )
                            indicators['MACD_Signal'] = pd.Series(
                                signal[buffer_size:].values,
                                index=valid_index,
                                name='MACD_Signal'
                            )
                            indicators['MACD_Hist'] = pd.Series(
                                hist[buffer_size:].values,
                                index=valid_index,
                                name='MACD_Hist'
                            )
                            
                        elif indicator == "WILLR":
                            willr = talib.WILLR(
                                df_extended['High'],
                                df_extended['Low'],
                                df_extended['Close'],
                                timeperiod=14
                            )
                            # Remove NaN values at the start
                            valid_willr = willr[buffer_size:]
                            indicators['WILLR'] = pd.Series(
                                valid_willr.values,
                                index=df.index[buffer_size:],
                                name='WILLR'
                            )
                        
                        # Log indicator statistics
                        if indicator in indicators:
                            ind_series = indicators[indicator]
                            logger.info(f"\n{indicator} Statistics:")
                            logger.info(f"Latest Value: {ind_series.iloc[-1]:.2f}")
                            logger.info(f"Range: {ind_series.min():.2f} to {ind_series.max():.2f}")
                            logger.info(f"Mean: {ind_series.mean():.2f}")
                            logger.info(f"NaN values: {ind_series.isna().sum()}")
                            
                    except Exception as ind_error:
                        logger.error(f"Error calculating {indicator}: {ind_error}")
                        raise
                
                # Align all indicators to the same index
                min_length = min(len(series) for series in indicators.values())
                aligned_index = df.index[-min_length:]
                
                # Trim all indicators to the same length
                for name in indicators:
                    indicators[name] = indicators[name].reindex(aligned_index)
                
                # Final validation
                for name, series in indicators.items():
                    if series.isna().any():
                        logger.error(f"Indicator {name} contains NaN values after alignment")
                        nan_count = series.isna().sum()
                        nan_positions = series.index[series.isna()]
                        logger.error(f"NaN count: {nan_count}")
                        logger.error(f"NaN positions: {nan_positions}")
                        # Fill remaining NaNs with forward fill then backward fill
                        series = series.fillna(method='ffill').fillna(method='bfill')
                        indicators[name] = series
                
                logger.info("\nIndicator Calculation Summary:")
                logger.info(f"Total indicators calculated: {len(indicators)}")
                logger.info("Calculated indicators: " + ", ".join(indicators.keys()))
                
                latest_values = {name: series.iloc[-1] for name, series in indicators.items()}
                logger.info("\nLatest Indicator Values:")
                for name, value in latest_values.items():
                    logger.info(f"{name}: {value:.2f}")
                
                # Verify data alignment
                logger.info("\nData Alignment Check:")
                for name, series in indicators.items():
                    logger.info(f"{name}: {len(series)} points, Index range: {series.index[0]} to {series.index[-1]}")
                
                return indicators
                
            except Exception as e:
                logger.error(f"Error calculating indicators: {e}")
                raise

    def create_trade_analysis(self, portfolio: vbt.Portfolio, result_dir: Path):
        """Create detailed trade analysis with visualizations"""
        try:
            trades = portfolio.trades
            
            # Trade distribution analysis
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Returns Distribution', 'Duration Distribution', 
                              'Returns by Hour', 'Cumulative Returns')
            )
            
            # Returns distribution
            fig.add_trace(
                go.Histogram(x=trades.returns * 100, name='Returns %',
                            nbinsx=50),
                row=1, col=1
            )
            
            # Duration distribution
            fig.add_trace(
                go.Histogram(x=trades.duration, name='Duration',
                            nbinsx=50),
                row=1, col=2
            )
            
            # Returns by hour
            hourly_returns = trades.records_readable.groupby(
                trades.entry_time.dt.hour
            )['return'].mean() * 100
            
            fig.add_trace(
                go.Bar(x=hourly_returns.index, y=hourly_returns.values,
                      name='Avg Return by Hour'),
                row=2, col=1
            )
            
            # Cumulative returns
            fig.add_trace(
                go.Scatter(x=portfolio.cum_returns().index,
                          y=portfolio.cum_returns().values * 100,
                          name='Cumulative Returns %'),
                row=2, col=2
            )
            
            fig.update_layout(
                height=800,
                title_text="Trade Analysis",
                showlegend=True
            )
            
            # Save trade analysis plot
            fig.write_html(str(result_dir / 'trade_analysis.html'))
            
            # Create trade log DataFrame
            trade_log = trades.records_readable.copy()
            trade_log['return_pct'] = trade_log['return'] * 100
            trade_log['duration'] = trade_log['exit_time'] - trade_log['entry_time']
            
            # Add market context
            trade_log['market_direction'] = np.where(
                trade_log['exit_price'] > trade_log['entry_price'],
                'Uptrend', 'Downtrend'
            )
            
            # Save detailed trade log
            trade_log.to_csv(result_dir / 'detailed_trade_log.csv')
            
            logger.info(f"Created trade analysis at {result_dir}")
            
        except Exception as e:
            logger.error(f"Error creating trade analysis: {e}")
            raise

    def calculate_risk_metrics(self, portfolio: vbt.Portfolio, result_dir: Path):
        """Calculate and save detailed risk metrics"""
        try:
            risk_metrics = {
                'Value at Risk (VaR)': {
                    '95%': portfolio.var,
                    '99%': portfolio.var_99,
                },
                'Expected Shortfall (CVaR)': {
                    '95%': portfolio.cvar,
                    '99%': portfolio.cvar_99,
                },
                'Drawdown Analysis': {
                    'Max Drawdown': portfolio.max_drawdown,
                    'Avg Drawdown': portfolio.drawdowns.avg_drawdown,
                    'Max Drawdown Duration': portfolio.drawdowns.max_duration,
                    'Avg Drawdown Duration': portfolio.drawdowns.avg_duration,
                },
                'Risk Ratios': {
                    'Sharpe Ratio': portfolio.sharpe_ratio,
                    'Sortino Ratio': portfolio.sortino_ratio,
                    'Calmar Ratio': portfolio.calmar_ratio,
                    'Information Ratio': portfolio.information_ratio,
                }
            }
            
            # Save risk metrics
            risk_file = result_dir / 'risk_metrics.json'
            with open(risk_file, 'w') as f:
                json.dump(risk_metrics, f, indent=4, default=str)
            
            # Create drawdown plot
            fig = portfolio.plot_drawdown()
            fig.write_html(str(result_dir / 'drawdown_analysis.html'))
            
            logger.info(f"Created risk metrics analysis at {result_dir}")
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            raise

if __name__ == "__main__":
    try:
        # Example usage
        company_name = "ZOMATO"  # Replace with your company name
        start_date = "2023-01-01 09:15:00+05:30"
        end_date = "2024-11-09 15:30:00+05:30"
        
        # Initialize backtester
        backtester = VectorBTBacktester()
        
        # Run backtest
        logger.info(f"Starting backtest for {company_name}")
        stats = backtester.run_backtest(company_name, start_date, end_date)
        
        # Print summary statistics
        print("\nBacktest Results Summary:")
        print("=" * 50)
        print(f"Total Return: {stats['Total Return']*100:.2f}%")
        print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown: {stats['Max Drawdown']*100:.2f}%")
        print(f"Total Trades: {stats['Total Trades']}")
        print(f"Win Rate: {stats['Win Rate']*100:.2f}%")
        
        logger.info("Backtest completed successfully")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise