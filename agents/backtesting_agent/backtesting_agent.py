import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # Add this import
from backtesting import Backtest, Strategy
import logging
from datetime import datetime
from pathlib import Path
import talib
import glob
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DynamicStrategy(Strategy):
    """A dynamic trading strategy that can handle various technical indicators."""
    
    def init(self):
        """Initialize all technical indicators based on conditions."""
        self.indicators = {}
        self.initialize_required_indicators()
        self.last_trade_date = None
        self.daily_trades = []

    def next(self):
        """Execute trading logic based on completed trade cycles."""
        try:
            current_datetime = self.data.index[-1]
            current_date = current_datetime.date()
            
            # Initialize new day tracking
            if self.last_trade_date != current_date:
                # Close any overnight positions and count them as completed trades
                if self.position:
                    # Record the trade completion before closing
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
            
            # Get current position status
            current_position = self.position.size if self.position else 0
            
            # Check trading conditions
            buy_signal = all(self.evaluate_condition(condition) for condition in self.buy_conditions)
            sell_signal = all(self.evaluate_condition(condition) for condition in self.sell_conditions)
            
            # Execute trades based on position and signals
            if current_position == 0:  # No position - can start either a buy or sell cycle
                if buy_signal:  # Start a buy cycle
                    self.buy()
                    self.daily_trades[-1]['trades'] += 1
                    logger.info(f"{current_datetime}: Opening buy position")
                elif sell_signal:  # Start a sell cycle
                    self.sell()
                    self.daily_trades[-1]['trades'] += 1
                    logger.info(f"{current_datetime}: Opening sell position")
                    
            elif current_position > 0:  # In a buy cycle - wait for sell signal
                if sell_signal:  # Complete the buy cycle
                    self.position.close()
                    self.daily_trades[-1]['buy_trades'] += 1
                    logger.info(f"{current_datetime}: Completing buy cycle")
                    
            elif current_position < 0:  # In a sell cycle - wait for buy signal
                if buy_signal:  # Complete the sell cycle
                    self.position.close()
                    self.daily_trades[-1]['sell_trades'] += 1
                    logger.info(f"{current_datetime}: Completing sell cycle")
            
            # End of day processing (15:29 is the last minute of trading)
            if current_datetime.time().strftime('%H:%M') == '15:29':
                # Close any open positions
                if self.position:
                    if self.position.size > 0:
                        self.daily_trades[-1]['buy_trades'] += 1
                    elif self.position.size < 0:
                        self.daily_trades[-1]['sell_trades'] += 1
                    self.position.close()
                
                # Calculate daily profit/loss
                if self.daily_trades:
                    daily_pnl = self.equity - self.daily_trades[-1]['start_equity']
                    daily_return = (daily_pnl / self.daily_trades[-1]['start_equity']) * 100
                    self.daily_trades[-1]['profit'] = daily_return
                    self.daily_trades[-1]['final_equity'] = self.equity
                    
        except Exception as e:
            logger.error(f"Error in next(): {str(e)}")
            raise

    def parse_indicator_params(self, raw_condition):
        """Parse indicator name and parameters from condition string."""
        try:
            match = re.match(r"([a-zA-Z]+)\(([^)]+)\)", raw_condition)
            if match:
                indicator_name = match.group(1).lower()
                params_str = match.group(2)
                
                # Special handling for MACD comparison with signal line
                if indicator_name == 'macd' and 'signal' in raw_condition.lower():
                    # Extract MACD parameters before "signal line" text
                    params_match = re.match(r"(\d+),\s*(\d+)", params_str)
                    if params_match:
                        fast_period = params_match.group(1)
                        slow_period = params_match.group(2)
                        signal_period = "9"  # Default signal period
                        return indicator_name, [fast_period, slow_period, signal_period]
                
                # Normal parameter parsing
                params = [p.strip() for p in params_str.split(',')]
                return indicator_name, params
            return None, None
            
        except Exception as e:
            logger.error(f"Error parsing indicator params: {str(e)}")
            return None, None

    def initialize_required_indicators(self):
        """Initialize all indicators needed for the strategy."""
        # Extract unique indicators from conditions
        all_conditions = (self.buy_conditions + self.sell_conditions)
        processed_indicators = set()
        
        for condition in all_conditions:
            if 'raw_condition' not in condition or condition['raw_condition'] == "**":
                continue
            
            raw_cond = condition['raw_condition'].split('"')[1] if '"' in condition['raw_condition'] else condition['raw_condition']
            indicator_name, params = self.parse_indicator_params(raw_cond)
            
            if indicator_name and (indicator_name, tuple(params)) not in processed_indicators:
                self.initialize_indicator(indicator_name, params)
                processed_indicators.add((indicator_name, tuple(params)))

    def evaluate_condition(self, condition):
        """Evaluate a single trading condition."""
        if 'raw_condition' not in condition or condition['raw_condition'] == "**":
            return True
            
        try:
            raw_cond = condition['raw_condition'].split('"')[1] if '"' in condition['raw_condition'] else condition['raw_condition']
            indicator_name, params = self.parse_indicator_params(raw_cond)
            
            if not indicator_name:
                return True
                
            # Extract comparison part
            comparison_part = raw_cond.split(')')[-1].strip()
            operator_match = re.match(r"([<>=]+)\s*([\d.-]+)", comparison_part)
            
            if not operator_match:
                return True
                
            operator = operator_match.group(1)
            value = float(operator_match.group(2))
            
            # Get indicator value
            indicator_key = self.get_indicator_key(indicator_name, params)
            if indicator_key not in self.indicators:
                return True
                
            current_value = self.indicators[indicator_key][-1]
            
            # Add logging for debugging
            logger.info(f"Evaluating {indicator_name}: Current Value = {current_value}, Target = {operator} {value}")
            
            # Compare values
            if operator == '>':
                return current_value > value
            elif operator == '<':
                return current_value < value
            elif operator == '>=':
                return current_value >= value
            elif operator == '<=':
                return current_value <= value
            elif operator == '==':
                return current_value == value
                
        except Exception as e:
            logger.error(f"Error evaluating condition {condition}: {str(e)}")
            return False
            
        return True

    def get_indicator_key(self, indicator_name, params):
        """Generate the indicator key based on name and parameters."""
        if indicator_name == 'macd':
            return f'macd_{params[0]}_{params[1]}_{params[2]}'
        elif indicator_name in ['bollinger', 'boll']:
            return f'boll_upper_{params[0]}_{params[1]}'
        elif indicator_name == 'stoch':
            return f'stoch_k_{params[0]}_{params[1]}'
        else:
            return f'{indicator_name}_{params[0]}'



    def initialize_indicator(self, indicator_name, params):
        """Initialize a specific technical indicator."""
        try:
            close = self.data.Close
            high = self.data.High
            low = self.data.Low
            volume = self.data.Volume
            
            # Moving Averages
            if indicator_name == 'sma':
                period = int(params[0])
                self.indicators[f'sma_{period}'] = self.I(talib.SMA, close, timeperiod=period)
            
            elif indicator_name == 'ema':
                period = int(params[0])
                self.indicators[f'ema_{period}'] = self.I(talib.EMA, close, timeperiod=period)
            
            elif indicator_name == 'wma':
                period = int(params[0])
                self.indicators[f'wma_{period}'] = self.I(talib.WMA, close, timeperiod=period)
            
            elif indicator_name == 'vwap':
                period = int(params[0])
                typical_price = (high + low + close) / 3
                vwap = self.I(lambda: talib.SMA(typical_price * volume, period) / talib.SMA(volume, period))
                self.indicators[f'vwap_{period}'] = vwap
            
            # Momentum Indicators
            elif indicator_name == 'rsi':
                period = int(params[0])
                self.indicators[f'rsi_{period}'] = self.I(talib.RSI, close, timeperiod=period)
            
            elif indicator_name == 'macd':
                # Set default values if not all parameters are provided
                fast_period = int(params[0]) if len(params) > 0 else 12
                slow_period = int(params[1]) if len(params) > 1 else 26
                signal_period = int(params[2]) if len(params) > 2 else 9
                
                macd_line, signal_line, hist = talib.MACD(
                    close, 
                    fastperiod=fast_period, 
                    slowperiod=slow_period, 
                    signalperiod=signal_period
                )
                
                self.indicators[f'macd_{fast_period}_{slow_period}_{signal_period}'] = self.I(lambda: macd_line)
                self.indicators[f'macdsignal_{fast_period}_{slow_period}_{signal_period}'] = self.I(lambda: signal_line)
                    
            elif indicator_name == 'stoch':
                # Default values if not provided
                k_period = int(params[0]) if len(params) > 0 else 14
                d_period = int(params[1]) if len(params) > 1 else 3
                
                slowk, slowd = talib.STOCH(high, low, close, 
                                        fastk_period=k_period, 
                                        slowk_period=d_period)
                self.indicators[f'stoch_k_{k_period}_{d_period}'] = self.I(lambda: slowk)
                self.indicators[f'stoch_d_{k_period}_{d_period}'] = self.I(lambda: slowd)
            
            elif indicator_name == 'willr':
                period = int(params[0])
                self.indicators[f'willr_{period}'] = self.I(talib.WILLR, high, low, close, timeperiod=period)
            
            # Volatility Indicators
            elif indicator_name == 'bollinger':
                # Default values if not provided
                period = int(params[0]) if len(params) > 0 else 20
                dev = float(params[1]) if len(params) > 1 else 2.0
                
                upper, middle, lower = talib.BBANDS(close, 
                                                timeperiod=period, 
                                                nbdevup=dev, 
                                                nbdevdn=dev)
                self.indicators[f'boll_upper_{period}_{dev}'] = self.I(lambda: upper)
                self.indicators[f'boll_lower_{period}_{dev}'] = self.I(lambda: lower)
                self.indicators[f'boll_middle_{period}_{dev}'] = self.I(lambda: middle)
            
            elif indicator_name == 'atr':
                period = int(params[0])
                self.indicators[f'atr_{period}'] = self.I(talib.ATR, high, low, close, timeperiod=period)
            
            elif indicator_name == 'keltner':
                period = int(params[0]) if len(params) > 0 else 20
                atr_mult = float(params[1]) if len(params) > 1 else 2.0
                
                basis = self.I(talib.EMA, close, timeperiod=period)
                atr = self.I(talib.ATR, high, low, close, timeperiod=period)
                self.indicators[f'keltner_upper_{period}'] = self.I(lambda: basis + (atr * atr_mult))
                self.indicators[f'keltner_lower_{period}'] = self.I(lambda: basis - (atr * atr_mult))
            
            # Volume Indicators
            elif indicator_name == 'obv':
                self.indicators['obv'] = self.I(talib.OBV, close, volume)
            
            elif indicator_name == 'mfi':
                period = int(params[0])
                self.indicators[f'mfi_{period}'] = self.I(talib.MFI, high, low, close, volume, timeperiod=period)
            
            # Trend Indicators
            elif indicator_name == 'adx':
                period = int(params[0])
                self.indicators[f'adx_{period}'] = self.I(talib.ADX, high, low, close, timeperiod=period)
            
            elif indicator_name == 'cci':
                period = int(params[0])
                self.indicators[f'cci_{period}'] = self.I(talib.CCI, high, low, close, timeperiod=period)
                
        except Exception as e:
            logger.error(f"Error initializing indicator {indicator_name}: {str(e)}")
            raise


class BacktestingAgent:
    def __init__(self):
        self.base_path = Path(os.getcwd())
        self.output_dir = self.base_path / 'output'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'backtest_results').mkdir(exist_ok=True)
        self.companies = []  # Initialize empty companies list

    def get_latest_algorithm(self, company_name):
        """Find the latest algorithm file for the given company."""
        pattern = str(self.output_dir / 'algo' / f"{company_name}_algorithm-*.json")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No algorithm files found for {company_name}")
            
        latest_file = max(files, key=lambda x: int(x.split('-')[-1].split('.')[0]))
        algo_num = int(latest_file.split('-')[-1].split('.')[0])
        
        logger.info(f"Using algorithm {algo_num} for {company_name}")
        return latest_file, algo_num

    def load_algorithm_conditions(self, company_name):
        """Load the latest algorithm conditions."""
        algo_file, algo_num = self.get_latest_algorithm(company_name)
        with open(algo_file, 'r') as f:
            conditions = json.load(f)
            self.buy_conditions = conditions.get('buy_conditions', [])
            self.sell_conditions = conditions.get('sell_conditions', [])
        return algo_num


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
        
        with open(output_file, 'w') as f:
            # Write algorithm conditions
            f.write("Algorithm -\n")
            f.write("Buy Conditions:\n")
            buy_conditions = "; ".join(f"- {cond['raw_condition']}" for cond in self.buy_conditions)
            f.write(f"{buy_conditions}\n\n")
            
            f.write("Sell Conditions:\n")
            sell_conditions = "; ".join(f"- {cond['raw_condition']}" for cond in self.sell_conditions)
            f.write(f"{sell_conditions}\n\n")
            
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
            
            # Write daily results in reverse chronological order
            for date in sorted(daily_stats.keys(), reverse=True):
                day_stat = daily_stats[date]
                f.write(f"\t{date} Profit: {day_stat['profit']:.2f}% ; "
                    f"total trade: {day_stat['total_trades']} ; "
                    f"buy_trade - {day_stat['buy_trades']} ; "
                    f"sell_trade - {day_stat['sell_trades']}\n")
                        
            # Write overall statistics
            f.write("\nOverall Statistics:\n")
            f.write(f"Start: {stats['Start']}\n")
            f.write(f"End: {stats['End']}\n")
            f.write(f"Duration: {stats['Duration']}\n")
            f.write(f"Exposure Time [%]: {stats['Exposure Time [%]']:.2f}\n")
            f.write(f"Equity Final [$]: {stats['Equity Final [$]']:.2f}\n")
            f.write(f"Equity Peak [$]: {stats['Equity Peak [$]']:.2f}\n")
            f.write(f"Return [%]: {stats['Return [%]']:.2f}\n")
            f.write(f"Buy & Hold Return [%]: {stats['Buy & Hold Return [%]']:.2f}\n")
            f.write(f"Return (Ann.) [%]: {stats['Return (Ann.) [%]']:.2f}\n")
            f.write(f"Volatility (Ann.) [%]: {stats['Volatility (Ann.) [%]']:.2f}\n")
            f.write(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}\n")
            f.write(f"Sortino Ratio: {stats['Sortino Ratio']:.2f}\n")
            f.write(f"Calmar Ratio: {stats['Calmar Ratio']:.2f}\n")
            f.write(f"Max. Drawdown [%]: {stats['Max. Drawdown [%]']:.2f}\n")
            f.write(f"Avg. Drawdown [%]: {stats['Avg. Drawdown [%]']:.2f}\n")
            f.write(f"Max. Drawdown Duration: {stats['Max. Drawdown Duration']}\n")
            f.write(f"Avg. Drawdown Duration: {stats['Avg. Drawdown Duration']}\n")
            f.write(f"# Trades: {stats['# Trades']}\n")
            f.write(f"Win Rate [%]: {stats['Win Rate [%]']:.2f}\n")
            f.write(f"Best Trade [%]: {stats['Best Trade [%]']:.2f}\n")
            f.write(f"Worst Trade [%]: {stats['Worst Trade [%]']:.2f}\n")
            f.write(f"Avg. Trade [%]: {stats['Avg. Trade [%]']:.2f}\n")
            f.write(f"Max. Trade Duration: {stats['Max. Trade Duration']}\n")
            f.write(f"Avg. Trade Duration: {stats['Avg. Trade Duration']}\n")
            f.write(f"Profit Factor: {stats['Profit Factor']:.2f}\n")
            f.write(f"Expectancy [%]: {stats['Expectancy [%]']:.2f}\n")
            f.write(f"SQN: {stats['SQN']:.2f}\n")
        
        logger.info(f"Results saved to {output_file}")

    def check_conditions(self, row, conditions):
        """Evaluate parsed conditions on a row of data."""
        for condition in conditions:
            if 'raw_condition' not in condition or condition['raw_condition'] == "**":
                continue
            try:
                _, cond = condition['raw_condition'].split('.', 1)
                indicator, operator, threshold = self.parse_condition(cond.strip().strip('"'))
                
                # Log the condition details and row data for debugging
                if indicator not in row:
                    logger.error(f"Indicator {indicator} not found in data columns.")
                    return False
                value = row[indicator]
                logger.info(f"Evaluating Condition: {indicator} {operator} {threshold}, Row Value: {value}")
                
                # Check conditions with added logs
                if operator == '>':
                    if not (value > threshold):
                        logger.info(f"Condition failed: {indicator} ({value}) not > {threshold}")
                        return False
                elif operator == '<':
                    if not (value < threshold):
                        logger.info(f"Condition failed: {indicator} ({value}) not < {threshold}")
                        return False
                elif operator == '>=':
                    if not (value >= threshold):
                        logger.info(f"Condition failed: {indicator} ({value}) not >= {threshold}")
                        return False
                elif operator == '<=':
                    if not (value <= threshold):
                        logger.info(f"Condition failed: {indicator} ({value}) not <= {threshold}")
                        return False
            except Exception as e:
                logger.error(f"Error evaluating condition '{condition}': {str(e)}")
                return False
        return True


    def run_backtest(self, company_name, algo_num=1):
        try:
            # Load historical data
            df = pd.read_csv(self.base_path / 'agents' / 'backtesting_agent' / 
                            'historical_data' / f"{company_name}_minute.csv")
            
            # Ensure proper datetime handling
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df.sort_values('Datetime', inplace=True)
            df.set_index('Datetime', inplace=True)
            
            logger.info(f"Loaded {len(df)} minute bars for backtesting")
            
            # Load algorithm conditions
            algo_num = self.load_algorithm_conditions(company_name)
            
            # Create strategy class with conditions
            class CurrentStrategy(DynamicStrategy):
                buy_conditions = self.buy_conditions
                sell_conditions = self.sell_conditions
            
            # Create and run backtest using minute data
            backtest = Backtest(
                df,
                CurrentStrategy,
                cash=100000,
                commission=.002,
                exclusive_orders=True,
                hedging=False,
            )
            
            stats = backtest.run()
            
            # Save results and plot
            self.save_results(stats, company_name, algo_num)
            self.plot_profits(stats, company_name, algo_num)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error during backtesting: {str(e)}")
            raise



def main():
    """Main function to run the backtesting agent."""
    try:
        logger.info("Starting backtesting agent")
        backtest_agent = BacktestingAgent()
        
        # Use the first company from config or default to TATASTEEL
        company_name = backtest_agent.companies[0] if backtest_agent.companies else "TATASTEEL"
        
        # Run backtest
        backtest_agent.run_backtest(company_name, algo_num=1)

        logger.info("Backtesting completed successfully")

    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
