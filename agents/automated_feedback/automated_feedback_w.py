import ollama
from datetime import datetime
import os
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

class EnhancedTradingFeedbackSystem:
    def __init__(self, model_name='llama2', base_dir='output'):
        self.model_name = model_name
        self.base_dir = Path(base_dir)
        self.data_dir = Path('agents/backtesting_agent/historical_data')  # Updated path
        self.analysis_metrics = {}


    def get_latest_algo_number(self, company_name: str) -> int:
        """Get the latest algorithm number for the company"""
        try:
            algo_dir = self.base_dir / 'algo'
            pattern = f"{company_name}_algorithm-*.json"
            
            # Get all matching algorithm files
            algo_files = list(algo_dir.glob(pattern))
            
            if not algo_files:
                return 1  # Return 1 if no algorithms exist
                
            # Extract numbers and find max
            numbers = []
            for file in algo_files:
                match = re.search(r'algorithm-(\d+)\.json', file.name)
                if match:
                    numbers.append(int(match.group(1)))
                    
            return max(numbers) if numbers else 1
            
        except Exception as e:
            print(f"Error getting latest algorithm number: {e}")
            return 1
        
    def analyze_market_conditions(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze specific market conditions for ZOMATO"""
        try:
            # Calculate daily volatility
            daily_vol = historical_data['Close'].pct_change().std() * np.sqrt(252)
            
            # Calculate average daily range
            daily_range = (historical_data['High'] - historical_data['Low']).mean()
            
            # Calculate average volume
            avg_volume = historical_data['Volume'].mean()
            
            return {
                'daily_volatility': daily_vol,
                'average_range': daily_range,
                'average_volume': avg_volume
            }
        except Exception as e:
            print(f"Error analyzing market conditions: {e}")
            return {}


    def analyze_historical_data(self, company_name: str) -> Dict[str, Any]:
        """Analyze historical data statistics"""
        try:
            # Updated path to match your directory structure
            data_path = self.data_dir / f"{company_name}_minute.csv"
            if not data_path.exists():
                print(f"Historical data file not found at: {data_path}")
                return {}
                
            df = pd.read_csv(data_path)
            metrics = {
                'price_volatility': float(df['Close'].std()),
                'avg_volume': float(df['Volume'].mean()),
                'avg_rsi': float(df['RSI'].mean()) if 'RSI' in df.columns else None,
                'avg_macd_hist': float(df['MACD_Hist'].mean()) if 'MACD_Hist' in df.columns else None,
                'bb_width_avg': float(((df['BB_Upper'] - df['BB_Lower'])/df['BB_Middle']).mean()) 
                    if all(x in df.columns for x in ['BB_Upper', 'BB_Lower', 'BB_Middle']) else None
            }
            return {k: v for k, v in metrics.items() if v is not None and not np.isnan(v)}
        except Exception as e:
            print(f"Error analyzing historical data: {e}")
            return {}

    def parse_algorithm_config(self, company_name: str, algo_num: int) -> Dict[str, Any]:
        """Parse algorithm configuration file"""
        try:
            algo_file = self.base_dir / 'algo' / f"{company_name}_algorithm-{algo_num}.json"
            if not algo_file.exists():
                print(f"Algorithm config file not found at: {algo_file}")
                return {}
                
            with open(algo_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error parsing algorithm config: {e}")
            return {}


    def analyze_backtest_results(self, company_name: str, algo_num: int) -> Dict[str, Any]:
        """Parse and analyze backtest results with improved metric handling"""
        try:
            result_file = self.base_dir / 'backtest_results' / f"{company_name}_algo{algo_num}" / 'summary_stats.txt'
            if not result_file.exists():
                print(f"Backtest results file not found at: {result_file}")
                return {}
                
            with open(result_file, 'r') as f:
                content = f.read()

            # Enhanced metric extraction with default values
            metrics = {
                'total_return': 0.0,
                'annual_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'volatility': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'total_trades': 0,
                'profit_factor': 0.0,
                'exposure_time': 0.0
            }

            patterns = {
                'total_return': r'Total Return: ([-\d.]+)%',
                'annual_return': r'Annual Return: ([-\d.]+)%',
                'sharpe_ratio': r'Sharpe Ratio: ([-\d.nan]+)',
                'max_drawdown': r'Max\. Drawdown: ([-\d.]+)%',
                'win_rate': r'Win Rate: ([-\d.nan]+)%',
                'volatility': r'Volatility \(Ann\.\): ([-\d.]+)%',
                'sortino_ratio': r'Sortino Ratio: ([-\d.nan]+)',
                'calmar_ratio': r'Calmar Ratio: ([-\d.nan]+)',
                'total_trades': r'Total Trades: (\d+)',
                'profit_factor': r'Profit Factor: ([-\d.nan]+)',
                'exposure_time': r'Exposure Time: ([-\d.]+)%'
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    value = match.group(1)
                    try:
                        parsed_value = float(value) if value != 'nan' else 0.0
                        metrics[key] = parsed_value
                    except ValueError:
                        # Keep the default value if parsing fails
                        pass

            return metrics
        except Exception as e:
            print(f"Error analyzing backtest results: {e}")
            return {}

    def generate_strategy_feedback(self, historical_analysis: Dict[str, Any], 
                                algo_config: Dict[str, Any], 
                                backtest_results: Dict[str, Any],
                                is_good: bool) -> str:
        """Generate comprehensive strategy feedback with market context"""
        
        # Enhanced performance assessment
        performance_assessment = []
        
        # Market context analysis
        volatility = historical_analysis.get('price_volatility', 0)
        volume = historical_analysis.get('avg_volume', 0)
        rsi = historical_analysis.get('avg_rsi', 0)
        
        # Assess market conditions
        market_condition = []
        if volatility > 0:
            if volatility > 15:
                market_condition.append("High volatility environment")
            elif volatility > 10:
                market_condition.append("Moderate volatility environment")
            else:
                market_condition.append("Low volatility environment")
                
        if rsi > 70:
            market_condition.append("Overbought conditions")
        elif rsi < 30:
            market_condition.append("Oversold conditions")
        else:
            market_condition.append("Neutral RSI conditions")
        
        # Assess trading activity
        trades = backtest_results.get('total_trades', 0)
        if trades == 0:
            performance_assessment.append("Strategy did not generate any trades - consider relaxing entry conditions")
        elif trades < 10:
            performance_assessment.append("Low trading frequency - consider adjusting signal sensitivity")
        else:
            performance_assessment.append(f"Strategy generated {trades} trades")

        # Format feedback message with market context
        feedback = f"""Trading Strategy Analysis and Feedback

    Market Environment:
    - Volatility Level: {volatility:.2f} ({"High" if volatility > 15 else "Moderate" if volatility > 10 else "Low"})
    - Average Daily Volume: {volume:.0f} shares
    - RSI Level: {rsi:.2f} ({"Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"})
    - Market Conditions: {', '.join(market_condition)}

    Strategy Configuration:
    {json.dumps(algo_config, indent=2)}

    Performance Metrics:
    - Total Trades: {backtest_results.get('total_trades', 0)}
    - Total Return: {backtest_results.get('total_return', 0):.2f}%
    - Sharpe Ratio: {backtest_results.get('sharpe_ratio', 0):.2f}
    - Maximum Drawdown: {backtest_results.get('max_drawdown', 0):.2f}%
    - Win Rate: {backtest_results.get('win_rate', 0):.2f}%

    Key Observations:
    {chr(10).join(f"- {item}" for item in performance_assessment)}

    Market-Specific Recommendations:"""

        # Add market-specific recommendations
        if volatility > 15:
            feedback += """
    1. High Volatility Environment:
    - Widen stop-loss levels to accommodate price swings
    - Reduce position sizes to manage risk
    - Consider using volatility-based position sizing
    - Add volatility filters to entry conditions"""
        elif volatility > 10:
            feedback += """
    1. Moderate Volatility Environment:
    - Use adaptive stop-loss levels
    - Maintain standard position sizes
    - Consider adding trend confirmation signals
    - Monitor volatility breakouts"""
        else:
            feedback += """
    1. Low Volatility Environment:
    - Tighten stop-loss levels
    - Consider increasing position sizes
    - Look for breakout opportunities
    - Add range-trading strategies"""

        feedback += f"\n\n2. Volume Considerations (Avg: {volume:.0f}):"
        if volume > 500000:
            feedback += """
    - High liquidity environment suitable for larger positions
    - Consider using volume-weighted prices
    - Monitor for unusual volume spikes"""
        else:
            feedback += """
    - Monitor liquidity constraints
    - Use smaller position sizes
    - Add volume filters to entry conditions"""

        feedback += f"\n\n3. RSI-Based Adjustments (Current: {rsi:.2f}):"
        if rsi > 70:
            feedback += """
    - Look for potential reversal signals
    - Tighten profit-taking levels
    - Add overbought filters"""
        elif rsi < 30:
            feedback += """
    - Watch for oversold bounces
    - Add trend confirmation signals
    - Consider gradual position building"""
        else:
            feedback += """
    - Maintain standard trading parameters
    - Monitor for trend developments
    - Use RSI for trade confirmation"""

        return feedback

    def process_feedback(self, company_name: str, algo_num: int, is_good: bool) -> None:
        """Main method to process and send feedback"""
        try:
            # Gather all analysis components
            historical_analysis = self.analyze_historical_data(company_name)
            algo_config = self.parse_algorithm_config(company_name, algo_num)
            backtest_results = self.analyze_backtest_results(company_name, algo_num)
            
            # Generate comprehensive feedback
            feedback = self.generate_strategy_feedback(
                historical_analysis,
                algo_config,
                backtest_results,
                is_good
            )
            
            # Send to Ollama for fine-tuning
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": feedback}]
            )
            
            # Log feedback and response
            self._log_feedback(company_name, algo_num, feedback, response)
            
            print(f"Feedback processed successfully at {datetime.now()}")
            return response
            
        except Exception as e:
            print(f"Error processing feedback: {e}")
            return None
    
    def _log_feedback(self, company_name: str, algo_num: int, feedback: str, response: Any) -> None:
        """Log feedback and model responses"""
        log_dir = self.base_dir / 'feedback_logs'
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"{company_name}_feedback_{algo_num}.txt"
        
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Feedback:\n{feedback}\n")
            f.write(f"Model Response:\n{response}\n")

if __name__ == "__main__":
    try:
        feedback_system = EnhancedTradingFeedbackSystem(
            model_name='qwen2.5:1.5b',
            base_dir='output'
        )
        
        company_name = 'ZOMATO'
        # Automatically get latest algorithm number
        algo_num = feedback_system.get_latest_algo_number(company_name)
        is_good = False
        
        print(f"Processing feedback for {company_name} algorithm {algo_num}")
        response = feedback_system.process_feedback(company_name, algo_num, is_good)
        
        if response:
            print("Feedback processed successfully")
            print("Model response:", response)
        else:
            print("Error processing feedback")
            
    except Exception as e:
        print(f"Main execution error: {e}")