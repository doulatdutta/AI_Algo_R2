
import ollama
from datetime import datetime
import os
import re
from typing import Dict, Any, Tuple

class TradingFeedbackSystem:
    def __init__(self, model_name='llama2', base_dir='output'):
        self.model_name = model_name
        self.base_dir = base_dir
        
    def read_files(self, company_name: str, algo_num: int) -> Tuple[Dict[str, Any], bool]:
        """
        Read and parse all relevant files for a specific algorithm
        """
        try:
            # Construct file paths
            algo_file = f"{self.base_dir}/algo/{company_name}_algorithm-{algo_num}.txt"
            checking_file = f"{self.base_dir}/checking_results/{company_name}_checking-{algo_num}.txt"
            backtest_file = f"{self.base_dir}/backtest_results/{company_name}_algorithm-{algo_num}_results.txt"
            
            # Read algorithm conditions
            with open(algo_file, 'r') as f:
                algo_content = f.read()
                
            # Read checking results
            with open(checking_file, 'r') as f:
                checking_content = f.read()
                is_good = 'Final result of checking: good' in checking_content.lower()
                
            # Read backtest results
            with open(backtest_file, 'r') as f:
                backtest_content = f.read()
            
            # Parse metrics from backtest results
            metrics = self._parse_backtest_metrics(backtest_content)
            
            return {
                'algo_content': algo_content,
                'backtest_metrics': metrics,
                'checking_result': checking_content
            }, is_good
            
        except FileNotFoundError as e:
            print(f"Error reading files: {e}")
            return None, False
            
    def _parse_backtest_metrics(self, content: str) -> Dict[str, Any]:
        """
        Parse metrics from backtest result content
        """
        metrics = {}
        
        # Define patterns for metric extraction
        patterns = {
            'return': r'Return \[%\]: ([-\d.]+)',
            'sharpe_ratio': r'Sharpe Ratio: ([-\d.nan]+)',
            'max_drawdown': r'Max\. Drawdown \[%\]: ([-\d.]+)',
            'win_rate': r'Win Rate \[%\]: ([-\d.nan]+)',
            'profit_factor': r'Profit Factor: ([-\d.nan]+)',
            'total_trades': r'# Trades: (\d+)',
        }
        
        # Extract metrics using patterns
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                value = match.group(1)
                try:
                    metrics[key] = float(value) if value != 'nan' else None
                except ValueError:
                    metrics[key] = None
                    
        return metrics

    def format_feedback(self, data: Dict[str, Any], is_good: bool) -> Dict[str, str]:
        """
        Format the feedback message for Ollama
        """
        metrics = data['backtest_metrics']
        
        return {
            "role": "user",
            "content": f"""Trading Strategy Performance Review

Strategy Configuration:
{data['algo_content']}

Performance Metrics:
- Return: {metrics.get('return', 'N/A')}%
- Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}
- Max Drawdown: {metrics.get('max_drawdown', 'N/A')}%
- Win Rate: {metrics.get('win_rate', 'N/A')}%
- Profit Factor: {metrics.get('profit_factor', 'N/A')}
- Total Trades: {metrics.get('total_trades', 'N/A')}

Strategy Assessment: {'GOOD' if is_good else 'NOT GOOD'}

{'Positive Aspects:' if is_good else 'Areas for Improvement:'}
1. {'Strong returns and risk management' if is_good else 'Risk Management: Consider adjusting indicator thresholds'}
2. {'Effective entry/exit timing' if is_good else 'Entry/Exit Timing: Review signal combinations'}
3. {'Well-balanced indicator combination' if is_good else 'Technical Indicators: Optimize parameter values'}

{'Maintain:' if is_good else 'For the next strategy, focus on:'}
1. {'Current risk management approach' if is_good else 'More stringent entry/exit conditions'}
2. {'Indicator parameter values' if is_good else 'Better trend confirmation signals'}
3. {'Overall strategy structure' if is_good else 'Improved risk-reward ratio'}

Use this feedback to {'maintain high performance' if is_good else 'generate improved trading strategies'}."""
        }

    def send_feedback(self, company_name: str, algo_num: int) -> None:
        """
        Process and send feedback to Ollama
        """
        # Read and parse files
        data, is_good = self.read_files(company_name, algo_num)
        if not data:
            return
        
        # Format feedback
        feedback = self.format_feedback(data, is_good)
        
        try:
            # Send feedback to Ollama
            response = ollama.chat(
                model=self.model_name,
                messages=[feedback]
            )
            
            # Log the feedback
            self._log_feedback(company_name, algo_num, feedback, response)
            
            print(f"Feedback sent to Ollama successfully at {datetime.now()}")
            return response
        except Exception as e:
            print(f"Error sending feedback to Ollama: {e}")
            return None
    
    def _log_feedback(self, company_name: str, algo_num: int, feedback: Dict, response: Any) -> None:
        """
        Log feedback events
        """
        log_dir = f"{self.base_dir}/feedback_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = f"{log_dir}/{company_name}_feedback_{algo_num}.txt"
        
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Feedback Sent:\n{feedback['content']}\n")
            f.write(f"Model Response:\n{response}\n")

# Example usage in main.py
if __name__ == "__main__":
    def process_algo_feedback(company_name: str, algo_num: int):
        """
        Main function to process and send feedback
        """
        feedback_system = TradingFeedbackSystem(
            model_name='llama3.2',  # or your specific model
            base_dir='output'     # your base directory
        )
        
        response = feedback_system.send_feedback(company_name, algo_num)
        
        if response:
            print("Feedback processed successfully")
            print("Ollama's response:", response)
        else:
            print("Error processing feedback")
    
    ### Example usage:
    # process_algo_feedback('ZOMATO', 1)
