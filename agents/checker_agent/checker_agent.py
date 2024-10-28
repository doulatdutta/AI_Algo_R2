import os
import yaml
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CheckerAgent:
    def __init__(self, config_path='config/config.yaml'):
        """Initialize the checker agent with configuration."""
        self.base_path = self.get_base_path()
        self.load_config(config_path)
        self.check_count = 0

    def get_base_path(self):
        """Get the base path of the project."""
        return Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    def load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            config_file = self.base_path / config_path
            with open(config_file, 'r') as file:
                self.config = yaml.safe_load(file)
            
            config = self.config.get('configuration', {})
            self.min_return = float(config.get('Return [%]', 30.00))
            self.max_drawdown = float(config.get('max_drawdown', 20))
            self.max_checks = int(config.get('number of checking', 10))
            self.reduce_percent = float(config.get('reduce %', 5))
            
        except FileNotFoundError:
            logger.error(f"Config file not found at {config_file}")
            raise

    def read_results(self, result_file):
        """Read the result file and parse the Return [%] value."""
        try:
            with open(result_file, 'r') as file:
                content = file.read()
                # Find the Return [%] value
                return_line = [line for line in content.split('\n') if 'Return [%]:' in line]
                if return_line:
                    return_str = return_line[0].split('Return [%]:')[1].split()[0]
                    return float(return_str)
                else:
                    logger.error("Return [%] not found in results file")
                    return 0.0
        except Exception as e:
            logger.error(f"Error reading results file: {str(e)}")
            return 0.0

    def check_results(self, company_name, algo_num=1):
        """Check the Return [%] against the required percentage."""
        result_file = self.base_path / 'output' / 'backtest_results' / f"{company_name}_algorithm-{algo_num}_results.txt"
        checking_file = self.base_path / 'output' / 'checking_results' / f"{company_name}_checking-{algo_num}.txt"

        if not result_file.exists():
            logger.error(f"Result file not found at {result_file}")
            return "not good"

        actual_return = self.read_results(result_file)
        
        # Calculate current threshold based on number of checks
        num_reductions = min(self.check_count // self.max_checks, 1)
        current_threshold = self.min_return - (num_reductions * self.reduce_percent)
        
        # Use 'with' context manager with explicit encoding
        with open(checking_file, 'w', encoding='utf-8') as f:
            f.write(f"Algorithm Check #{self.check_count + 1}\n")
            f.write(f"Required Return: {current_threshold}%\n")
            f.write(f"Max Drawdown: {self.max_drawdown}%\n\n")
            
            is_good = actual_return >= current_threshold
            status = "Good" if is_good else "Not Good"
            
            # Using simple ASCII characters instead of Unicode
            f.write(f"Return [%] Checking: {status} ({actual_return:.2f}% {'>=' if is_good else '<'} {current_threshold}%)\n")
            
            final_result = "good" if is_good else "not good"
            f.write(f"\nFinal result of checking: {final_result}\n")
            
            logger.info(f"Check #{self.check_count + 1}: {final_result} - Return: {actual_return:.2f}% (Required: {current_threshold}%)")
        
        # Increment check count if result is not good
        if not is_good:
            self.check_count += 1
        
        return final_result

# #Example usage

# def main():
#     """Main function to run the checker agent."""
#     try:
#         logger.info("Starting checker agent")
#         checker_agent = CheckerAgent()
        
#         company_name = "ZOMATO"  # Replace with actual company name as needed
        
#         final_status = checker_agent.check_results(company_name, algo_num=1)
#         logger.info(f"Checker agent status: {final_status}")

#     except Exception as e:
#         logger.error(f"Main execution failed: {str(e)}")
#         raise

# if __name__ == "__main__":
#     main()