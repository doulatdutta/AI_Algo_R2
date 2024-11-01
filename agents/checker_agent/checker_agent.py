# Add these imports at the top of your file
import os
import yaml
import logging
import traceback  # Add this for detailed error tracking
from datetime import datetime
from pathlib import Path

# # Modify the logging configuration
# logging.basicConfig(
#     level=logging.DEBUG,  # Change to DEBUG level for more detailed logs
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

logger = logging.getLogger('CheckerAgent')
logger.setLevel(logging.DEBUG)

class CheckerAgent:
    def __init__(self, config_path='config/config.yaml'):
        """Initialize the checker agent with configuration."""
        try:
            self.base_path = self.get_base_path()
            logger.debug(f"Base path: {self.base_path}")
            self.setup_logging()  # Add this line
            self.load_config(config_path)
            self.check_count = 0
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}\n{traceback.format_exc()}")
            raise


    def setup_logging(self):
        """Set up logging configuration."""
        try:
            # Create logs directory if it doesn't exist
            log_dir = self.base_path / 'output' / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)

            # Create file handler
            log_file = log_dir / 'checker_agent.log'
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)

            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # Remove any existing handlers to avoid duplicates
            logger.handlers.clear()

            # Add handlers to logger
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

            logger.debug("Logging setup completed")

        except Exception as e:
            print(f"Error setting up logging: {str(e)}\n{traceback.format_exc()}")
            raise

    def get_base_path(self):
        """Get the base path of the project."""
        current_file = Path(__file__).resolve()
        base_path = current_file.parent.parent.parent
        logger.debug(f"Calculated base path: {base_path}")
        return base_path

    def load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            config_file = self.base_path / config_path
            logger.debug(f"Looking for config file at: {config_file}")
            
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found at {config_file}")
            
            with open(config_file, 'r') as file:
                self.config = yaml.safe_load(file)
            
            config = self.config.get('configuration', {})
            logger.debug(f"Loaded configuration: {config}")
            
            self.min_return = float(config.get('Return [%]', 30.00))
            self.max_drawdown = float(config.get('max_drawdown', 20))
            self.max_checks = int(config.get('number of checking', 10))
            self.reduce_percent = float(config.get('reduce %', 5))
            
            logger.debug(f"Parsed config values: min_return={self.min_return}, " 
                        f"max_drawdown={self.max_drawdown}, "
                        f"max_checks={self.max_checks}, "
                        f"reduce_percent={self.reduce_percent}")
            
        except Exception as e:
            logger.error(f"Config loading failed: {str(e)}\n{traceback.format_exc()}")
            raise

    def get_latest_algo_number(self, company_name):
        """Find the latest algorithm number for the given company."""
        try:
            backtest_path = self.base_path / 'output' / 'backtest_results'
            logger.debug(f"Searching for algorithms in: {backtest_path}")
            
            if not backtest_path.exists():
                logger.error(f"Backtest results directory not found: {backtest_path}")
                return None
            
            pattern = f"{company_name}_algo*"
            matching_folders = list(backtest_path.glob(pattern))
            logger.debug(f"Found matching folders: {matching_folders}")
            
            if not matching_folders:
                logger.error(f"No backtest results found for {company_name}")
                return None
            
            algo_numbers = []
            for folder in matching_folders:
                try:
                    # Extract the number after 'algo'
                    algo_str = str(folder).split('algo')[-1]
                    # Remove any non-numeric characters
                    algo_str = ''.join(filter(str.isdigit, algo_str))
                    algo_num = int(algo_str)
                    algo_numbers.append(algo_num)
                    logger.debug(f"Found algorithm number: {algo_num}")
                except (ValueError, IndexError) as e:
                    logger.warning(f"Could not parse algo number from folder {folder}: {str(e)}")
                    continue
            
            if not algo_numbers:
                logger.error("No valid algorithm numbers found")
                return None
            
            latest_algo = max(algo_numbers)
            logger.debug(f"Latest algorithm number: {latest_algo}")
            return latest_algo
            
        except Exception as e:
            logger.error(f"Error finding latest algo number: {str(e)}\n{traceback.format_exc()}")
            return None

    def get_latest_backtest_folder(self, company_name, algo_num):
        """Find the latest backtest results folder for the given company and algorithm."""
        backtest_path = self.base_path / 'output' / 'backtest_results'
        pattern = f"{company_name}_algo{algo_num}_*"
        
        # Get all matching folders
        matching_folders = list(backtest_path.glob(pattern))
        
        if not matching_folders:
            logger.error(f"No backtest results found for {company_name} algo{algo_num}")
            return None
        
        # Sort by folder name (which includes timestamp) to get the latest
        latest_folder = sorted(matching_folders)[-1]
        return latest_folder

    def parse_performance_summary(self, content):
        """Parse the performance summary and extract relevant metrics."""
        metrics = {
            'return': 0.0,
            'drawdown': 0.0,
            'total_trades': 0
        }
        
        try:
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                
                if 'Total Return:' in line and 'Annual Return:' not in line:
                    value_str = line.split(':')[1].strip().replace('%', '').strip()
                    if value_str.lower() != 'nan':
                        metrics['return'] = float(value_str)
                        
                elif 'Max Drawdown:' in line:
                    value_str = line.split(':')[1].strip().replace('%', '').replace('-', '').strip()
                    if value_str.lower() != 'nan':
                        metrics['drawdown'] = float(value_str)
                        
                elif 'Total Trades:' in line:
                    value_str = line.split(':')[1].strip()
                    if value_str.lower() != 'nan':
                        metrics['total_trades'] = int(value_str)
            
            logger.info(f"Parsed metrics - Return: {metrics['return']}%, "
                    f"Drawdown: {metrics['drawdown']}%, "
                    f"Total Trades: {metrics['total_trades']}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error parsing performance summary: {str(e)}")
            return metrics

    def read_results(self, result_file):
        """Read and parse the performance summary file."""
        try:
            logger.debug(f"Reading results file: {result_file}")
            with open(result_file, 'r') as file:
                content = file.read()
                logger.debug("Successfully read the file")
                # Log the first few lines for debugging
                preview = '\n'.join(content.split('\n')[:10])
                logger.debug(f"First few lines of content:\n{preview}")
                return self.parse_performance_summary(content)
        except Exception as e:
            logger.error(f"Error reading results file: {str(e)}\n{traceback.format_exc()}")
            return {'return': 0.0, 'drawdown': 0.0}

    def get_current_thresholds(self):
        """Calculate current thresholds based on number of checks."""
        try:
            num_reductions = max(0, self.check_count // self.max_checks)  # Ensure non-negative
            
            # Calculate reduced return threshold (minimum 0)
            current_return_threshold = max(0, self.min_return - (num_reductions * self.reduce_percent))
            
            # Calculate increased drawdown threshold (with upper limit)
            max_allowed_drawdown = 100  # Maximum possible drawdown percentage
            current_drawdown_threshold = min(
                max_allowed_drawdown,
                self.max_drawdown + (num_reductions * self.reduce_percent)
            )
            
            logger.info(
                f"Current thresholds - Return: {current_return_threshold}%, "
                f"Drawdown: {current_drawdown_threshold}% "
                f"(After {num_reductions} reductions)"
            )
            
            return current_return_threshold, current_drawdown_threshold
        except Exception as e:
            logger.error(f"Error calculating thresholds: {str(e)}")
            return self.min_return, self.max_drawdown  # Return original values if calculation fails

    def check_results(self, company_name, algo_num=None):
        """Check the backtest results against the required thresholds."""
        try:
            # Input validation
            if not company_name:
                logger.error("Company name cannot be empty")
                return "not good"

            if algo_num is not None and not isinstance(algo_num, int):
                logger.error("Algorithm number must be an integer")
                return "not good"
            
            # Get the latest algo number if none provided
            if algo_num is None:
                algo_num = self.get_latest_algo_number(company_name)
                if algo_num is None:
                    logger.error(f"No valid algorithm results found for {company_name}")
                    return "not good"
            
            logger.info(f"Using algorithm #{algo_num} for {company_name}")
            
            # Set up file paths
            result_file = self.base_path / 'output' / 'backtest_results' / f"{company_name}_algo{algo_num}" / 'summary_stats.txt'
            checking_file = self.base_path / 'output' / 'checking_results' / f"{company_name}_checking-{algo_num}.txt"

            if not result_file.exists():
                logger.error(f"Result file not found at {result_file}")
                return "not good"
            
            # Read and parse results
            metrics = self.read_results(result_file)
            actual_return = metrics['return']
            actual_drawdown = metrics['drawdown']

            # Additional validation for no-trade condition
            total_trades = metrics.get('total_trades', 0)
            if total_trades == 0:
                logger.warning(f"No trades were executed in algorithm #{algo_num}")
                
            # Get current thresholds
            current_return_threshold, current_drawdown_threshold = self.get_current_thresholds()

            # Perform checks
            return_check = actual_return >= current_return_threshold
            drawdown_check = actual_drawdown <= current_drawdown_threshold
            is_good = return_check and drawdown_check

            # Ensure the checking_results directory exists
            checking_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write detailed results
            with open(checking_file, 'w', encoding='utf-8') as f:
                # Write header with timestamp
                f.write(f"Algorithm Check #{self.check_count + 1}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Company: {company_name}\n")
                f.write(f"Algorithm: #{algo_num}\n\n")
                
                # Write trade information
                f.write("Trading Information:\n")
                f.write(f"Total Trades: {total_trades}\n")
                if total_trades == 0:
                    f.write("Warning: No trades were executed during this period\n\n")
                
                # Write threshold information
                f.write(f"Current Thresholds after {self.check_count // self.max_checks} reductions:\n")
                f.write(f"Required Return: {current_return_threshold:.2f}%\n")
                f.write(f"Max Drawdown: {current_drawdown_threshold:.2f}%\n\n")
                
                # Write check results
                f.write("Check Results:\n")
                f.write(f"Return: {'PASS' if return_check else 'FAIL'} ")
                f.write(f"({actual_return:.2f}% {'>=' if return_check else '<'} {current_return_threshold:.2f}%)\n")
                f.write(f"Drawdown: {'PASS' if drawdown_check else 'FAIL'} ")
                f.write(f"({actual_drawdown:.2f}% {'<=' if drawdown_check else '>'} {current_drawdown_threshold:.2f}%)\n\n")
                
                # Write final result
                final_result = "good" if is_good else "not good"
                f.write(f"Final result of checking: {final_result}\n")
                
                # Write reduction information if check failed
                if not is_good:
                    next_check = self.check_count + 1
                    remaining_checks = self.max_checks - (next_check % self.max_checks)
                    f.write(f"\nNote: After {remaining_checks} more failed checks,\n")
                    f.write(f"thresholds will be reduced by another {self.reduce_percent}%\n")

            # Log results
            logger.info(f"Check #{self.check_count + 1}: {final_result}")
            logger.info(f"Return: {actual_return:.2f}% (Required: {current_return_threshold:.2f}%)")
            logger.info(f"Drawdown: {actual_drawdown:.2f}% (Maximum: {current_drawdown_threshold:.2f}%)")
            if total_trades == 0:
                logger.warning("No trades were executed - strategy may need adjustment")

            # Increment check count if result is not good
            if not is_good:
                self.check_count += 1

            return final_result
                
        except Exception as e:
            logger.error(f"Error in check_results: {str(e)}\n{traceback.format_exc()}")
            return "not good"


# def main():
#     """Main function to run the checker agent."""
#     try:
#         logger.info("Starting checker agent")
        
#         # Create the checker agent
#         checker_agent = CheckerAgent()
#         logger.debug("Checker agent initialized successfully")
        
#         # Specify company name
#         company_name = "ZOMATO"
#         logger.debug(f"Checking results for company: {company_name}")
        
#         # Print the current directory and base path for debugging
#         logger.debug(f"Current working directory: {os.getcwd()}")
#         logger.debug(f"Base path: {checker_agent.base_path}")
        
#         # Check if required directories exist
#         backtest_path = checker_agent.base_path / 'output' / 'backtest_results'
#         if not backtest_path.exists():
#             logger.error(f"Backtest results directory not found: {backtest_path}")
#             raise FileNotFoundError(f"Backtest results directory not found: {backtest_path}")
        
#         # Will automatically find and use the latest algo number
#         final_status = checker_agent.check_results(company_name)
#         logger.info(f"Checker agent final status: {final_status}")

#     except Exception as e:
#         logger.error(f"Main execution failed: {str(e)}\n{traceback.format_exc()}")
#         raise

# if __name__ == "__main__":
#     main()