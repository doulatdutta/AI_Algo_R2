import logging
import os
import time
from datetime import datetime, date
from pathlib import Path
import yaml
from typing import Optional, Tuple
import questionary
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
import json


# Import agents
from agents.algo_agent.algo_agent import AlgoAgent
from agents.backtesting_agent.backtesting_agent import BacktestingAgent
from agents.backtesting_agent.data_download import HistoricalDataDownloader
from agents.checker_agent.checker_agent import CheckerAgent
from agents.automated_feedback.automated_feedback import TradingFeedbackSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/logs/manager_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ManagerAgent:
    def __init__(self, config_path='config/config.yaml'):
        """Initialize the Manager Agent with configuration."""
        self.base_path = Path(os.getcwd())
        self.console = Console()
        self.config = self.load_config(config_path)
        self.initialize_agents()
        self.setup_directories()
        self.download_tracker_file = self.base_path / 'output' / 'data_download_tracker.json'
        self.initialize_download_tracker()

    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            config_file = self.base_path / config_path
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)
            logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            # Return default config if loading fails
            return {
                'ollama_model': {'model': 'mistral'},
                'configuration': {
                    'Return [%]': 30.00,
                    'max_drawdown': 20,
                    'number of checking': 10,
                    'reduce %': 5
                }
            }

    def initialize_agents(self):
        """Initialize all required agents."""
        try:
            # Initialize data downloader
            self.data_downloader = HistoricalDataDownloader()
            
            # Initialize backtesting agent
            self.backtesting_agent = BacktestingAgent()
            
            # Initialize checker agent
            self.checker_agent = CheckerAgent()
            
            # Initialize feedback system
            self.feedback_system = TradingFeedbackSystem(
                model_name=self.config['ollama_model']['model'],
                base_dir='output'
            )
            
            logger.info("All agents initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            raise

    def setup_directories(self):
        """Set up required directories for output."""
        directories = [
            'output/algo',
            'output/backtest_results',
            'output/checking_results',
            'output/feedback_logs',
            'output/logs'
        ]
        
        for directory in directories:
            os.makedirs(self.base_path / directory, exist_ok=True)
        logger.info("Directory structure set up successfully")

    def initialize_download_tracker(self):
        """Initialize or load the download tracker."""
        if not self.download_tracker_file.exists():
            self.download_tracker = {}
            self.save_download_tracker()
        else:
            self.load_download_tracker()

    def load_download_tracker(self):
        """Load the download tracker from JSON file."""
        try:
            with open(self.download_tracker_file, 'r') as f:
                self.download_tracker = json.load(f)
        except Exception as e:
            logger.error(f"Error loading download tracker: {e}")
            self.download_tracker = {}

    def save_download_tracker(self):
        """Save the download tracker to JSON file."""
        try:
            with open(self.download_tracker_file, 'w') as f:
                json.dump(self.download_tracker, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving download tracker: {e}")

    def check_data_download_needed(self, company_name: str) -> bool:
        """Check if data download is needed for today."""
        today = date.today().isoformat()
        
        if (company_name in self.download_tracker and 
            self.download_tracker[company_name].get('last_download_date') == today):
            
            data_file = self.base_path / 'agents' / 'backtesting_agent' / 'historical_data' / f"{company_name}_minute.csv"
            if data_file.exists():
                return False
                
        return True

    def update_download_tracker(self, company_name: str):
        """Update the download tracker after successful download."""
        self.download_tracker[company_name] = {
            'last_download_date': date.today().isoformat(),
            'last_download_timestamp': datetime.now().isoformat()
        }
        self.save_download_tracker()

    def save_final_results(self, company_name: str, attempt: int, stats):
        """Save final results to a summary file."""
        try:
            output_file = self.base_path / 'output' / f'{company_name}_final_result.txt'
            
            with open(output_file, 'w') as f:
                f.write(f"Final Results for {company_name}\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"Generation completed at: {datetime.now()}\n")
                f.write(f"Successful algorithm found on attempt: {attempt}\n\n")
                
                f.write("Performance Metrics:\n")
                f.write(f"Return [%]: {stats['Return [%]']:.2f}\n")
                f.write(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}\n")
                f.write(f"Win Rate [%]: {stats['Win Rate [%]']:.2f}\n")
                f.write(f"Max Drawdown [%]: {stats['Max. Drawdown [%]']:.2f}\n")
                f.write(f"Total Trades: {stats['# Trades']}\n")
                
                # Add file references
                f.write("\nRelevant Files:\n")
                f.write(f"Algorithm: output/algo/{company_name}_algorithm-{attempt}.txt\n")
                f.write(f"Backtest Results: output/backtest_results/{company_name}_algorithm-{attempt}_results.txt\n")
                f.write(f"Performance Graph: output/backtest_results/{company_name}_algorithm-{attempt}_performance.png\n")
            
            rprint(f"\n[green]Final results saved to {output_file}[/green]")
            
        except Exception as e:
            logger.error(f"Error saving final results: {e}")
            raise

    def display_final_summary(self, company_name: str, success: bool, attempts: int):
        """Display a summary of the process."""
        rprint("\n[bold cyan]Process Summary[/bold cyan]")
        rprint(f"{'='*50}")
        rprint(f"Company: [bold]{company_name}[/bold]")
        rprint(f"Status: [{'green' if success else 'red'}]{('Success' if success else 'Failed')}[/{'green' if success else 'red'}]")
        rprint(f"Attempts: [yellow]{attempts}[/yellow]")
        
        if success:
            rprint("\n[green]Results have been saved in the output directory:[/green]")
            rprint(f"- output/{company_name}_final_result.txt")
            rprint(f"- output/algo/{company_name}_algorithm-{attempts}.txt")
            rprint(f"- output/backtest_results/{company_name}_algorithm-{attempts}_results.txt")
            rprint(f"- output/backtest_results/{company_name}_algorithm-{attempts}_performance.png")

    def run_trading_cycle(self, company_name: str, max_attempts: int = 10):
        """Run the complete trading cycle for a company."""
        try:
            rprint(f"\n[bold green]Starting trading cycle for {company_name}[/bold green]")
            
            # Step 1: Check and Download historical data if needed
            if self.check_data_download_needed(company_name):
                with self.console.status("[bold green]Downloading historical data...") as status:
                    success = self.data_downloader.download_historical_data(company_name)
                    if success:
                        self.update_download_tracker(company_name)
                        rprint("[green]✓[/green] Historical data downloaded successfully")
                    else:
                        raise Exception("Failed to download historical data")
            else:
                rprint("[yellow]ℹ[/yellow] Using existing data downloaded today")
            
            attempt = 1
            good_algorithm_found = False
            
            while attempt <= max_attempts and not good_algorithm_found:
                try:
                    rprint(f"\n[bold blue]Attempt {attempt}/{max_attempts}[/bold blue]")
                    
                    # Step 2: Generate algorithm
                    with self.console.status("[bold yellow]Generating new algorithm...") as status:
                        algo_agent = AlgoAgent(company_name, num_algorithms=1)
                        algo_agent.generate_algorithms()
                        rprint("[green]✓[/green] Algorithm generated")
                    
                    # Step 3: Run backtesting with error handling
                    with self.console.status("[bold yellow]Running backtesting...") as status:
                        try:
                            stats = self.backtesting_agent.run_backtest(company_name, attempt)
                            rprint("[green]✓[/green] Backtesting completed")
                        except Exception as e:
                            rprint(f"[red]✗[/red] Backtesting failed: {str(e)}")
                            logger.error(f"Backtesting failed on attempt {attempt}: {str(e)}")
                            attempt += 1
                            continue
                    
                    # Step 4: Check results
                    with self.console.status("[bold yellow]Checking results...") as status:
                        result = self.checker_agent.check_results(company_name, attempt)
                        rprint(f"[{'green' if result == 'good' else 'red'}]✓[/{'green' if result == 'good' else 'red'}] Results checked")
                    
                    # Step 5: Process feedback
                    with self.console.status("[bold yellow]Processing feedback...") as status:
                        self.feedback_system.send_feedback(company_name, attempt)
                        rprint("[green]✓[/green] Feedback processed")
                    
                    if result == "good":
                        good_algorithm_found = True
                        rprint(f"\n[bold green]Successfully found good algorithm on attempt {attempt}![/bold green]")
                        self.save_final_results(company_name, attempt, stats)
                    else:
                        rprint(f"\n[yellow]Attempt {attempt} did not meet criteria. Trying again...[/yellow]")
                        attempt += 1
                        time.sleep(2)
                        
                except Exception as e:
                    logger.error(f"Error in attempt {attempt}: {str(e)}")
                    rprint(f"\n[red]Error in attempt {attempt}: {str(e)}[/red]")
                    attempt += 1
                    continue
            
            if not good_algorithm_found:
                rprint(f"\n[bold red]Failed to find good algorithm after {max_attempts} attempts[/bold red]")
            
            return good_algorithm_found, attempt
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            rprint(f"\n[bold red]Error: {str(e)}[/bold red]")
            raise

    def display_data_status(self, company_name: str):
        """Display the current data download status for a company."""
        if company_name in self.download_tracker:
            last_download = self.download_tracker[company_name]['last_download_date']
            last_timestamp = self.download_tracker[company_name]['last_download_timestamp']
            rprint(f"\n[cyan]Data Status for {company_name}:[/cyan]")
            rprint(f"Last download date: {last_download}")
            rprint(f"Last download time: {last_timestamp}")
        else:
            rprint(f"\n[yellow]No previous downloads found for {company_name}[/yellow]")

    def get_company_input(self) -> Tuple[str, int]:
        """Get company name input from user."""
        company_name = questionary.text(
            "Enter company name:",
            validate=lambda text: len(text) > 0 and text.isalpha()
        ).ask().upper()
        
        max_attempts = questionary.text(
            "Enter maximum number of attempts (default: 10):",
            default="10",
            validate=lambda text: text.isdigit() and 1 <= int(text) <= 100
        ).ask()
        
        return company_name, int(max_attempts)

def main():
    """Main function to run the manager agent with fixed number of runs."""
    try:
        manager = ManagerAgent()
        max_runs = 2  # Set fixed number of runs
        current_run = 1
        
        rprint(f"\n[bold cyan]Trading Algorithm Generator[/bold cyan]")
        rprint(f"[yellow]Running for {max_runs} iterations[/yellow]")
        
        while current_run <= max_runs:
            rprint(f"\n[bold blue]Run {current_run} of {max_runs}[/bold blue]")
            
            # Get company input
            company_name, max_attempts = manager.get_company_input()
            
            # Run trading cycle
            success, attempts = manager.run_trading_cycle(company_name, max_attempts)
            
            # Display summary
            manager.display_final_summary(company_name, success, attempts)
            
            if current_run < max_runs:
                rprint("\n[yellow]Starting next run...[/yellow]")
                time.sleep(2)  # Add small delay between runs
            
            current_run += 1
        
        rprint("\n[bold green]All runs completed![/bold green]")
        rprint("[bold cyan]Thank you for using the Trading Algorithm Generator![/bold cyan]")
                
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        rprint(f"\n[bold red]Error: {str(e)}[/bold red]")
        raise

if __name__ == "__main__":
    main()