import logging
import logging.handlers
import os
import time
from datetime import datetime, date
from pathlib import Path
import yaml
from typing import Optional, Tuple, Dict, Any
import questionary
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
import json
import pandas as pd


# Import agents
from agents.algo_agent.algo_agent import AlgoAgent
from agents.backtesting_agent.backtesting_agent import BacktestingAgent
from agents.backtesting_agent.data_download import HistoricalDataDownloader
from agents.checker_agent.checker_agent import CheckerAgent
from agents.automated_feedback.automated_feedback import EnhancedTradingFeedbackSystem

# Create logger
logger = logging.getLogger('ManagerAgent')
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(console_handler)

class ManagerAgent:
    """
    Enhanced Manager Agent for coordinating algorithmic trading operations.
    Handles the complete lifecycle of trading strategy generation, backtesting,
    and performance evaluation.
    """
    
    def __init__(self, config_path='config/config.yaml'):
        """Initialize the Manager Agent with configuration."""
        self.base_path = Path(os.getcwd())
        self.console = Console()
        self.setup_logging()
        self.config = self.load_config(config_path)
        self.initialize_agents()
        self.setup_directories()
        self.download_tracker_file = self.base_path / 'output' / 'data_download_tracker.json'
        self.initialize_download_tracker()
        self.performance_metrics = {}

    def setup_logging(self):
        """Configure logging with enhanced formatting."""
        log_dir = self.base_path / 'output' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'manager_agent.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_config(self, config_path: str) -> dict:
        """Load and validate configuration from YAML file."""
        try:
            config_file = self.base_path / config_path
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
                
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)
                
            # Validate required configuration sections
            required_sections = ['api_provider', 'configuration', 'historical_days']
            for section in required_sections:
                if section not in config:
                    self.logger.warning(f"Missing config section: {section}")
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return self.get_default_config()

    def get_default_config(self) -> Dict:
        """Return default configuration if loading fails."""
        return {
            'api_provider': {
                'model': 'qwen2.5:1.5b',
                'provider': 'ollama'
            },
            'configuration': {
                'Return [%]': 30.00,
                'max_drawdown': 20,
                'number of checking': 10,
                'reduce %': 5
            },
            'historical_days': 30,
            'chunk_size': 7,
            'data_interval': '1m',
            'max_attempts': 6,
            'delay_between_attempts': 10,
            'delay_between_chunks': 60
        }


    def initialize_agents(self):
        """Initialize all required trading system agents."""
        try:
            # Initialize with progress tracking
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                
                task = progress.add_task("[cyan]Initializing agents...", total=4)
                
                # Historical data downloader
                self.data_downloader = HistoricalDataDownloader()
                progress.update(task, advance=1, description="[green]Data downloader initialized")
                
                # Initialize backtesting agent
                self.backtesting_agent = BacktestingAgent()
                progress.update(task, advance=1, description="[green]Backtesting agent initialized")
                
                # Initialize checker agent
                self.checker_agent = CheckerAgent()
                progress.update(task, advance=1, description="[green]Checker agent initialized")
                
                # Initialize feedback system with safe config handling
                if isinstance(self.config, dict):
                    api_provider = self.config.get('api_provider', {})
                    if isinstance(api_provider, dict):
                        model_name = api_provider.get('model', 'qwen2.5:1.5b')
                    else:
                        model_name = 'qwen2.5:1.5b'
                else:
                    model_name = 'qwen2.5:1.5b'
                
                self.feedback_system = EnhancedTradingFeedbackSystem(
                    model_name=model_name,
                    base_dir='output'
                )
                progress.update(task, advance=1, description="[green]Feedback system initialized")
                
                logger.info("All agents initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            raise

    def setup_directories(self):
        """Set up required directory structure."""
        directories = [
            'output/algo',
            'output/backtest_results',
            'output/checking_results',
            'output/feedback_logs',
            'output/logs',
            'output/performance_metrics',
            'agents/backtesting_agent/historical_data'
        ]
        
        for directory in directories:
            dir_path = self.base_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {directory}")

    def initialize_download_tracker(self):
        """Initialize or load the download tracking system."""
        try:
            if not self.download_tracker_file.exists():
                self.download_tracker = {
                    'last_update': datetime.now().isoformat(),
                    'companies': {}
                }
                self.save_download_tracker()
            else:
                self.load_download_tracker()
                
        except Exception as e:
            self.logger.error(f"Error initializing download tracker: {e}")
            self.download_tracker = {'companies': {}}

    def load_download_tracker(self):
        """Load and validate download tracker data."""
        try:
            with open(self.download_tracker_file, 'r') as f:
                self.download_tracker = json.load(f)
                
            # Validate tracker structure
            if 'companies' not in self.download_tracker:
                self.download_tracker['companies'] = {}
                
        except Exception as e:
            self.logger.error(f"Error loading download tracker: {e}")
            self.download_tracker = {'companies': {}}

    def save_download_tracker(self):
        """Save current download tracker state."""
        try:
            with open(self.download_tracker_file, 'w') as f:
                json.dump(self.download_tracker, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Error saving download tracker: {e}")

    def check_data_download_needed(self, company_name: str) -> bool:
        """
        Determine if data download is needed based on last download time
        and data freshness requirements.
        """
        today = date.today().isoformat()
        company_data = self.download_tracker.get('companies', {}).get(company_name, {})
        
        if not company_data:
            return True
            
        last_download = company_data.get('last_download_date')
        if last_download != today:
            return True
            
        # Check if data file exists and is not empty
        data_file = self.base_path / 'agents' / 'backtesting_agent' / 'historical_data' / f"{company_name}_minute.csv"
        if not data_file.exists():
            return True
            
        try:
            df = pd.read_csv(data_file)
            if df.empty:
                return True
        except Exception:
            return True
            
        return False

    def update_download_tracker(self, company_name: str):
        """Update download tracker with latest download information."""
        if 'companies' not in self.download_tracker:
            self.download_tracker['companies'] = {}
            
        self.download_tracker['companies'][company_name] = {
            'last_download_date': date.today().isoformat(),
            'last_download_timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        self.save_download_tracker()

    def save_performance_metrics(self, company_name: str, attempt: int, stats: Dict):
        """Save detailed performance metrics for analysis."""
        try:
            metrics_dir = self.base_path / 'output' / 'performance_metrics'
            metrics_file = metrics_dir / f"{company_name}_metrics.json"
            
            current_metrics = {
                'timestamp': datetime.now().isoformat(),
                'attempt': attempt,
                'metrics': {
                    'total_return': stats['Return [%]'],
                    'annual_return': stats['Return (Ann.) [%]'],
                    'sharpe_ratio': stats['Sharpe Ratio'],
                    'max_drawdown': stats['Max. Drawdown [%]'],
                    'win_rate': stats['Win Rate [%]'],
                    'total_trades': stats['# Trades'],
                    'profit_factor': stats.get('Profit Factor', 0),
                    'recovery_factor': stats.get('Recovery Factor', 0),
                    'avg_trade_duration': str(stats.get('Avg. Trade Duration', '')),
                    'exposure_time': stats.get('Exposure Time [%]', 0)
                }
            }
            
            # Load existing metrics if available
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    existing_metrics = json.load(f)
                existing_metrics.append(current_metrics)
                metrics_data = existing_metrics
            else:
                metrics_data = [current_metrics]
            
            # Save updated metrics
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Error saving performance metrics: {e}")

    def run_trading_cycle(self, company_name: str, max_attempts: int = 10) -> Tuple[bool, int]:
        """
        Run the complete trading cycle for a company with enhanced error handling
        and performance monitoring.
        """
        try:
            rprint(f"\n[bold green]Starting trading cycle for {company_name}[/bold green]")
            
            # Data download check and execution
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
            best_performance = {'sharpe_ratio': -float('inf')}
            
            while attempt <= max_attempts and not good_algorithm_found:
                try:
                    rprint(f"\n[bold blue]Attempt {attempt}/{max_attempts}[/bold blue]")
                    
                    # Algorithm generation
                    with self.console.status("[bold yellow]Generating new algorithm...") as status:
                        algo_agent = AlgoAgent(company_name)
                        strategy = algo_agent.generate_algorithms()
                        rprint("[green]✓[/green] Algorithm generated")
                    
                    # Backtesting
                    with self.console.status("[bold yellow]Running backtesting...") as status:
                        try:
                            stats = self.backtesting_agent.run_backtest(
                                company_name=company_name,
                                start_date=(datetime.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d'),
                                end_date=datetime.now().strftime('%Y-%m-%d')
                            )
                            rprint("[green]✓[/green] Backtesting completed")
                            
                            # Track best performance
                            if stats['Sharpe Ratio'] > best_performance['sharpe_ratio']:
                                best_performance = stats.copy()
                                best_performance['attempt'] = attempt
                                
                        except Exception as e:
                            rprint(f"[red]✗[/red] Backtesting failed: {str(e)}")
                            self.logger.error(f"Backtesting failed on attempt {attempt}: {str(e)}")
                            attempt += 1
                            continue
                    
                    # Results checking
                    with self.console.status("[bold yellow]Checking results...") as status:
                        result = self.checker_agent.check_results(company_name, attempt)
                        rprint(f"[{'green' if result == 'good' else 'red'}]✓[/{'green' if result == 'good' else 'red'}] Results checked")
                    
                    # Feedback processing
                    with self.console.status("[bold yellow]Processing feedback...") as status:
                        feedback_response = self.feedback_system.process_feedback(
                            company_name, 
                            attempt,
                            result == "good"
                        )
                        rprint("[green]✓[/green] Feedback processed")
                    
                    if result == "good":
                        good_algorithm_found = True
                        self.save_performance_metrics(company_name, attempt, stats)
                        rprint(f"\n[bold green]Successfully found good algorithm on attempt {attempt}![/bold green]")
                    else:
                        rprint(f"\n[yellow]Attempt {attempt} did not meet criteria. Trying again...[/yellow]")
                        attempt += 1
                        time.sleep(2)
                        
                except Exception as e:
                    self.logger.error(f"Error in attempt {attempt}: {str(e)}")
                    rprint(f"\n[red]Error in attempt {attempt}: {str(e)}[/red]")
                    attempt += 1
                    continue
            
            # Save best performance if no good algorithm found
            if not good_algorithm_found and best_performance['sharpe_ratio'] > -float('inf'):
                self.save_performance_metrics(
                    company_name,
                    best_performance['attempt'],
                    best_performance
                )
                rprint(f"\n[yellow]No algorithm met criteria. Best performance on attempt {best_performance['attempt']}[/yellow]")
            
            return good_algorithm_found, attempt
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
            rprint(f"\n[bold red]Error: {str(e)}[/bold red]")
            raise

    def get_company_input(self) -> Tuple[str, int]:
            """Get and validate company name and maximum attempts from user."""
            questions = [
                {
                    'type': 'text',
                    'name': 'company_name',
                    'message': 'Enter company name (NSE symbol):',
                    'validate': lambda text: len(text) > 0 and text.replace('.', '').isalnum(),
                    'filter': lambda text: text.upper()
                },
                {
                    'type': 'text',
                    'name': 'max_attempts',
                    'message': 'Enter maximum number of attempts (5-50):',
                    'default': '10',
                    'validate': lambda text: text.isdigit() and 5 <= int(text) <= 50
                }
            ]
            
            answers = questionary.prompt(questions)
            if not answers:
                raise KeyboardInterrupt("User cancelled input")
                
            return answers['company_name'], int(answers['max_attempts'])

    def display_data_status(self, company_name: str):
        """Display detailed data download status for a company."""
        company_data = self.download_tracker.get('companies', {}).get(company_name, {})
        
        if company_data:
            rprint("\n[cyan]Data Status:[/cyan]")
            rprint(f"Company: [bold]{company_name}[/bold]")
            rprint(f"Last download date: [yellow]{company_data['last_download_date']}[/yellow]")
            rprint(f"Last download time: [yellow]{company_data['last_download_timestamp']}[/yellow]")
            
            # Check data file
            data_file = self.base_path / 'agents' / 'backtesting_agent' / 'historical_data' / f"{company_name}_minute.csv"
            if data_file.exists():
                try:
                    df = pd.read_csv(data_file)
                    rprint(f"Data points: [green]{len(df):,}[/green]")
                    rprint(f"Date range: [green]{df['Datetime'].min()} to {df['Datetime'].max()}[/green]")
                except Exception as e:
                    rprint(f"[red]Error reading data file: {e}[/red]")
            else:
                rprint("[red]Warning: Data file not found[/red]")
        else:
            rprint(f"\n[yellow]No previous downloads found for {company_name}[/yellow]")

    def display_final_summary(self, company_name: str, success: bool, attempts: int, stats: Optional[Dict] = None):
        """Display comprehensive final summary of the trading strategy generation process."""
        rprint("\n[bold cyan]Process Summary[/bold cyan]")
        rprint("=" * 50)
        
        # Basic information
        rprint(f"Company: [bold]{company_name}[/bold]")
        rprint(f"Status: [{'green' if success else 'red'}]{('Success' if success else 'Failed')}[/{'green' if success else 'red'}]")
        rprint(f"Total Attempts: [yellow]{attempts}[/yellow]")
        
        # Performance metrics if available
        if stats:
            rprint("\n[cyan]Performance Metrics:[/cyan]")
            rprint(f"Total Return: [{'green' if stats['Return [%]'] > 0 else 'red'}]{stats['Return [%]']:.2f}%[/{'green' if stats['Return [%]'] > 0 else 'red'}]")
            rprint(f"Sharpe Ratio: [{'green' if stats['Sharpe Ratio'] > 1 else 'yellow'}]{stats['Sharpe Ratio']:.2f}[/{'green' if stats['Sharpe Ratio'] > 1 else 'yellow'}]")
            rprint(f"Win Rate: [{'green' if stats['Win Rate [%]'] > 50 else 'red'}]{stats['Win Rate [%]']:.2f}%[/{'green' if stats['Win Rate [%]'] > 50 else 'red'}]")
            rprint(f"Max Drawdown: [{'green' if stats['Max. Drawdown [%]'] < 20 else 'red'}]{stats['Max. Drawdown [%]']:.2f}%[/{'green' if stats['Max. Drawdown [%]'] < 20 else 'red'}]")
            rprint(f"Total Trades: [blue]{stats['# Trades']}[/blue]")
        
        # File locations
        if success:
            rprint("\n[green]Generated Files:[/green]")
            rprint(f"- Strategy: output/algo/{company_name}_algorithm-{attempts}.json")
            rprint(f"- Backtest Results: output/backtest_results/{company_name}_algo{attempts}")
            rprint(f"- Performance Metrics: output/performance_metrics/{company_name}_metrics.json")
            rprint(f"- Feedback Log: output/feedback_logs/{company_name}_feedback_{attempts}.txt")
        
        # Next steps
        rprint("\n[cyan]Next Steps:[/cyan]")
        if success:
            rprint("1. Review the generated strategy files")
            rprint("2. Analyze the backtest results in detail")
            rprint("3. Consider running additional validation tests")
        else:
            rprint("1. Review the feedback logs for improvement suggestions")
            rprint("2. Adjust the acceptance criteria if needed")
            rprint("3. Try running the process again with modified parameters")

def main():
    """Enhanced main function with better error handling and user interaction."""
    try:
        manager = ManagerAgent()
        rprint("\n[bold cyan]🤖 Trading Algorithm Generator[/bold cyan]")
        rprint("[yellow]Press Ctrl+C at any time to exit[/yellow]\n")
        
        # Get number of runs
        num_runs = int(questionary.text(
            "Enter number of runs (1-5):",
            default="2",
            validate=lambda text: text.isdigit() and 1 <= int(text) <= 5
        ).ask())
        
        current_run = 1
        results = []
        
        while current_run <= num_runs:
            try:
                rprint(f"\n[bold blue]Run {current_run} of {num_runs}[/bold blue]")
                rprint("=" * 50)
                
                # Get company input
                company_name, max_attempts = manager.get_company_input()
                
                # Display current data status
                manager.display_data_status(company_name)
                
                # Confirm proceed
                if not questionary.confirm("Proceed with this company?").ask():
                    continue
                
                # Run trading cycle
                success, attempts = manager.run_trading_cycle(company_name, max_attempts)
                
                # Get final stats if available
                stats = None
                if success:
                    metrics_file = manager.base_path / 'output' / 'performance_metrics' / f"{company_name}_metrics.json"
                    if metrics_file.exists():
                        with open(metrics_file, 'r') as f:
                            metrics_data = json.load(f)
                            if metrics_data:
                                stats = metrics_data[-1]['metrics']  # Get latest metrics
                
                # Display summary
                manager.display_final_summary(company_name, success, attempts, stats)
                
                # Store results
                results.append({
                    'run': current_run,
                    'company': company_name,
                    'success': success,
                    'attempts': attempts,
                    'stats': stats
                })
                
                current_run += 1
                
                if current_run <= num_runs:
                    rprint("\n[yellow]Starting next run in 5 seconds...[/yellow]")
                    time.sleep(5)
                
            except KeyboardInterrupt:
                rprint("\n[yellow]Run interrupted by user[/yellow]")
                if not questionary.confirm("Continue with next run?").ask():
                    break
                current_run += 1
            except Exception as e:
                logger.error(f"Error in run {current_run}: {e}")
                rprint(f"\n[red]Error in run {current_run}: {str(e)}[/red]")
                if not questionary.confirm("Continue with next run?").ask():
                    break
                current_run += 1
        
        # Display overall summary
        if results:
            rprint("\n[bold cyan]Overall Summary[/bold cyan]")
            rprint("=" * 50)
            successful_runs = sum(1 for r in results if r['success'])
            rprint(f"Total Runs: [blue]{len(results)}[/blue]")
            rprint(f"Successful Runs: [green]{successful_runs}[/green]")
            rprint(f"Failed Runs: [red]{len(results) - successful_runs}[/red]")
            
            if successful_runs > 0:
                # Calculate average performance metrics
                avg_metrics = {
                    'return': sum(r['stats']['total_return'] for r in results if r['success'] and r['stats']) / successful_runs,
                    'sharpe': sum(r['stats']['sharpe_ratio'] for r in results if r['success'] and r['stats']) / successful_runs,
                    'trades': sum(r['stats']['total_trades'] for r in results if r['success'] and r['stats']) / successful_runs
                }
                
                rprint("\n[cyan]Average Performance (Successful Runs):[/cyan]")
                rprint(f"Return: [{'green' if avg_metrics['return'] > 0 else 'red'}]{avg_metrics['return']:.2f}%[/{'green' if avg_metrics['return'] > 0 else 'red'}]")
                rprint(f"Sharpe Ratio: [{'green' if avg_metrics['sharpe'] > 1 else 'yellow'}]{avg_metrics['sharpe']:.2f}[/{'green' if avg_metrics['sharpe'] > 1 else 'yellow'}]")
                rprint(f"Trades per Strategy: [blue]{avg_metrics['trades']:.0f}[/blue]")
        
        rprint("\n[bold green]Trading Algorithm Generator completed![/bold green]")
        
    except KeyboardInterrupt:
        rprint("\n[yellow]Process terminated by user[/yellow]")
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        rprint(f"\n[bold red]Fatal error: {str(e)}[/bold red]")
        raise

if __name__ == "__main__":
    main()