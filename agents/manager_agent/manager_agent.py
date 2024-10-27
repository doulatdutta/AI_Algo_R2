import yaml
import os

from agents.algo_agent.algo_agent import AlgoAgent
from agents.backtesting_agent.data_download import HistoricalDataDownloader
from agents.backtesting_agent.backtesting_agent import BacktestingAgent
from agents.checker_agent.checker_agent import CheckerAgent

class ManagerAgent:
    def __init__(self, company_name):
        self.company_name = company_name

    def run(self):
        # Step 1: Ask for company name and assign job to algo agent
        algo_agent = AlgoAgent(self.company_name, 1)
        algorithm_filenames = algo_agent.generate_algorithms()

        # Step 2: Assign the generated algorithm to the backtesting agent
        data_download_agent = HistoricalDataDownloader()
        backtesting_agent = BacktestingAgent()

        # Step 3: Run the backtesting agent
        for algo_filename in algorithm_filenames:
            data_filename = f"{self.company_name}.csv"
            result_filename = algo_filename.replace(".txt", "_result.txt")

            # Download the historical data
            data_download_agent.download_data(self.company_name, data_filename)

            # Backtest the algorithm
            backtesting_agent.backtest(self.company_name, data_filename, algo_filename, result_filename)

        # Step 4: Check the backtesting results
        algo_number = 1
        while True:
            result_filename = f"{self.company_name}_algoritham-{algo_number}_result.txt"
            if CheckerAgent(self.company_name, algo_number):
                print(f"Algorithm {algo_number} return f")
                break
            else:
                print(f"Algorithm {algo_number} did not meet the requirements. Generating a new algorithm...")
                algo_agent = AlgoAgent(self.company_name, 1)
                algorithm_filenames = algo_agent.generate_algorithms()
                algo_number += 1

if __name__ == "__main__":
    manager_agent = ManagerAgent("")
    manager_agent.run()
