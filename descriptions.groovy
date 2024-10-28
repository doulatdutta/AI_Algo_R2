#description of full project




Here is a basic outline of how you might structure your code. Please note that this is a simplified version and may need to be adjusted based on your specific requirements and constraints.

Step-1:- 'manager_agent' will ask for company name, and then asign job to algo agent. 

Step-2:-'algo agent' will get the company name from manager agent and generate one algoritham (buy and sell condition) with help of ollama, and save it in output/algo folder as {company_name_algoritham-1}.txt
only use technical indicator which are sutable with ta-lib and pandas-ta library.

         - Moving Averages: SMA, EMA, WMA, VWAP
         - Momentum Indicators: RSI, MACD, Stochastic Oscillator, Williams %R
         - Volatility Indicators: Bollinger Bands, ATR, Keltner Channels
         - Volume Indicators: OBV, Volume Price Trend, MFI
         - Trend Indicators: ADX, CCI, Ichimoku Cloud

        formatting of {company_name_algoritham-1}.txt will be as follows:
        Buy Conditions:
        Short-term SMA > Long-term SMA (bullish crossover).
        RSI < 40 (oversold condition).
        MACD signal > MACD histogram (bullish momentum).
        Sell Conditions:
        Short-term SMA < Long-term SMA (bearish crossover).
        RSI > 60 (overbought condition).
        MACD signal < MACD histogram (bearish momentum).


Step-3:-Then manager agent will ask 'data_download' to download pervious 30 days data everyminuite using ytfinance and save it in agents/backetsting_agent/historical_data folder as {company_name}.csv
        Also, data_download_agent will check everyday if it has already downloaded the 30 days data for that day and for that companyor not, if it has already downloaded then it will skip that day and continue with next day.

        Format will be as follows:-
        Date,Open,High,Low,Close,Adj Close,Volume

Step-4:-After'data_download' completed its work 'manager_agent' will asign the generated algoritham (buy and sell condition) to 'backtesting_agent'.

Step-5:-The 'Backtesting_agent' will backtest the algoritham(inside output\algo\company_name_algorithm-1_date_time.txt) with the historical data (inside agents\backtesting_agent\historical_data\ folder as .csv file) downloaded by 'data_download' 
        and save the result in output\backtest_results folder as (company_name_algoritham-1_result).txt and also save the graph of backtested data in output\backtest_results folder as (company_name_algoritham-1_performance).png. 
        'Backtesting_agent' will backtest realtime trading as per the generated algoritham, it will calculate total number of trade for eatch day, calculate total profit or loss (in % for 1000000 cash and 0.02% brockarage charges) for each day and save it.
        It will calculate total trade from two type of trade "buy_trade" or "sell trade"
        it will consider one "buy_trade" as buy stock as per the buy condition mentioned in (company_name_algoritham-1).txt and sell stock as per the sell condition mentioned in (company_name_algoritham-1).txt 
        and it will consider one "sell_trade" as sell stock as per the sell condition mentioned in (company_name_algoritham-1).txt and buy condition mentioned in (company_name_algoritham-1).txt. 
        after completing any one trade, it can trade another. do not mix one with anoter.        
        Number of stock to buy and sell will be written in config.yaml file.
        type of graph will be line graph with two line one for profit or loss  and one for date.

        Use backtesting.py, pandas-ta, tulip, ta-lib, pandas libreary for backtesting and graph plotting.

    Formatting of (company_name_algoritham-1_result).txt will be as follows:

        Algorithm -
        Buy Conditions:
        - **; - rsi(14) < 30; - sma(20) > sma(50); - macd(12, 26) > macd(12, 26)_lag1; - stochastic oscillator(14, 3, 3) < 20

        Sell Conditions:
        - **; - rsi(14) > 70; - sma(20) < sma(50); - macd(12, 26) < macd(12, 26)_lag1; - williams %r(9) > -80

        Daily Results:
            2024-10-25 Profit: 0.00% ; total trade: 0 ; buy_trade - 0 ; sell_trade - 0
            2024-10-24 Profit: 0.00% ; total trade: 0 ; buy_trade - 0 ; sell_trade - 0
            2024-10-23 Profit: 0.00% ; total trade: 0 ; buy_trade - 0 ; sell_trade - 0
            2024-10-22 Profit: 0.00% ; total trade: 0 ; buy_trade - 0 ; sell_trade - 0
            2024-10-21 Profit: 0.00% ; total trade: 0 ; buy_trade - 0 ; sell_trade - 0
            2024-10-18 Profit: 0.00% ; total trade: 0 ; buy_trade - 0 ; sell_trade - 0
            2024-10-17 Profit: 0.00% ; total trade: 0 ; buy_trade - 0 ; sell_trade - 0
            2024-10-16 Profit: 0.00% ; total trade: 0 ; buy_trade - 0 ; sell_trade - 0
            2024-10-15 Profit: 0.00% ; total trade: 0 ; buy_trade - 0 ; sell_trade - 0
            2024-10-14 Profit: 0.00% ; total trade: 0 ; buy_trade - 0 ; sell_trade - 0
            2024-10-11 Profit: 0.00% ; total trade: 0 ; buy_trade - 0 ; sell_trade - 0
            2024-10-10 Profit: 0.00% ; total trade: 0 ; buy_trade - 0 ; sell_trade - 0
            2024-10-09 Profit: 0.00% ; total trade: 0 ; buy_trade - 0 ; sell_trade - 0
            2024-10-08 Profit: 0.00% ; total trade: 0 ; buy_trade - 0 ; sell_trade - 0
            2024-10-07 Profit: 0.00% ; total trade: 0 ; buy_trade - 0 ; sell_trade - 0
            2024-10-04 Profit: 0.00% ; total trade: 0 ; buy_trade - 0 ; sell_trade - 0
            2024-10-03 Profit: 0.00% ; total trade: 0 ; buy_trade - 0 ; sell_trade - 0
            2024-10-01 Profit: 0.00% ; total trade: 0 ; buy_trade - 0 ; sell_trade - 0
            2024-09-30 Profit: 0.00% ; total trade: 0 ; buy_trade - 0 ; sell_trade - 0
            2024-09-27 Profit: 0.00% ; total trade: 0 ; buy_trade - 0 ; sell_trade - 0

        Overall Statistics:
        Start: 2024-09-27 09:15:00+05:30
        End: 2024-10-25 15:29:00+05:30
        Duration: 28 days 06:14:00
        Exposure Time [%]: 0.00
        Equity Final [$]: 100000.00
        Equity Peak [$]: 100000.00
        Return [%]: 40.00
        Buy & Hold Return [%]: -12.88
        Return (Ann.) [%]: 40.00
        Volatility (Ann.) [%]: 0.00
        Sharpe Ratio: nan
        Sortino Ratio: nan
        Calmar Ratio: nan
        Max. Drawdown [%]: -0.00
        Avg. Drawdown [%]: nan
        Max. Drawdown Duration: nan
        Avg. Drawdown Duration: nan
        # Trades: 0
        Win Rate [%]: nan
        Best Trade [%]: nan
        Worst Trade [%]: nan
        Avg. Trade [%]: nan
        Max. Trade Duration: nan
        Avg. Trade Duration: nan
        Profit Factor: nan
        Expectancy [%]: nan
        Squeeze Ratio: nan



Step-6:- then 'manager_agent' will ask 'checker_agent' to check the result saved in output/backtest_results folder as {company_name_algoritham-1_result}.txt 
    and if the result is meeting the % mentioned in config.yaml then it will return to manager agent as 'good' else it will return as 'not good'.
    if the number of checking is equal to the number mentioned in config.yaml, it will reduce the % mentioned by 5% everytime. (As like 1st time it will check with 70% if it is not good then it will check with 65% and so on.)
    Final data will be saved in log file as (companyname_checking-1.txt) 
    
    formatting of (companyname_checking-1.txt) will be like:- 

        Algorithm Check #1
        Required Return: 30.0%
        Max Drawdown: 20.0%
        Return [%] Checking: Good (40.00% >= 30.0%)
        Final result of checking: good

    
Step-7:- if 'checker_agent' return 'good' then 'manager_agent' will stop the process and publish the {company_name_algoritham-1_result}.txt as final result in output folder as {company_name_final_result}.txt 
    

Step-8:- if 'checker_agent' return 'not good', 'manager_agent' will ask 'automated_feedback' to provide feedback to ollama model to finetune the model to generate better algoritham.
        and the feedback will be saved in output/feedback folder as {company_name_feedback-1}.txt

    formatting of ollama feedback will be like:-

Ollama Trainer Feedback #1
        	Algorithm: 
                Buy Conditions:
                **
                rsi(14) < 30
                sma(20) > sma(50)
                macd(12, 26) > macd(12, 26)_lag1
                stochastic oscillator(14, 3, 3) < 20

                Sell Conditions:
                **
                rsi(14) > 70
                sma(20) < sma(50)
                macd(12, 26) < macd(12, 26)_lag1
                williams %r(9) > -80

            Final result of checking: good
            Return [%]: 40.00
            Max. Drawdown [%]: -0.00
            Feedback: 
            Areas for Improvement:
                    1. Risk Management: The strategy needs better risk control mechanisms
                    2. Entry/Exit Timing: Review and optimize the timing of trades
                    3. Technical Indicators: Reconsider the combination and parameters of indicators

                    For the next strategy suggestion, please focus on:
                    1. More robust entry/exit conditions
                    2. Better risk-reward ratios
                    3. Improved drawdown protection
                    4. More effective trend confirmation signals

                    
Step-9:- Then 'manager_agent' will ask 'algo_agent' to generate new algoritham and save it in output/logs folder as {company_name_algoritham-2}.txt and so on until the result is 'good' or the number of algoritham generated is equal to the number mentioned in config.yaml.


Folder strucure will be :- 

AI_Algo_R2
│
├── main.py                  # Entry point for running the Manager Agent. This script will only run the manager_agent. 
│
├── README.md                # Documentation file providing an overview of the project, installation instructions, and usage guide
│
├── requirements.txt         # List of dependencies required for the project
│
├── /agents                  # Directory for various agent implementations
│   ├── /algo_agent          # Algo Agent directory
│   │   ├── __init__.py      # Init file to treat this folder as a package
│   │   ├── algo_agent.py    # Logic for generating new trading algorithms
│   │
│   ├── /backtesting_agent          # Backtesting Agent directory
│   │   ├── __init__.py             # Init file for this agent
│   │   ├── backtesting_agent.py    # Logic for backtesting algorithms
│   │   ├── /historical_data        # Folder for storing historical data files
│   │   └── data_download.py        # Script for downloading historical data
│   │
│   ├── /checker_agent          # Checker Agent directory
│   │   ├── __init__.py         # Init file for this agent
│   │   └── checker_agent.py    # Logic for evaluating algorithm profitability
│   │
│   ├── /automated_feedback        # Checker Agent directory
│   │   ├── __init__.py        # Init file for this agent
│   │   └── automated_feedback.py  # Logic for finetuning ollama model
│   │
│   ├── /manager_agent          # Manager Agent directory
│   │   ├── __init__.py         # Init file for this agent
│   │   └── manager_agent.py    # Coordinates the overall process between all agents
│   │
│   └── /output                                 # Output directory for results and logs
│       ├── {company_name_final_result}.txt     # Final output file containing the generated algorithm and profit/loss details
│       │
│       ├── /algo                                   # Folder for save all backtest result
│       │   ├── {company_name_algoritham-1}.txt     # 1st generated algorithm file
│       │   └── {company_name_algoritham-1}.json    # 1st generated algorithm json file
│       │
│       ├── /backtest_result                                # Folder for save all backtest result
│       │   ├── {company_name_algoritham-1_result}.txt      # 1st generated algorithm result file
│       │   └── {company_name_algoritham-1_graph}.png       # 1st generated algorithm graph file
│       │
│       ├── /checking_result                            # Folder for save all backtest result
│       │   └── {company_name_checking-1}.png           # 1st checking result file.
│       │ 
│       └── logs                                        # Directory for logs generated during the execution of agents
│
│── /input                          # Input directory for data files (if any)
│   └── Algoritham_book.pdf         # Input file containing the book of algoritham (if any). This will be used to finetune the ollama model
│
│
└── /config                  # Configuration files
    └── config.yaml          # Configuration file for agents, Ollama model settings, etc.



**technical indicator**

* Simple Moving Average 'SMA'
* Simple Moving Median 'SMM'
* Smoothed Simple Moving Average 'SSMA'
* Exponential Moving Average 'EMA'
* Double Exponential Moving Average 'DEMA'
* Triple Exponential Moving Average 'TEMA'
* Triangular Moving Average 'TRIMA'
* Triple Exponential Moving Average Oscillator 'TRIX'
* Volume Adjusted Moving Average 'VAMA'
* Kaufman Efficiency Indicator 'ER'
* Kaufmans Adaptive Moving Average 'KAMA'
* Zero Lag Exponential Moving Average 'ZLEMA'
* Weighted Moving Average 'WMA'
* Hull Moving Average 'HMA'
* Elastic Volume Moving Average 'EVWMA'
* Volume Weighted Average Price 'VWAP'
* Smoothed Moving Average 'SMMA'
* Fractal Adaptive Moving Average 'FRAMA'
* Moving Average Convergence Divergence 'MACD'
* Percentage Price Oscillator 'PPO'
* Volume-Weighted MACD 'VW_MACD'
* Elastic-Volume weighted MACD 'EV_MACD'
* Market Momentum 'MOM'
* Rate-of-Change 'ROC'
* Relative Strenght Index 'RSI'
* Inverse Fisher Transform RSI 'IFT_RSI'
* True Range 'TR'
* Average True Range 'ATR'
* Stop-and-Reverse 'SAR'
* Bollinger Bands 'BBANDS'
* Bollinger Bands Width 'BBWIDTH'
* Momentum Breakout Bands 'MOBO'
* Percent B 'PERCENT_B'
* Keltner Channels 'KC'
* Donchian Channel 'DO'
* Directional Movement Indicator 'DMI'
* Average Directional Index 'ADX'
* Pivot Points 'PIVOT'
* Fibonacci Pivot Points 'PIVOT_FIB'
* Stochastic Oscillator %K 'STOCH'
* Stochastic oscillator %D 'STOCHD'
* Stochastic RSI 'STOCHRSI'
* Williams %R 'WILLIAMS'
* Ultimate Oscillator 'UO'
* Awesome Oscillator 'AO'
* Mass Index 'MI'
* Vortex Indicator 'VORTEX'
* Know Sure Thing 'KST'
* True Strength Index 'TSI'
* Typical Price 'TP'
* Accumulation-Distribution Line 'ADL'
* Chaikin Oscillator 'CHAIKIN'
* Money Flow Index 'MFI'
* On Balance Volume 'OBV'
* Weighter OBV 'WOBV'
* Volume Zone Oscillator 'VZO'
* Price Zone Oscillator 'PZO'
* Elders Force Index 'EFI'
* Cummulative Force Index 'CFI'
* Bull power and Bear Power 'EBBP'
* Ease of Movement 'EMV'
* Commodity Channel Index 'CCI'
* Coppock Curve 'COPP'
* Buy and Sell Pressure 'BASP'
* Normalized BASP 'BASPN'
* Chande Momentum Oscillator 'CMO'
* Chandelier Exit 'CHANDELIER'
* Qstick 'QSTICK'
* Twiggs Money Index 'TMF'
* Wave Trend Oscillator 'WTO'
* Fisher Transform 'FISH'
* Ichimoku Cloud 'ICHIMOKU'
* Adaptive Price Zone 'APZ'
* Squeeze Momentum Indicator 'SQZMI'
* Volume Price Trend 'VPT'
* Finite Volume Element 'FVE'
* Volume Flow Indicator 'VFI'
* Moving Standard deviation 'MSD'
* Schaff Trend Cycle 'STC'
* Mark Whistlers WAVE PM 'WAVEPM'

Ollama Trainer Feedback #1
        	Algorithm: 
                Buy Conditions:
                **
                rsi(14) < 30
                sma(20) > sma(50)
                macd(12, 26) > macd(12, 26)_lag1
                stochastic oscillator(14, 3, 3) < 20

                Sell Conditions:
                **
                rsi(14) > 70
                sma(20) < sma(50)
                macd(12, 26) < macd(12, 26)_lag1
                williams %r(9) > -80

            Final result of checking: good
            Return [%]: 40.00
            Max. Drawdown [%]: -0.00
            Feedback: 
            Areas for Improvement:
                    1. Risk Management: The strategy needs better risk control mechanisms
                    2. Entry/Exit Timing: Review and optimize the timing of trades
                    3. Technical Indicators: Reconsider the combination and parameters of indicators

                    For the next strategy suggestion, please focus on:
                    1. More robust entry/exit conditions
                    2. Better risk-reward ratios
                    3. Improved drawdown protection
                    4. More effective trend confirmation signals


it will take leatest algo_num (aslike if there is two file companyname_algoritham-1.txt and  companyname_algoritham-2.txt it will take data from companyname_algoritham-2.txt)
"Algorithm" from output/algo/{company_name}_algoritham-{algo_num}.txt 
"Final result of checking" from output/checking_result/{company_name}_checking-{algo_num}.txt
"Return [%]" from output/backtest_result/{company_name}_algoritham-{algo_num}_result.txt
"Max. Drawdown [%]:" from output/backtest_result/{company_name}_algoritham-{algo_num}_result.txt


    