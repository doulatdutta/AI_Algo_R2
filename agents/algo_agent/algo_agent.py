import json
import re
import yaml
import ollama
import os
from datetime import datetime

class AlgoAgent:
    def __init__(self, company_name, num_algorithms):
        self.company_name = company_name
        self.num_algorithms = num_algorithms
        self.model = self.load_model_name()
        
    def load_model_name(self):
        try:
            with open("config/config.yaml", "r") as file:
                config = yaml.safe_load(file)
            return config['ollama_model']['model']
        except Exception as e:
            print(f"Error loading config: {e}")
            return "mistral"  # Default fallback model
            
    def generate_algorithms(self):
        print(f"Generating {self.num_algorithms} indicator-based algorithms for {self.company_name} using {self.model}...")
        
        for i in range(self.num_algorithms):
            algo_num = 1
            while True:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{self.company_name}_algorithm-{algo_num}"
                if not os.path.exists(os.path.join('output', 'algo', f"{output_filename}.txt")):
                    break
                algo_num += 1
            
            try:
                algorithm = self.generate_algorithm()
                self.save_algorithm(algorithm, output_filename)
                print(f"Algorithm {algo_num}: saved as {output_filename}.txt and {output_filename}.json")
            except Exception as e:
                print(f"Error generating algorithm {algo_num}: {e}")

    def generate_algorithm(self):
        system_prompt = """You are a intradey 1-minute trading algorithm expert. Generate algorithmic trading rules using technical indicators that can be implemented with the Python backtesting.py library."""

        user_prompt = """Generate clear buy and sell conditions based on technical indicators. Format each condition as:
        "INDICATOR_NAME OPERATOR VALUE" (e.g., "RSI < 30" or "SMA20 > SMA50")
        
        Use these indicators:
         - Moving Averages: SMA, EMA, WMA, VWAP
         - Momentum Indicators: RSI, MACD, Stochastic Oscillator, Williams %R
         - Volatility Indicators: Bollinger Bands, ATR, Keltner Channels
         - Volume Indicators: OBV, Volume Price Trend, MFI
         - Trend Indicators: ADX, CCI, Ichimoku Cloud
        
        Requirements:
        1. Provide exactly 5 buy conditions and 5 sell conditions, only raw conditions, no explanation, no serial number needed.
        2. Use specific values for indicators (e.g., "RSI(14) < 30")
        3. Each condition should be a single line. only write condition, no explanation needed.
        4. Use operators: >, <, >=, <=, ==
        5. All conditions must be implementable using backtesting.py library
        6. Include period values for indicators (e.g., SMA(20))
        
        Format example:
        Buy Conditions:
        RSI(14) < 30
        SMA(20) > SMA(50)
        
        Sell Conditions:
        RSI(14) > 70
        SMA(20) < SMA(50)
        """

        try:
            response = ollama.chat(model=self.model, messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ])
            
            if 'message' not in response or 'content' not in response['message']:
                raise ValueError("Invalid response format from Ollama")
                
            return response['message']['content']
        except Exception as e:
            raise Exception(f"Error generating algorithm: {e}")

    def format_algorithm(self, raw_algorithm):
        """Format the algorithm output into structured conditions"""
        buy_conditions = []
        sell_conditions = []
        
        # Split into buy and sell sections
        sections = raw_algorithm.lower().split('sell conditions:')
        if len(sections) == 2:
            buy_section = sections[0].split('buy conditions:')[-1].strip()
            sell_section = sections[1].strip()
            
            # Extract conditions
            buy_conditions = [line.strip() for line in buy_section.split('\n') if line.strip()]
            sell_conditions = [line.strip() for line in sell_section.split('\n') if line.strip()]
        
        # Format for text output
        txt_output = "Buy Conditions:\n"
        for condition in buy_conditions[:5]:
            txt_output += f"{condition}\n"
            
        txt_output += "\nSell Conditions:\n"
        for condition in sell_conditions[:5]:
            txt_output += f"{condition}\n"
            
        # Format for JSON output
        def parse_condition(condition):
            # Regular expression to match indicator patterns
            pattern = r'(\w+(?:\(\d+\))?)\s*([<>=]+)\s*([\w\d.()]+)'
            match = re.match(pattern, condition)
            if match:
                indicator, operator, value = match.groups()
                # Extract period if present
                period_match = re.search(r'\((\d+)\)', indicator)
                period = int(period_match.group(1)) if period_match else None
                
                return {
                    # "indicator": indicator.split('(')[0],  # Base indicator name
                    # "period": period,
                    # "operator": operator,
                    # "value": value,
                    "raw_condition": condition  # Store original condition
                }
            return {"raw_condition": condition}  # Fallback for complex conditions
        
        json_output = {
            "buy_conditions": [parse_condition(cond) for cond in buy_conditions[:5]],
            "sell_conditions": [parse_condition(cond) for cond in sell_conditions[:5]],
            "metadata": {
                "company": self.company_name,
                "generated_at": datetime.now().isoformat(),
                "model": self.model
            }
        }
        
        return txt_output, json_output

    def save_algorithm(self, algorithm_text, output_filename):
        output_path = os.path.join('output', 'algo')
        os.makedirs(output_path, exist_ok=True)
        
        # Format and save both versions
        txt_content, json_content = self.format_algorithm(algorithm_text)
        
        # Save TXT version
        txt_path = os.path.join(output_path, f"{output_filename}.txt")
        with open(txt_path, 'w') as f:
            f.write(txt_content)
        
        # Save JSON version
        json_path = os.path.join(output_path, f"{output_filename}.json")
        with open(json_path, 'w') as f:
            json.dump(json_content, f, indent=4)



if __name__ == "__main__":
    algo_agent = AlgoAgent("tatasteel", 1)
    algo_agent.generate_algorithms()
