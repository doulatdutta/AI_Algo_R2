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
            
            # This function will generate algorithms for the given company using the specified model.
    def generate_algorithms(self):
        print(f"Generating {self.num_algorithms} indicator-based algorithms for {self.company_name} using {self.model}...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i in range(self.num_algorithms):
            try:
                algorithm = self.generate_algorithm()
                output_filename = f"{self.company_name}_algorithm-{i + 1}_{timestamp}.txt"
                self.save_algorithm(self.format_algorithm(algorithm), output_filename)
                print(f"Algorithm {i + 1}: saved to {output_filename}")
            except Exception as e:
                print(f"Error generating algorithm {i + 1}: {e}")

                # return algorithm
    def generate_algorithm(self):
        system_prompt = """You are a trading algorithm expert. Generate algorithmic trading rules using technical indicators."""

        user_prompt = """Generate clear buy and sell conditions based on the following indicators:
            - Moving Averages: SMA, EMA, WMA, VWAP
            - Momentum Indicators: RSI, MACD, Stochastic Oscillator, Williams %R
            - Volatility Indicators: Bollinger Bands, ATR, Keltner Channels
            - Volume Indicators: OBV, Volume Price Trend, MFI
            - Trend Indicators: ADX, CCI, Ichimoku Cloud
       
        Must include exactly 3 conditions for each buy and sell signal.
        Must use specific values for indicators (e.g., periods, thresholds).
        Must be technical and precise.
        must Ensure conditions are precise and can be easily implemented with Python libraries.
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
        """Format the algorithm output to match the required structure"""
        # Extract buy and sell conditions
        buy_conditions = []
        sell_conditions = []
        
        # Parse the raw algorithm text
        lines = raw_algorithm.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if 'buy' in line.lower():
                current_section = 'buy'
                continue
            elif 'sell' in line.lower():
                current_section = 'sell'
                continue
                
            if line and current_section == 'buy':
                buy_conditions.append(line.strip(',.'))
            elif line and current_section == 'sell':
                sell_conditions.append(line.strip(',.'))
        
        # Format the output
        formatted_output = "Buy Conditions:\n"
        for condition in buy_conditions[:3]:  # Ensure exactly 3 conditions
            formatted_output += f"{condition}.\n"
            
        formatted_output += "\nSell Conditions:\n"
        for condition in sell_conditions[:3]:  # Ensure exactly 3 conditions
            formatted_output += f"{condition}.\n"
            
        return formatted_output

    def save_algorithm(self, algorithm_conditions, output_filename):
        output_path = os.path.join('output', 'logs')
        try:
            os.makedirs(output_path, exist_ok=True)
            full_path = os.path.join(output_path, output_filename)
            
            with open(full_path, 'w') as f:
                f.write(algorithm_conditions)
                
            print(f"Algorithm saved to {full_path}")
        except Exception as e:
            print(f"Error saving algorithm: {e}")

if __name__ == "__main__":
    algo_agent = AlgoAgent("tatasteel", 1)
    algo_agent.generate_algorithms()