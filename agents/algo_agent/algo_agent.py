import glob
import json
import os
from datetime import datetime
import re
import ollama
import yaml
import openai
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from groq import Groq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlgoAgent:
    def __init__(self, company_name: str):
        """Initialize AlgoAgent with company name."""
        self.company_name = company_name
        self.base_path = Path(os.getcwd())
        self.config = self.load_config()
        self.setup_api_client()

    def load_config(self) -> dict:
        """Load configuration from config file."""
        try:
            with open("config/config.yaml", "r") as file:
                config = yaml.safe_load(file)
                
            # Get API provider
            self.api_provider = config.get('api_provider', 'ollama').lower()
            
            # Set up API configurations
            if self.api_provider == 'openai':
                openai.api_key = config.get('openai', {}).get('api_key')
                self.model = config.get('openai', {}).get('model', 'gpt-4-turbo-preview')
            elif self.api_provider == 'groq':
                self.groq_api_key = config.get('groq', {}).get('api_key')
                self.model = config.get('groq', {}).get('model', 'llama-3.1-70b-versatile')
            else:  # ollama
                self.model = config.get('ollama', {}).get('model', 'qwen2.5:1.5b')
            
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def setup_api_client(self) -> None:
        """Setup the appropriate API client based on configuration."""
        try:
            if self.api_provider == 'openai':
                if not openai.api_key:
                    raise ValueError("OpenAI API key not found in configuration")
                logger.info("Using OpenAI API for generation")
                
            elif self.api_provider == 'groq':
                if not self.groq_api_key:
                    raise ValueError("Groq API key not found in configuration")
                self.groq_client = Groq(api_key=self.groq_api_key)
                logger.info("Using Groq API for generation")
                
            else:
                logger.info("Using Ollama for generation")
                
        except Exception as e:
            logger.error(f"Error setting up API client: {e}")
            raise

    def generate_chat_completion(self, system_prompt: str, user_prompt: str) -> str:
        """Generate completion using the configured API provider."""
        try:
            if self.api_provider == 'openai':
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                return response.choices[0].message.content
                
            elif self.api_provider == 'groq':
                completion = self.groq_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                return completion.choices[0].message.content
                
            else:  # ollama
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ]
                )
                return response['message']['content']
                
        except Exception as e:
            logger.error(f"Error generating completion with {self.api_provider}: {e}")
            raise

    def generate_strategy_json(self) -> Dict:
        """Generate a trading strategy in JSON format."""
        system_prompt = """You are a professional algorithmic trader. Your task is to generate a trading strategy JSON. 
        You must ONLY output valid JSON - no explanations or additional text. The output must be parseable by json.loads()."""

        user_prompt = f"""Create a trading strategy JSON for {self.company_name} with this exact structure:
        {{
            "initial_capital": 100000,
            "commission": 0.002,
            "trading_hours": {{
                "start": "09:15",
                "end": "15:20"
            }},
            "moving_averages": [
                {{
                    "type": "ALMA",
                    "length": 2,
                    "offset": 0.85,
                    "sigma": 5.0,
                    "source": "Close",
                    "name": "close_ma"
                }}
            ],
            "entry_conditions": [
                {{
                    "indicator1": "close_ma",
                    "indicator2": "open_ma",
                    "condition": "crossover",
                    "action": "buy",
                    "size": 0.99
                }}
            ],
            "exit_conditions": [
                {{
                    "indicator1": "close_ma",
                    "indicator2": "open_ma",
                    "condition": "crossunder",
                    "action": "exit_long"
                }}
            ]
        }}

        Respond ONLY with a valid JSON object following this structure. Add more indicators and conditions as needed, but maintain the exact format.

        """
        try:
            response = self.generate_chat_completion(system_prompt, user_prompt)
            
            # Clean and validate the response
            response = self.clean_json_response(response)
            
            # Try to parse the JSON
            try:
                strategy_json = json.loads(response)
                # Validate required fields
                required_fields = ['initial_capital', 'commission', 'trading_hours', 
                                'moving_averages', 'entry_conditions', 'exit_conditions']
                for field in required_fields:
                    if field not in strategy_json:
                        raise ValueError(f"Missing required field: {field}")
                return strategy_json
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response: {response}")
                logger.error(f"JSON parsing error: {str(e)}")
                # Return a default strategy if parsing fails
                return self.get_default_strategy()
                
        except Exception as e:
            logger.error(f"Error generating strategy JSON: {e}")
            return self.get_default_strategy()

    def clean_json_response(self, response: str) -> str:
        """Clean the AI response to ensure it's valid JSON."""
        try:
            # Remove any markdown code block indicators
            response = response.replace('```json', '').replace('```', '')
            
            # Find the first '{' and last '}' to extract just the JSON object
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start >= 0 and end > start:
                response = response[start:end]
            
            return response.strip()
        except Exception as e:
            logger.error(f"Error cleaning JSON response: {e}")
            return "{}"


    def get_indicator_settings(self, indicator_type: str, settings: dict) -> dict:
        """Get indicator settings with default values if missing."""
        default_settings = {
            'SMA': {
                'period': 20
            },
            'EMA': {
                'period': 20
            },
            'RSI': {
                'period': 14
            },
            'MACD': {
                'fast_length': 12,
                'slow_length': 26,
                'signal_length': 9
            },
            'BB': {
                'period': 20,
                'multiplier': 2
            }
        }
        
        if indicator_type not in default_settings:
            raise ValueError(f"Unsupported indicator type: {indicator_type}")
            
        # Merge provided settings with defaults
        return {**default_settings[indicator_type], **(settings or {})}

    def get_default_strategy(self) -> Dict:
        """Return a default strategy if JSON generation fails."""
        return {
            "initial_capital": 100000,
            "commission": 0.002,
            "trading_hours": {
                "start": "09:15",
                "end": "15:20"
            },
            "moving_averages": [
                {
                    "type": "ALMA",
                    "length": 2,
                    "offset": 0.85,
                    "sigma": 5.0,
                    "source": "Close",
                    "name": "close_ma"
                },
                {
                    "type": "ALMA",
                    "length": 2,
                    "offset": 0.85,
                    "sigma": 5.0,
                    "source": "Open",
                    "name": "open_ma"
                }
            ],
            "entry_conditions": [
                {
                    "indicator1": "close_ma",
                    "indicator2": "open_ma",
                    "condition": "crossover",
                    "action": "buy",
                    "size": 0.99
                },
                {
                    "indicator1": "close_ma",
                    "indicator2": "open_ma",
                    "condition": "crossunder",
                    "action": "sell",
                    "size": 0.99
                }
            ],
            "exit_conditions": [
                {
                    "indicator1": "close_ma",
                    "indicator2": "open_ma",
                    "condition": "crossunder",
                    "action": "exit_long"
                },
                {
                    "indicator1": "close_ma",
                    "indicator2": "open_ma",
                    "condition": "crossover",
                    "action": "exit_short"
                }
            ]
        }

    def json_to_pine(self, strategy_json: Dict) -> str:
        """Convert strategy JSON to Pine Script."""
        try:
            # Start with strategy declaration
            pine_script = f"""//@version=5
    strategy("Strategy", 
        overlay=true, 
        initial_capital={strategy_json['initial_capital']}, 
        commission_type=strategy.commission.percent,
        commission_value={strategy_json['commission']},
        default_qty_type=strategy.percent_of_equity)

    // Trading Hours
    var start_time = timestamp("{strategy_json['trading_hours']['start']}")
    var end_time = timestamp("{strategy_json['trading_hours']['end']}")
    is_trading_time = time >= start_time and time <= end_time

    // Moving Averages"""

            # Add moving averages
            for ma in strategy_json['moving_averages']:
                pine_script += f"""
    {ma['name']} = ta.alma({ma['source'].lower()}, {ma['length']}, {ma['offset']}, {ma['sigma']})"""

            # Add entry conditions
            pine_script += "\n\n// Entry Conditions"
            for entry in strategy_json['entry_conditions']:
                if entry['condition'] == 'crossover':
                    pine_script += f"""
    if (ta.crossover({entry['indicator1']}, {entry['indicator2']}) and is_trading_time)
        strategy.entry("{entry['action']}", strategy.{entry['action']}, qty=strategy.equity * {entry['size']})"""
                elif entry['condition'] == 'crossunder':
                    pine_script += f"""
    if (ta.crossunder({entry['indicator1']}, {entry['indicator2']}) and is_trading_time)
        strategy.entry("{entry['action']}", strategy.{entry['action']}, qty=strategy.equity * {entry['size']})"""

            # Add exit conditions
            pine_script += "\n\n// Exit Conditions"
            for exit in strategy_json['exit_conditions']:
                if exit['condition'] == 'crossover':
                    pine_script += f"""
    if (ta.crossover({exit['indicator1']}, {exit['indicator2']}) and is_trading_time)
        strategy.{exit['action']}()"""
                elif exit['condition'] == 'crossunder':
                    pine_script += f"""
    if (ta.crossunder({exit['indicator1']}, {exit['indicator2']}) and is_trading_time)
        strategy.{exit['action']}()"""

            return pine_script

        except Exception as e:
            logger.error(f"Error converting JSON to Pine Script: {e}")
            raise



    def generate_algorithms(self) -> None:
        """Generate trading algorithm in both JSON and Pine Script formats."""
        logger.info(f"Generating trading algorithm for {self.company_name}")
        
        # Find the next available algorithm number
        algo_num = self.get_next_algorithm_number()
        output_filename = f"{self.company_name}_algorithm-{algo_num}"
        
        try:
            # Generate strategy JSON
            strategy_json = self.generate_strategy_json()
            
            # Convert JSON to Pine Script
            pine_script = self.json_to_pine(strategy_json)
            
            # Save both formats
            self.save_algorithm(strategy_json, pine_script, output_filename)
            
            logger.info(f"Algorithm saved as {output_filename}.pine and {output_filename}.json")
            
        except Exception as e:
            logger.error(f"Error generating algorithm: {e}")
            raise

    def get_next_algorithm_number(self) -> int:
        """Find the next available algorithm number."""
        try:
            output_path = os.path.join('output', 'algo')
            if not os.path.exists(output_path):
                return 1
                
            pattern = f"{self.company_name}_algorithm-*.pine"
            existing_files = glob.glob(os.path.join(output_path, pattern))
            
            if not existing_files:
                return 1
                
            numbers = []
            for file in existing_files:
                match = re.search(rf"{self.company_name}_algorithm-(\d+)\.pine", file)
                if match:
                    numbers.append(int(match.group(1)))
            
            return max(numbers) + 1 if numbers else 1
            
        except Exception as e:
            logger.error(f"Error finding next algorithm number: {e}")
            return 1

    def save_algorithm(self, strategy_json: Dict, pine_script: str, output_filename: str) -> None:
        """Save the generated algorithm in both JSON and Pine Script formats."""
        try:
            output_path = os.path.join('output', 'algo')
            os.makedirs(output_path, exist_ok=True)
            
            # Add metadata to JSON
            strategy_json.update({
                "generated_at": datetime.now().isoformat(),
                "company": self.company_name,
                "api_provider": self.api_provider,
                "model": self.model
            })
            
            # Save JSON
            json_path = os.path.join(output_path, f"{output_filename}.json")
            with open(json_path, 'w') as f:
                json.dump(strategy_json, f, indent=4)
            
            # Save Pine Script
            pine_path = os.path.join(output_path, f"{output_filename}.pine")
            with open(pine_path, 'w') as f:
                f.write(pine_script)
                
        except Exception as e:
            logger.error(f"Error saving algorithm: {e}")
            raise

# if __name__ == "__main__":
#     try:
#         algo_agent = AlgoAgent("ZOMATO")
#         algo_agent.generate_algorithms()
#     except Exception as e:
#         logger.error(f"Main execution failed: {e}")
#         raise