import json
import os
from datetime import datetime
import ollama
import yaml
import openai
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlgoAgent:
    def __init__(self, company_name: str, num_algorithms: int):
        self.company_name = company_name
        self.num_algorithms = num_algorithms
        self.base_path = Path(os.getcwd())
        self.config = self.load_config()
        self.setup_api_client()
        
    def load_config(self) -> dict:
        """Load configuration from config file."""
        try:
            with open("config/config.yaml", "r") as file:
                config = yaml.safe_load(file)
                
            # Set API preferences
            self.use_openai = config.get('use_openai', False)
            
            if self.use_openai:
                openai.api_key = config.get('OPENAI_API_KEY')
                self.openai_model = config.get('openai_model', 'gpt-4-turbo-preview')
            else:
                self.ollama_model = config.get('ollama_model', {}).get('model', 'mistral')
                
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def setup_api_client(self) -> None:
        """Setup the appropriate API client based on configuration."""
        if self.use_openai:
            if not openai.api_key:
                raise ValueError("OpenAI API key not found in configuration")
            logger.info("Using OpenAI API for generation")
        else:
            logger.info("Using Ollama for generation")

    def generate_chat_completion(self, system_prompt: str, user_prompt: str) -> str:
        """Generate completion using either OpenAI or Ollama."""
        try:
            if self.use_openai:
                response = openai.ChatCompletion.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                return response.choices[0].message.content
            else:
                response = ollama.chat(
                    model=self.ollama_model,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ]
                )
                return response['message']['content']
                
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            raise

    def generate_algorithms(self) -> None:
        """Generate multiple PineScript trading algorithms."""
        logger.info(f"Generating {self.num_algorithms} PineScript algorithms for {self.company_name}")
        
        for i in range(self.num_algorithms):
            algo_num = 1
            while True:
                output_filename = f"{self.company_name}_algorithm-{algo_num}"
                if not os.path.exists(os.path.join('output', 'algo', f"{output_filename}.pine")):
                    break
                algo_num += 1
            
            try:
                pine_script = self.generate_pinescript()
                self.save_algorithm(pine_script, output_filename)
                logger.info(f"Algorithm {algo_num}: saved as {output_filename}.pine and {output_filename}.json")
            except Exception as e:
                logger.error(f"Error generating algorithm {algo_num}: {e}")
                continue

    def generate_pinescript(self) -> str:
        """Generate a complete PineScript trading strategy """
        system_prompt = """You are a professional Pine Script programmer specializing in algorithmic trading strategies. 
        Generate a complete Pine Script v5 trading strategy that includes multiple technical indicators and clear entry/exit conditions.
        Just write pine code, do not write introductory or explanatory text. Do not include any comments in the code."""

        user_prompt = """Create a complete Pine Script v5 strategy with the following requirements:

        1. Use a combination of these indicators:
           - Moving Averages (SMA, EMA, VWAP)
           - Momentum (RSI, MACD, Stochastic)
           - Volatility (Bollinger Bands, ATR)
           - Volume (OBV, MFI)
           - Trend (ADX, Supertrend)

        2. Include:
           - Strategy settings with input parameters
           - Clear entry and exit conditions
           - Position sizing and risk management
           - Stop loss and take profit logic
           - Proper variable declarations and calculations
           - Complete strategy.entry and strategy.exit calls

        3. Format:
           - Must be valid Pine Script v5 syntax
           - Include proper comments and sections
           - Use risk management parameters
           - Include plotting for visual reference

        Generate a complete, working strategy that could be directly copied into TradingView. Write 1000 lines of code or more. 
        Also think before wite the code so it will be currect code 100% and will not give any error.
        
        Example of currect pine code:- 


        //@version=5
        strategy(title='SAIYAN OCC Strategy R5.41', overlay=true, pyramiding=0, default_qty_type=strategy.percent_of_equity, default_qty_value=10, calc_on_every_tick=false)

        // === INPUTS ===
        res = input.timeframe(title='TIMEFRAME', defval='15', group="NON REPAINT")
        useRes = input(defval=true, title='Use Alternate Signals')
        intRes = input(defval=8, title='Multiplier for Alternate Signals')
        stratRes = timeframe.ismonthly ? str.tostring(timeframe.multiplier * intRes, '###M') : 
                timeframe.isweekly ? str.tostring(timeframe.multiplier * intRes, '###W') : 
                timeframe.isdaily ? str.tostring(timeframe.multiplier * intRes, '###D') : 
                timeframe.isintraday ? str.tostring(timeframe.multiplier * intRes, '####') : '60'

        // MA Type Selection
        basisType = input.string(defval='ALMA', title='MA Type', options=['TEMA', 'HullMA', 'ALMA'])
        basisLen = input.int(defval=2, title='MA Period', minval=1)
        offsetSigma = input.float(defval=5.0, title='Offset for LSMA / Sigma for ALMA', minval=0.0)
        offsetALMA = input.float(defval=0.85, title='Offset for ALMA', minval=0.0, step=0.01)
        scolor = input(true, title='Show colored Bars to indicate Trend?')
        delayOffset = input.int(defval=0, title='Delay Open/Close MA (Forces Non-Repainting)', minval=0, step=1)
        tradeType = input.string('BOTH', title='Trade Direction', options=['LONG', 'SHORT', 'BOTH', 'NONE'])

        // Heikin Ashi Option
        h = input(false, title='Use Heikin Ashi Candles')
        src = h ? request.security(ticker.heikinashi(syminfo.tickerid), timeframe.period, close, lookahead=barmerge.lookahead_off) : close

        // Supply/Demand Settings
        swing_length = input.int(10, title='Swing High/Low Length', group='Settings', minval=1, maxval=50)
        history_of_demand_to_keep = input.int(20, title='History To Keep', minval=5, maxval=50)
        box_width = input.float(2.5, title='Supply/Demand Box Width', group='Settings', minval=1, maxval=10, step=0.5)

        // Visual Settings
        show_zigzag = input.bool(false, title='Show Zig Zag', group='Visual Settings')
        show_price_action_labels = input.bool(false, title='Show Price Action Labels', group='Visual Settings')

        // Colors
        supply_color = input.color(color.new(#EDEDED, 70), title='Supply', group='Visual Settings')
        supply_outline_color = input.color(color.new(color.white, 75), title='Supply Outline', group='Visual Settings')
        demand_color = input.color(color.new(#00FFFF, 70), title='Demand', group='Visual Settings')
        demand_outline_color = input.color(color.new(color.white, 75), title='Demand Outline', group='Visual Settings')
        bos_label_color = input.color(color.white, title='BOS Label', group='Visual Settings')
        poi_label_color = input.color(color.white, title='POI Label', group='Visual Settings')
        swing_type_color = input.color(color.black, title='Price Action Label', group='Visual Settings')
        zigzag_color = input.color(color.new(#000000, 0), title='Zig Zag', group='Visual Settings')

        // Strategy Settings
        slPoints = input.int(defval=0, title='Initial Stop Loss Points (0 to disable)', minval=0)
        tpPoints = input.int(defval=0, title='Initial Take Profit Points (0 to disable)', minval=0)
        max_bars_back = input.int(defval=4000, title='Number of Bars for Back Testing', minval=0)

        // Alert Messages
        i_alert_txt_entry_long = input.text_area(defval="", title="Long Entry Message", group="Alerts")
        i_alert_txt_entry_short = input.text_area(defval="", title="Short Entry Message", group="Alerts")

        // === FUNCTIONS ===
        // Moving Average Variant Function
        variant(type, src, len, offSig, offALMA) =>
            float result = 0.0
            if type == 'EMA'
                result := ta.ema(src, len)
            else if type == 'TEMA'
                ema1 = ta.ema(src, len)
                ema2 = ta.ema(ema1, len)
                ema3 = ta.ema(ema2, len)
                result := 3 * (ema1 - ema2) + ema3
            else if type == 'HullMA'
                result := ta.wma(2 * ta.wma(src, len / 2) - ta.wma(src, len), math.round(math.sqrt(len)))
            else if type == 'ALMA'
                result := ta.alma(src, len, offALMA, offSig)
            else
                result := ta.sma(src, len)
            result

        // Security wrapper with non-repainting
        securityNoRep(sym, res, src) =>
            request.security(sym, res, src, barmerge.gaps_off, barmerge.lookahead_off)

        // === CALCULATIONS ===
        closeSeries = variant(basisType, close[delayOffset], basisLen, offsetSigma, offsetALMA)
        openSeries = variant(basisType, open[delayOffset], basisLen, offsetSigma, offsetALMA)

        // Get alternate resolution if selected
        closeSeriesAlt = useRes ? request.security(syminfo.tickerid, stratRes, closeSeries, barmerge.gaps_off, barmerge.lookahead_off) : closeSeries
        openSeriesAlt = useRes ? request.security(syminfo.tickerid, stratRes, openSeries, barmerge.gaps_off, barmerge.lookahead_off) : openSeries

        // Entry conditions
        buy = ta.crossover(closeSeriesAlt, openSeriesAlt) and (tradeType == 'LONG' or tradeType == 'BOTH')
        sell = ta.crossunder(closeSeriesAlt, openSeriesAlt) and (tradeType == 'SHORT' or tradeType == 'BOTH')

        // Plot signals
        plotshape(buy, title="Buy", text="Buy", style=shape.labelup, location=location.belowbar, color=#00DBFF, textcolor=color.white, size=size.tiny)
        plotshape(sell, title="Sell", text="Sell", style=shape.labeldown, location=location.abovebar, color=#E91E63, textcolor=color.white, size=size.tiny)

        // Strategy orders
        if buy
            strategy.entry("Long", strategy.long, alert_message=i_alert_txt_entry_long)
            if tpPoints > 0
                strategy.exit("TP Long", "Long", limit=close + tpPoints, stop=slPoints > 0 ? close - slPoints : na)

        if sell
            strategy.entry("Short", strategy.short, alert_message=i_alert_txt_entry_short)
            if tpPoints > 0
                strategy.exit("TP Short", "Short", limit=close - tpPoints, stop=slPoints > 0 ? close + slPoints : na)

        """

        try:
            return self.generate_chat_completion(system_prompt, user_prompt)
        except Exception as e:
            raise Exception(f"Error generating Pine Script: {e}")

    def extract_strategy_conditions(self, pine_script: str) -> Dict:
        """Extract and parse strategy conditions from Pine Script."""
        conditions = {
            "entry_conditions": [],
            "exit_conditions": [],
            "indicators": [],
            "risk_params": {}
        }
        
        try:
            # Extract indicators
            indicator_patterns = [
                r'(rsi|macd|sma|ema|bb|atr|obv|mfi|adx|supertrend)\s*=\s*ta\.',
                r'input\.(float|int|bool)\s*\(\s*["\'](\w+)["\']'
            ]
            
            # Extract entry conditions
            entry_patterns = [
                r'strategy\.entry\s*\([^)]*\)',
                r'if\s+(.+?)\s+strategy\.entry'
            ]
            
            # Extract exit conditions
            exit_patterns = [
                r'strategy\.exit\s*\([^)]*\)',
                r'if\s+(.+?)\s+strategy\.close'
            ]
            
            # Extract risk parameters
            risk_patterns = {
                'stop_loss': r'stop\s*=\s*(\d+(\.\d+)?)',
                'take_profit': r'limit\s*=\s*(\d+(\.\d+)?)',
                'qty': r'qty\s*=\s*(\d+(\.\d+)?)'
            }
            
            # Parse the script and populate conditions
            for line in pine_script.split('\n'):
                line = line.strip()
                
                # Parse indicators
                for pattern in indicator_patterns:
                    if indicator := self._extract_pattern(pattern, line):
                        conditions["indicators"].append(indicator)
                
                # Parse entry conditions
                for pattern in entry_patterns:
                    if condition := self._extract_pattern(pattern, line):
                        conditions["entry_conditions"].append(condition)
                
                # Parse exit conditions
                for pattern in exit_patterns:
                    if condition := self._extract_pattern(pattern, line):
                        conditions["exit_conditions"].append(condition)
                
                # Parse risk parameters
                for param, pattern in risk_patterns.items():
                    if value := self._extract_pattern(pattern, line):
                        conditions["risk_params"][param] = float(value)
            
            return conditions
            
        except Exception as e:
            logger.error(f"Error extracting strategy conditions: {e}")
            return conditions

    def _extract_pattern(self, pattern: str, line: str) -> Optional[str]:
        """Helper method to extract pattern from line."""
        import re
        try:
            match = re.search(pattern, line)
            if match:
                return match.group(1)
        except Exception as e:
            logger.error(f"Error extracting pattern: {e}")
        return None

    def save_algorithm(self, pine_script: str, output_filename: str) -> None:
        """Save the generated algorithm as Pine Script and JSON files."""
        try:
            output_path = os.path.join('output', 'algo')
            os.makedirs(output_path, exist_ok=True)
            
            # Save Pine Script
            pine_path = os.path.join(output_path, f"{output_filename}.pine")
            with open(pine_path, 'w') as f:
                f.write(pine_script)
            
            # Extract and save strategy conditions as JSON
            conditions = self.extract_strategy_conditions(pine_script)
            
            json_content = {
                "pine_script_version": 5,
                "strategy_name": output_filename,
                "company": self.company_name,
                "generated_at": datetime.now().isoformat(),
                "model": self.openai_model if self.use_openai else self.ollama_model,
                "api_provider": "OpenAI" if self.use_openai else "Ollama",
                "conditions": conditions
            }
            
            json_path = os.path.join(output_path, f"{output_filename}.json")
            with open(json_path, 'w') as f:
                json.dump(json_content, f, indent=4)
                
        except Exception as e:
            logger.error(f"Error saving algorithm: {e}")
            raise

#       ### Example usage
# if __name__ == "__main__":
#     try:
#         algo_agent = AlgoAgent("ZOMATO", 1)
#         algo_agent.generate_algorithms()
#     except Exception as e:
#         logger.error(f"Main execution failed: {e}")
#         raise