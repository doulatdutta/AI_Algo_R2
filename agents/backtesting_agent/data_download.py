import yfinance as yf
import pandas as pd
import numpy as np
import yaml
import os
from datetime import datetime, timedelta, date
import pytz
import talib
import logging
from typing import List, Optional, Tuple
import time

class HistoricalDataDownloader:
    def __init__(self):
        """Initialize the HistoricalDataDownloader with configuration and logging"""
        self.setup_logging()
        self.config = self.load_config()
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        
    def setup_logging(self):
        """Set up logging configuration"""
        log_dir = os.path.join('agents', 'backtesting_agent', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'data_download.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_config(self) -> dict:
        """Load configuration from config.yaml and ensure all necessary parameters are present"""
        try:
            with open("config/config.yaml", "r") as file:
                config = yaml.safe_load(file)
                self.logger.info("Configuration loaded successfully")
                
                # Required configuration keys
                required_keys = ["historical_days", "chunk_size", "data_interval", "max_attempts", 
                                "delay_between_attempts", "delay_between_chunks"]
                                
                # Check for missing keys
                for key in required_keys:
                    if key not in config:
                        self.logger.error(f"Missing '{key}' in config.yaml")
                        raise KeyError(f"Missing '{key}' in config.yaml")
                        
                return config
                
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            raise

    def check_existing_data(self, company_name: str) -> Tuple[bool, Optional[pd.DataFrame], Optional[datetime]]:
        """
        Check if data file exists and return its contents and last date
        Returns: (exists, dataframe, last_date)
        """
        output_path = os.path.join('agents', 'backtesting_agent', 'historical_data')
        file_path = os.path.join(output_path, f"{company_name}_minute.csv")
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if not df.empty:
                    # Convert datetime column
                    df['Datetime'] = pd.to_datetime(df['Datetime'])
                    last_date = df['Datetime'].max()
                    
                    self.logger.info(f"Found existing data for {company_name} up to {last_date}")
                    return True, df, last_date
                    
            except Exception as e:
                self.logger.error(f"Error reading existing file: {e}")
                
        return False, None, None

    def is_market_holiday(self, date) -> bool:
        """Check if a given date is a market holiday"""
        holidays = [
            "2024-01-26", "2024-03-08", "2024-03-25", "2024-03-29",
            "2024-04-11", "2024-04-17", "2024-05-01", "2024-06-17",
            "2024-07-17", "2024-08-15", "2024-09-02", "2024-10-02",
            "2024-11-01", "2024-11-15", "2024-12-25"
        ]
        
        if isinstance(date, str):
            date = pd.to_datetime(date).date()
        elif isinstance(date, pd.Timestamp):
            date = date.date()
            
        if date.weekday() >= 5:  # Weekend
            return True
            
        return date.strftime("%Y-%m-%d") in holidays

    def get_market_hours(self, date) -> tuple:
        """Get market hours for a given date in IST"""
        date_str = date.strftime("%Y-%m-%d")
        market_start = pd.Timestamp(f"{date_str} 09:15:00", tz=self.ist_tz)
        market_end = pd.Timestamp(f"{date_str} 15:30:00", tz=self.ist_tz)
        return market_start, market_end

    def download_minute_data(self, ticker: yf.Ticker, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Download minute data with retry mechanism"""
        max_attempts = self.config.get('max_attempts', 6)
        delay = self.config.get('delay_between_attempts', 10)
        
        for attempt in range(max_attempts):
            try:
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval="1m"
                )
                if not data.empty:
                    return data
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_attempts - 1:
                    time.sleep(delay)
                    
        return pd.DataFrame()

    def merge_and_save_data(self, old_data: Optional[pd.DataFrame], new_data: pd.DataFrame, 
                           company_name: str) -> pd.DataFrame:
        """Merge old and new data, remove duplicates, and save"""
        try:
            if old_data is not None and not old_data.empty:
                # Combine old and new data
                combined_data = pd.concat([old_data, new_data])
                
                # Convert datetime to consistent format
                combined_data['Datetime'] = pd.to_datetime(combined_data['Datetime'])
                
                # Remove duplicates based on Datetime
                combined_data = combined_data.drop_duplicates(subset=['Datetime'], keep='last')
                
                # Sort by datetime
                combined_data = combined_data.sort_values('Datetime')
            else:
                combined_data = new_data
            
            # Process the combined data
            processed_data = self.process_data(combined_data)
            
            # Save the data
            self.save_data(processed_data, company_name)
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error merging data: {e}")
            raise

    def download_historical_data(self, company_name: str) -> bool:
        """Download historical minute data in 30-day batches with waiting periods"""
        try:
            symbol = self.get_stock_symbol(company_name)
            total_days = self.config.get('historical_days', 120)  # Total days to download
            chunk_size = self.config.get('chunk_size', 7)  # Days per chunk
            chunk_delay = self.config.get('delay_between_chunks', 65)  # Delay between chunks
            batch_delay = self.config.get('batch_delay', 180)  # Delay between 30-day batches
            
            # Check existing data
            has_existing, existing_data, last_date = self.check_existing_data(company_name)
            
            end_date = datetime.now(self.ist_tz).date()
            start_date = end_date - timedelta(days=total_days)
            
            if has_existing and last_date:
                start_date = last_date.date() + timedelta(days=1)
                self.logger.info(f"Downloading incremental data from {start_date} to {end_date}")
            else:
                self.logger.info(f"Downloading full {total_days} days of data for {symbol}")
            
            if start_date >= end_date:
                self.logger.info("Data is already up to date")
                return True
            
            ticker = yf.Ticker(symbol)
            new_data = []
            
            # Calculate number of 30-day batches needed
            current_batch_end = end_date
            remaining_days = (current_batch_end - start_date).days
            
            while remaining_days > 0:
                # Calculate batch dates
                batch_start = max(current_batch_end - timedelta(days=30), start_date)
                self.logger.info(f"\nStarting new 30-day batch: {batch_start} to {current_batch_end}")
                
                # Download data in 7-day chunks within this 30-day batch
                current_chunk_end = current_batch_end
                batch_data = []
                
                while current_chunk_end > batch_start:
                    current_chunk_start = max(current_chunk_end - timedelta(days=chunk_size), batch_start)
                    
                    self.logger.info(f"Downloading chunk: {current_chunk_start} to {current_chunk_end}")
                    
                    current_date = current_chunk_start
                    chunk_data = []
                    
                    while current_date <= current_chunk_end:
                        if not self.is_market_holiday(current_date):
                            market_start, market_end = self.get_market_hours(current_date)
                            
                            # Download data for the day
                            day_data = self.download_minute_data(ticker, market_start, market_end)
                            
                            if not day_data.empty:
                                chunk_data.append(day_data)
                                self.logger.info(f"Successfully downloaded data for {current_date}")
                            else:
                                self.logger.warning(f"No data available for {current_date}")
                        
                        current_date += timedelta(days=1)
                    
                    if chunk_data:
                        batch_data.extend(chunk_data)
                        
                        # Add delay between chunks within the batch
                        if current_chunk_start > batch_start:
                            self.logger.info(f"Waiting {chunk_delay} seconds before next chunk...")
                            time.sleep(chunk_delay)
                    
                    current_chunk_end = current_chunk_start - timedelta(days=1)
                
                # Add batch data to overall new data
                if batch_data:
                    new_data.extend(batch_data)
                    
                    # Save intermediate results after each 30-day batch
                    if len(new_data) > 0:
                        combined_new_data = pd.concat(new_data)
                        self.merge_and_save_data(existing_data, combined_new_data, company_name)
                        self.logger.info(f"Saved intermediate data after batch {batch_start} to {current_batch_end}")
                
                # Update for next batch
                current_batch_end = batch_start - timedelta(days=1)
                remaining_days = (current_batch_end - start_date).days
                
                # Add delay between 30-day batches if more data to download
                if remaining_days > 0:
                    self.logger.info(f"\nWaiting {batch_delay} seconds before starting next 30-day batch...")
                    time.sleep(batch_delay)
            
            if not new_data:
                if has_existing:
                    self.logger.info("No new data to download")
                    return True
                else:
                    self.logger.error(f"No data downloaded for {symbol}")
                    return False
            
            # Final merge and save
            combined_new_data = pd.concat(new_data)
            self.merge_and_save_data(existing_data, combined_new_data, company_name)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading data for {company_name}: {e}")
            return False

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process and clean the minute data"""
        try:
            data = data.reset_index()
            
            # Ensure datetime is in IST
            if data['Datetime'].dt.tz is None:
                data['Datetime'] = data['Datetime'].dt.tz_localize('UTC').dt.tz_convert(self.ist_tz)
            else:
                data['Datetime'] = data['Datetime'].dt.tz_convert(self.ist_tz)
            
            # Add date and time columns
            data['Date'] = data['Datetime'].dt.date
            data['Time'] = data['Datetime'].dt.time
            
            # Drop rows with NaN values
            data = data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
            
            # Add holiday information
            data['is_holiday'] = data['Date'].map(self.is_market_holiday)
            
            # Calculate returns
            data['Minute_Return'] = data['Close'].pct_change()
            data['Log_Return'] = np.log(data['Close']).diff()
            
            # Add technical indicators if enough data points
            if len(data) > 50:
                data['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
                data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
                
                data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = talib.MACD(
                    data['Close'], fastperiod=12, slowperiod=26, signalperiod=9
                )
                
                data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = talib.BBANDS(
                    data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2
                )
                
                data['OBV'] = talib.OBV(data['Close'], data['Volume'])
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error processing data: {e}")
            return data

    def get_stock_symbol(self, company_name: str) -> str:
        """Ensure company name has proper stock exchange suffix"""
        if not company_name.endswith(('.NS', '.BO')):
            company_name = f"{company_name}.NS"
        return company_name

    def save_data(self, data: pd.DataFrame, company_name: str) -> str:
        """Save the processed data to CSV file"""
        output_path = os.path.join('agents', 'backtesting_agent', 'historical_data')
        os.makedirs(output_path, exist_ok=True)
        
        output_file = os.path.join(output_path, f"{company_name}_minute.csv")
        data.to_csv(output_file, index=False)
        self.logger.info(f"Data saved to {output_file}")
        
        return output_file


## Example usage
def main():
    """Example usage"""
    downloader = HistoricalDataDownloader()
    stocks = ["ZOMATO"]  # Start with one stock for testing
    
    for stock in stocks:
        success = downloader.download_historical_data(stock)
        if success:
            print(f"Successfully downloaded/updated data for {stock}")
        else:
            print(f"Failed to download/update data for {stock}")

if __name__ == "__main__":
    main()