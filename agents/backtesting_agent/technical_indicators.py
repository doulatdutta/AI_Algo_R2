import numpy as np
import pandas as pd
import talib

class TechnicalIndicators:
    @staticmethod
    def supertrend(high, low, close, period=10, multiplier=3):
        """Calculate SuperTrend indicator"""
        try:
            atr = talib.ATR(high, low, close, timeperiod=period)
            
            # Calculate basic upper and lower bands
            hl2 = (high + low) / 2
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)
            
            # Initialize SuperTrend
            supertrend = np.zeros_like(close)
            direction = np.zeros_like(close)  # 1 for uptrend, -1 for downtrend
            
            # Calculate SuperTrend values
            for i in range(1, len(close)):
                if close[i] > upper_band[i-1]:
                    direction[i] = 1
                elif close[i] < lower_band[i-1]:
                    direction[i] = -1
                else:
                    direction[i] = direction[i-1]
                    
                    if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                        lower_band[i] = lower_band[i-1]
                    if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                        upper_band[i] = upper_band[i-1]
                
                if direction[i] == 1:
                    supertrend[i] = lower_band[i]
                else:
                    supertrend[i] = upper_band[i]
                    
            return supertrend, direction
            
        except Exception as e:
            raise Exception(f"Error calculating SuperTrend: {str(e)}")

    @staticmethod
    def vwap(high, low, close, volume):
        """Calculate VWAP (Volume Weighted Average Price)"""
        try:
            typical_price = (high + low + close) / 3
            vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)
            return vwap
            
        except Exception as e:
            raise Exception(f"Error calculating VWAP: {str(e)}")

    @staticmethod
    def stochastic(high, low, close, k_period=14, d_period=3, slowing=3):
        """Calculate Stochastic Oscillator"""
        try:
            k, d = talib.STOCH(high, low, close, 
                              fastk_period=k_period,
                              slowk_period=slowing,
                              slowk_matype=0,
                              slowd_period=d_period,
                              slowd_matype=0)
            return k, d
            
        except Exception as e:
            raise Exception(f"Error calculating Stochastic: {str(e)}")

    @staticmethod
    def calculate_all_indicators(df, params=None):
        """Calculate all technical indicators with given parameters"""
        try:
            if params is None:
                params = {
                    'sma_period': 20,
                    'ema_period': 20,
                    'rsi_period': 14,
                    'macd_fast': 12,
                    'macd_slow': 26,
                    'macd_signal': 9,
                    'bb_period': 20,
                    'bb_dev': 2,
                    'atr_period': 14,
                    'supertrend_period': 10,
                    'supertrend_multiplier': 3,
                    'stoch_k': 14,
                    'stoch_d': 3,
                    'stoch_slowing': 3
                }

            indicators = {}
            
            # Moving Averages
            indicators['sma'] = talib.SMA(df['close'], timeperiod=params['sma_period'])
            indicators['ema'] = talib.EMA(df['close'], timeperiod=params['ema_period'])
            indicators['vwap'] = TechnicalIndicators.vwap(df['high'], df['low'], 
                                                        df['close'], df['volume'])

            # Momentum Indicators
            indicators['rsi'] = talib.RSI(df['close'], timeperiod=params['rsi_period'])
            indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = \
                talib.MACD(df['close'], fastperiod=params['macd_fast'],
                          slowperiod=params['macd_slow'], 
                          signalperiod=params['macd_signal'])
            indicators['stoch_k'], indicators['stoch_d'] = \
                TechnicalIndicators.stochastic(df['high'], df['low'], df['close'],
                                             params['stoch_k'], params['stoch_d'],
                                             params['stoch_slowing'])

            # Volatility Indicators
            indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = \
                talib.BBANDS(df['close'], timeperiod=params['bb_period'],
                           nbdevup=params['bb_dev'], nbdevdn=params['bb_dev'])
            indicators['atr'] = talib.ATR(df['high'], df['low'], df['close'],
                                        timeperiod=params['atr_period'])

            # Volume Indicators
            indicators['obv'] = talib.OBV(df['close'], df['volume'])
            indicators['mfi'] = talib.MFI(df['high'], df['low'], df['close'],
                                        df['volume'], timeperiod=params['rsi_period'])

            # Trend Indicators
            indicators['adx'] = talib.ADX(df['high'], df['low'], df['close'],
                                        timeperiod=params['atr_period'])
            indicators['supertrend'], indicators['supertrend_direction'] = \
                TechnicalIndicators.supertrend(df['high'], df['low'], df['close'],
                                             params['supertrend_period'],
                                             params['supertrend_multiplier'])

            return indicators
            
        except Exception as e:
            raise Exception(f"Error calculating indicators: {str(e)}")
