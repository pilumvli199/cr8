# railway_crypto_bot.py (updated: uses `ta` and pandas-only calculations; no TA-Lib)
import os
import asyncio
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import ccxt
from openai import OpenAI
import telegram
from telegram.ext import Application, CommandHandler
import schedule
import time
from typing import List, Dict, Any

# ta indicators
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CryptoTradingBot:
    def __init__(self):
        # Environment variables
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        # Initialize Telegram bot
        self.telegram_app = Application.builder().token(self.telegram_token).build()
        
        # Initialize Binance (public endpoints)
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
        })
        
        # Trading pairs to monitor
        self.symbols = [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT',
            'LTC/USDT', 'AVAX/USDT', 'LINK/USDT', 'ADA/USDT', 'DOGE/USDT'
        ]
        
        # Technical analysis parameters
        self.timeframes = ['30m', '1h', '4h']
        self.rsi_period = 14
        self.ema_periods = [20, 50, 200]
        
        # Signal thresholds
        self.min_confidence = 80
        self.max_signals_per_hour = 3
        self.signal_count = 0
        self.last_signal_time = datetime.now()

    async def fetch_market_data(self, symbol: str, timeframe: str = '30m', limit: int = 100) -> pd.DataFrame:
        \"\"\"Fetch OHLCV data from Binance\"\"\"
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            # ensure numeric types
            df = df.astype(float)
            return df
        except Exception as e:
            logger.error(f\"Error fetching data for {symbol}: {str(e)}\")
            return pd.DataFrame()

    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        \"\"\"Calculate technical indicators using `ta` and pandas\"\"\"
        if df.empty or len(df) < 50:
            return {}
        
        try:
            indicators = {}
            close_s = df['close']
            high_s = df['high']
            low_s = df['low']
            vol_s = df['volume']

            # RSI
            rsi = RSIIndicator(close_s, window=self.rsi_period).rsi()
            indicators['rsi'] = float(rsi.iloc[-1])

            # MACD
            macd_obj = MACD(close=close_s)
            indicators['macd'] = float(macd_obj.macd().iloc[-1])
            indicators['macd_signal'] = float(macd_obj.macd_signal().iloc[-1])
            indicators['macd_histogram'] = float(macd_obj.macd_diff().iloc[-1])

            # EMAs
            for period in self.ema_periods:
                ema_val = EMAIndicator(close_s, window=period).ema_indicator().iloc[-1]
                indicators[f'ema_{period}'] = float(ema_val)

            # Bollinger Bands (20, 2)
            bb = BollingerBands(close_s, window=20, window_dev=2)
            indicators['bb_upper'] = float(bb.bollinger_hband().iloc[-1])
            indicators['bb_middle'] = float(bb.bollinger_mavg().iloc[-1])
            indicators['bb_lower'] = float(bb.bollinger_lband().iloc[-1])

            # Volume analysis
            indicators['volume_sma'] = float(vol_s.rolling(window=20).mean().iloc[-1])
            indicators['current_volume'] = float(vol_s.iloc[-1])
            indicators['volume_ratio'] = float(vol_s.iloc[-1] / indicators['volume_sma']) if indicators['volume_sma'] and indicators['volume_sma'] > 0 else 0.0

            # Price levels
            indicators['current_price'] = float(close_s.iloc[-1])
            # If 30m timeframe and limit >=48, last 48 bars ~ 24h
            indicators['price_change_24h'] = float(((close_s.iloc[-1] - close_s.iloc[-48]) / close_s.iloc[-48]) * 100) if len(close_s) >= 48 else 0.0

            return indicators
        except Exception as e:
            logger.error(f\"Error calculating indicators: {str(e)}\")
            return {}

    def detect_patterns(self, df: pd.DataFrame) -> List[str]:
        \"\"\"Heuristic candlestick & trend pattern detection (no TA-Lib)\"\"\"
        if df.empty or len(df) < 20:
            return []
        
        patterns = []
        try:
            open_p = df['open'].values
            high_p = df['high'].values
            low_p = df['low'].values
            close_p = df['close'].values

            # Last two candles
            o1, h1, l1, c1 = open_p[-2], high_p[-2], low_p[-2], close_p[-2]
            o2, h2, l2, c2 = open_p[-1], high_p[-1], low_p[-1], close_p[-1]

            # Doji: very small body relative to range
            body = abs(c2 - o2)
            range2 = h2 - l2 if (h2 - l2) > 0 else 1e-9
            if body <= 0.15 * range2:
                patterns.append(\"Doji\")


            # Hammer: long lower wick and small body near top of candle
            lower_wick = min(o2, c2) - l2
            upper_wick = h2 - max(o2, c2)
            if lower_wick > 2 * body and upper_wick < 0.5 * body:
                patterns.append(\"Hammer\")

            # Engulfing (simple): current body engulfs previous and opposite direction
            prev_body = abs(c1 - o1)
            curr_body = abs(c2 - o2)
            if curr_body > prev_body:
                # bull engulfing
                if (c1 < o1) and (c2 > o2) and (c2 >= o1) and (o2 <= c1):
                    patterns.append(\"Bullish Engulfing\")
                # bear engulfing
                if (c1 > o1) and (c2 < o2) and (o2 >= c1) and (c2 <= o1):
                    patterns.append(\"Bearish Engulfing\")

            # Simple trend: higher highs / lower highs in last 5 highs
            recent_highs = high_p[-10:]
            recent_lows = low_p[-10:]
            if len(recent_highs) >= 5:
                if all(recent_highs[i] >= recent_highs[i-1] for i in range(1,5)):
                    patterns.append(\"Higher Highs\")
                elif all(recent_highs[i] <= recent_highs[i-1] for i in range(1,5)):
                    patterns.append(\"Lower Highs\")

            return patterns
        except Exception as e:
            logger.error(f\"Error detecting patterns: {str(e)}\")
            return []

    async def analyze_with_openai(self, symbol: str, indicators: Dict, patterns: List[str]) -> Dict[str, Any]:
        \"\"\"Use OpenAI to analyze the data and generate signals\"\"\"
        try:
            prompt = f\"\"\"
            Analyze this crypto trading data for {symbol}:

            Technical Indicators:
            - RSI: {indicators.get('rsi', 'N/A')}
            - MACD: {indicators.get('macd', 'N/A')}
            - MACD Signal: {indicators.get('macd_signal', 'N/A')}
            - Current Price: ${indicators.get('current_price', 'N/A')}
            - 24h Change: {indicators.get('price_change_24h', 'N/A')}%
            - Volume Ratio: {indicators.get('volume_ratio', 'N/A')}x
            - EMA 20: {indicators.get('ema_20', 'N/A')}
            - EMA 50: {indicators.get('ema_50', 'N/A')}

            Detected Patterns: {', '.join(patterns) if patterns else 'None'}

            Based on this data, provide:
            1. Signal: BUY, SELL, or HOLD
            2. Confidence: 0-100%
            3. Reason: Brief explanation
            4. Target: Potential price target if BUY/SELL
            5. Stop Loss: Risk management level

            Respond in JSON format only:
            {{
                \"signal\": \"BUY/SELL/HOLD\",
                \"confidence\": 85,
                \"reason\": \"Strong bullish momentum with volume confirmation\",
                \"target\": 45000,
                \"stop_loss\": 42000
            }}
            \"\"\"

            response = self.openai_client.chat.completions.create(
                model=\"gpt-4o-mini\",
                messages=[{\"role\": \"user\", \"content\": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            
            # Parse JSON response (guarded)
            content = response.choices[0].message.content
            analysis = json.loads(content)
            return analysis
            
        except Exception as e:
            logger.error(f\"Error in OpenAI analysis: {str(e)}\")
            return {\"signal\": \"HOLD\", \"confidence\": 0, \"reason\": \"Analysis error\"}

    async def send_telegram_alert(self, symbol: str, analysis: Dict, indicators: Dict):
        \"\"\"Send signal to Telegram\"\"\"
        try:
            signal = analysis.get('signal', 'HOLD')
            confidence = analysis.get('confidence', 0)
            
            if signal == 'HOLD' or confidence < self.min_confidence:
                return
            
            # Rate limiting
            current_time = datetime.now()
            if (current_time - self.last_signal_time).total_seconds() < 1200:  # 20 minutes
                if self.signal_count >= self.max_signals_per_hour:
                    return
            else:
                self.signal_count = 0
                self.last_signal_time = current_time
            
            emoji = "🚀" if signal == "BUY" else "🔻"
            
            message = f\"\"\"
{emoji} <b>STRONG {signal} SIGNAL</b>

<b>Coin:</b> {symbol}
<b>Price:</b> ${indicators.get('current_price', 0):.4f}
<b>24h Change:</b> {indicators.get('price_change_24h', 0):.2f}%

<b>Analysis:</b>
• {analysis.get('reason', 'No reason provided')}
• Confidence: {confidence}%

<b>Targets:</b>
• Target: ${analysis.get('target', 0):.4f}
• Stop Loss: ${analysis.get('stop_loss', 0):.4f}

<b>Volume:</b> {indicators.get('volume_ratio', 0):.2f}x average
<b>RSI:</b> {indicators.get('rsi', 0):.1f}

<i>Time: {datetime.now().strftime('%H:%M UTC')}</i>
            \"\"\"
            
            bot = telegram.Bot(token=self.telegram_token)
            await bot.send_message(
                chat_id=self.telegram_chat_id,
                text=message,
                parse_mode='HTML'
            )
            
            self.signal_count += 1
            logger.info(f\"Signal sent for {symbol}: {signal} at {confidence}% confidence\")
            
        except Exception as e:
            logger.error(f\"Error sending Telegram message: {str(e)}\")

    async def scan_all_symbols(self):
        \"\"\"Scan all symbols for trading opportunities\"\"\"
        logger.info(\"Starting market scan...\")
        
        for symbol in self.symbols:
            try:
                # Fetch market data
                df = await self.fetch_market_data(symbol)
                if df.empty:
                    continue
                
                # Calculate indicators
                indicators = self.calculate_technical_indicators(df)
                if not indicators:
                    continue
                
                # Detect patterns
                patterns = self.detect_patterns(df)
                
                # Get AI analysis
                analysis = await self.analyze_with_openai(symbol, indicators, patterns)
                
                # Send alert if strong signal
                await self.send_telegram_alert(symbol, analysis, indicators)
                
                # Small delay to respect rate limits
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f\"Error processing {symbol}: {str(e)}\")
                continue
        
        logger.info(\"Market scan completed\")


    async def start_telegram_commands(self):
        \"\"\"Set up Telegram bot commands\"\"\"
        async def status_command(update, context):
            message = f\"\"\"
🤖 <b>Crypto Bot Status</b>

<b>Monitored Coins:</b> {len(self.symbols)}
<b>Active:</b> ✅ Running
<b>Last Scan:</b> {datetime.now().strftime('%H:%M UTC')}
<b>Signals Sent Today:</b> {self.signal_count}

<b>Coins:</b>
{', '.join([s.replace('/USDT', '') for s in self.symbols])}
            \"\"\"
            await update.message.reply_text(message, parse_mode='HTML')
        
        self.telegram_app.add_handler(CommandHandler("status", status_command))
        
        # Start the bot
        await self.telegram_app.initialize()
        await self.telegram_app.start()

    def schedule_scans(self):
        \"\"\"Schedule the scanning every 30 minutes\"\"\"
        schedule.every(30).minutes.do(lambda: asyncio.create_task(self.scan_all_symbols()))
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

async def main():
    \"\"\"Main function\"\"\"
    try:
        bot = CryptoTradingBot()
        
        # Start Telegram bot
        await bot.start_telegram_commands()
        
        # Initial scan
        await bot.scan_all_symbols()
        
        # Start scheduled scans
        logger.info(\"Bot started successfully! Scanning every 30 minutes...\")
        bot.schedule_scans()
        
    except Exception as e:
        logger.error(f\"Error in main: {str(e)}\")


if __name__ == \"__main__\":
    # Railway.app compatible
    port = int(os.environ.get(\"PORT\", 8080))
    
    # Run the bot
    asyncio.run(main())
