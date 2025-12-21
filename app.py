import base64
import io
from flask import Flask, request, render_template_string
import matplotlib
matplotlib.use('Agg') # Essential for web servers to prevent GUI errors
import matplotlib.pyplot as plt

# ==========================================
# 1. CORE LIBRARIES (EXACTLY AS PROVIDED)
# ==========================================
import numpy as np
import pandas as pd
import yfinance as yf
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import warnings
import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from collections import defaultdict
import time

try:
    import pandas_ta as ta
    from ta.trend import EMAIndicator
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ==========================================
# 2. THE UI (HTML & CSS)
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nifty 50 AI | Institutional Grade Predictor</title>
    <style>
        :root {
            --bg-main: #0a0e17;
            --bg-card: #151b2b;
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --accent-cyan: #00f0ff;
            --accent-blue: #3b82f6;
            --success: #10b981;
            --danger: #ef4444;
            --border: #2d3748;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--bg-main);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px 20px;
        }

        .container {
            width: 100%;
            max-width: 1200px;
        }

        /* HEADER */
        .header {
            text-align: center;
            margin-bottom: 40px;
            position: relative;
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #fff, var(--text-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            letter-spacing: -1px;
        }

        .header .badge {
            background: rgba(0, 240, 255, 0.1);
            color: var(--accent-cyan);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            border: 1px solid rgba(0, 240, 255, 0.2);
        }

        /* CARD STYLES */
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            margin-bottom: 30px;
        }

        /* FORM GRID */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 25px;
            align-items: end;
        }

        .input-group label {
            display: block;
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-bottom: 8px;
            font-weight: 500;
        }

        .input-group input, .input-group select {
            width: 100%;
            background: #0a0e17;
            border: 1px solid var(--border);
            color: #fff;
            padding: 14px;
            border-radius: 8px;
            font-size: 1rem;
            transition: all 0.3s ease;
        }

        .input-group input:focus, .input-group select:focus {
            outline: none;
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
        }

        button.generate-btn {
            background: linear-gradient(135deg, var(--accent-blue), #2563eb);
            color: white;
            border: none;
            padding: 14px 28px;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            transition: transform 0.2s;
            box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
        }

        button.generate-btn:hover {
            transform: translateY(-2px);
        }

        /* RESULTS DASHBOARD */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 250px 1fr;
            gap: 25px;
        }

        @media (max-width: 900px) {
            .dashboard-grid { grid-template-columns: 1fr; }
        }

        .stats-sidebar {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .stat-card {
            background: rgba(255,255,255,0.03);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.05);
        }

        .stat-label {
            color: var(--text-secondary);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }

        .stat-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #fff;
        }

        .sentiment-pos { color: var(--success); }
        .sentiment-neg { color: var(--danger); }
        .sentiment-neu { color: var(--text-secondary); }

        .chart-container {
            background: #000;
            border-radius: 12px;
            border: 1px solid var(--border);
            padding: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }

        /* DATA TABLE */
        .table-container {
            max-height: 300px;
            overflow-y: auto;
            margin-top: 20px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        th {
            text-align: left;
            padding: 12px;
            color: var(--accent-cyan);
            border-bottom: 1px solid var(--border);
            font-size: 0.9rem;
            position: sticky;
            top: 0;
            background: var(--bg-card);
        }

        td {
            padding: 12px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            font-family: 'Courier New', monospace;
        }

        /* LOADER */
        .loader-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(10, 14, 23, 0.9);
            z-index: 1000;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        
        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid rgba(59, 130, 246, 0.3);
            border-radius: 50%;
            border-top-color: var(--accent-blue);
            animation: spin 1s linear infinite;
            margin-bottom: 20px;
        }
        
        .loading-text {
            color: var(--accent-cyan);
            font-size: 1.2rem;
            animation: pulse 1.5s infinite;
        }

        @keyframes spin { 100% { transform: rotate(360deg); } }
        @keyframes pulse { 50% { opacity: 0.5; } }

    </style>
    <script>
        function startLoading() {
            document.getElementById('loader').style.display = 'flex';
        }
    </script>
</head>
<body>

    <div id="loader" class="loader-overlay">
        <div class="spinner"></div>
        <div class="loading-text">Analyzing Market Data & Training...</div>
        <div style="color: #666; margin-top:10px; font-size: 0.9rem;">(Your prediction is on the way! Don't refresh)</div>
    </div>

    <div class="container">
        <div class="header">
            <span class="badge">Nifty 50 Predictor</span>
            <h1>STOCKS-PRO</h1>
        </div>

        <div class="card">
            <form action="/" method="post" onsubmit="startLoading()">
                <div class="form-grid">
                    <div class="input-group">
                        <label>Target Date</label>
                        <input type="date" name="date" value="{{ date }}" required>
                    </div>
                    <div class="input-group">
                        <label>Start Time (HH:MM)</label>
                        <input type="time" name="time" value="{{ time }}" required>
                    </div>
                    <div class="input-group">
                        <label>Forecast Horizon</label>
                        <select name="mode">
                            <option value="30" {% if mode == '30' %}selected{% endif %}>Next 30 Minutes (Intraday)</option>
                            <option value="EOD" {% if mode == 'EOD' %}selected{% endif %}>End of Day (Close)</option>
                        </select>
                    </div>
                    <button type="submit" class="generate-btn">Predict</button>
                </div>
            </form>
        </div>

        {% if plot_url %}
        <div class="card">
            <div class="dashboard-grid">
                
                <div class="stats-sidebar">
                    <div class="stat-card">
                        <div class="stat-label">Sentiment Signal</div>
                        <div class="stat-value {{ sentiment_class }}">{{ sentiment_text }}</div>
                        <div style="font-size: 0.8rem; color: #666; margin-top: 5px;">Score: {{ sentiment_score }}</div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-label">Predicted Avg Price</div>
                        <div class="stat-value">₹{{ avg_price }}</div>
                    </div>

                    {% if metrics %}
                    <div class="stat-card" style="border-color: var(--accent-blue);">
                        <div class="stat-label">Model Accuracy</div>
                        <div class="stat-value" style="color: var(--accent-blue);">{{ metrics.acc }}%</div>
                        <div style="font-size: 0.8rem; color: #666; margin-top: 5px;">MAE: {{ metrics.mae }}</div>
                    </div>
                    {% endif %}
                </div>

                <div class="chart-container">
                    <img src="data:image/png;base64,{{ plot_url }}" alt="Prediction Chart">
                </div>
            </div>

            {% if predictions %}
            <div style="margin-top: 30px;">
                <h3 style="color: var(--text-primary); margin-bottom: 15px;">Predicted Price Action</h3>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Timestamp</th>
                                <th>Predicted Close (₹)</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for row in predictions %}
                            <tr>
                                <td>{{ row.time }}</td>
                                <td style="color: var(--accent-cyan);">{{ row.price }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
            {% endif %}
        </div>
        {% endif %}

        {% if error %}
        <div class="card" style="border-left: 4px solid var(--danger);">
            <h3 style="color: var(--danger); margin-bottom: 10px;">System Error</h3>
            <p style="color: var(--text-secondary);">{{ error }}</p>
        </div>
        {% endif %}

    </div>
</body>
</html>
"""

# ==========================================
# 3. BACKEND LOGIC (PRESERVED EXACTLY)
# ==========================================

def adjust_to_last_weekday(date_obj):
    while date_obj.weekday() > 4:
        date_obj -= timedelta(days=1)
    return date_obj

class MarketSentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.news_cache = {}
        
    def clean_text(self, text):
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def get_sentiment_vader(self, text):
        scores = self.vader.polarity_scores(text)
        return scores['compound']
    
    def get_sentiment_textblob(self, text):
        blob = TextBlob(text)
        return blob.sentiment.polarity
    
    def classify_sentiment(self, score):
        if score > 0.1: return 1
        elif score < -0.1: return -1
        else: return 0
    
    def fetch_news_headlines(self, date_str, keywords=["nifty", "sensex", "nse", "indian stock market", "indian economy"]):
        api_key = "a2e3b4e6c2c147ad9308fd202b927fcd"
        headlines = []
        try:
            now = datetime.utcnow()
            from_time = (now - timedelta(hours=3)).strftime('%Y-%m-%dT%H:%M:%SZ')
            to_time = now.strftime('%Y-%m-%dT%H:%M:%SZ')
            query = ' OR '.join(keywords)
            url = f"https://newsapi.org/v2/everything?q={query}&from={from_time}&to={to_time}&language=en&sortBy=publishedAt&apiKey={api_key}"
            
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                if articles:
                    for i, article in enumerate(articles[:10], start=1):
                        title = article.get('title', '')
                        description = article.get('description', '')
                        combined = f"{title}. {description}".strip()
                        if combined: headlines.append(combined)
            
            if not headlines:
                headlines = ["No significant market movement reported.", "Steady market sentiment observed."]
        except Exception:
            headlines = ["No significant market movement reported.", "Steady market sentiment observed."]
        return headlines

    def fetch_reddit_posts(self, subreddits=['IndianStockMarket', 'IndiaInvestments'], limit=50):
        posts = []
        for subreddit in subreddits:
            try:
                url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
                headers = {'User-Agent': 'Mozilla/5.0 (compatible; StockPredictorBot/1.0)'}
                response = requests.get(url, headers=headers, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    items = data.get('data', {}).get('children', [])
                    if items:
                        for post in items[:10]:
                            title = post['data'].get('title', '')
                            body = post['data'].get('selftext', '')
                            combined = f"{title}. {body}".strip()
                            if len(combined) > 10: posts.append(combined)
                time.sleep(0.5)
            except Exception: continue

        if not posts:
            posts = ["No recent discussions found on Reddit.", "Market sentiment appears steady."]
        return posts
    
    def analyze_market_sentiment(self, date_str, current_time):
        all_texts = []
        # Maintaining exact logic: Fetch news + Reddit (Twitter was commented out in provided text)
        headlines = self.fetch_news_headlines(date_str)
        reddit_posts = self.fetch_reddit_posts()
        
        all_texts.extend(headlines)
        all_texts.extend(reddit_posts[:20])
        
        sentiments = []
        sentiment_scores = []
        
        for text in all_texts:
            if text and len(text) > 10:
                clean_text = self.clean_text(text)
                vader_score = self.get_sentiment_vader(clean_text)
                blob_score = self.get_sentiment_textblob(clean_text)
                avg_score = (vader_score + blob_score) / 2
                sentiment_scores.append(avg_score)
                sentiments.append(self.classify_sentiment(avg_score))
        
        if sentiments:
            positive_pct = sentiments.count(1) / len(sentiments)
            negative_pct = sentiments.count(-1) / len(sentiments)
            neutral_pct = sentiments.count(0) / len(sentiments)
            avg_sentiment = np.mean(sentiment_scores)
            
            return {
                'sentiment_positive_pct': positive_pct,
                'sentiment_negative_pct': negative_pct,
                'sentiment_neutral_pct': neutral_pct,
                'sentiment_avg_score': avg_sentiment,
                'sentiment_std': np.std(sentiment_scores),
                'sentiment_signal': 1 if avg_sentiment > 0.1 else (-1 if avg_sentiment < -0.1 else 0)
            }
        else:
            return {
                'sentiment_positive_pct': 0.33, 'sentiment_negative_pct': 0.33,
                'sentiment_neutral_pct': 0.34, 'sentiment_avg_score': 0.0,
                'sentiment_std': 0.0, 'sentiment_signal': 0
            }

class IntradayNiftyPredictor:
    def __init__(self, symbol="^NSEI", lookback_days=365, fft_window=75):
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.fft_window = fft_window
        self.model = None
        self.history_df = None
        self.feature_columns = None
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        self.sentiment_features = None
        
        # EXACT PARAMS FROM PROVIDED CODE
        self.lgb_params = {
            'n_estimators': 1000,   # You explicitly asked to maintain accuracy/logic
            'learning_rate': 0.05,
            'num_leaves': 64,
            'objective': 'regression',
            'random_state': 42,
            'verbose': -1,
            'n_jobs': 1 # Added only to prevent web server crash, does not affect accuracy
        }

    def load_historical(self, date_str):
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        end = dt - timedelta(days=1)
        start = end - timedelta(days=self.lookback_days)
        dfh = yf.Ticker(self.symbol).history(start=start, end=end, interval="1h")[['Open', 'High', 'Low', 'Close', 'Volume']]
        if dfh.empty:
            dfh = yf.Ticker(self.symbol).history(start=start, end=end, interval="1d")[['Open', 'High', 'Low', 'Close', 'Volume']]
        if dfh.index.tz is not None: dfh.index = dfh.index.tz_localize(None)

        ohlc = dfh[['Open', 'High', 'Low', 'Close']].resample('5T').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
        vol = dfh['Volume'].resample('5T').mean()
        df5 = pd.concat([ohlc, vol], axis=1).fillna(method='ffill')
        df5 = df5.between_time('09:15', '15:30').dropna()

        df5['vwap_12'] = (df5['Close'] * df5['Volume']).rolling(12).sum() / df5['Volume'].rolling(12).sum()
        df5['atr_14'] = (df5['High'] - df5['Low']).rolling(14).mean()
        df5[['vwap_12', 'atr_14']] = df5[['vwap_12', 'atr_14']].shift(1)
        self.history_df = df5.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'})

    def engineer_features(self, df, include_sentiment=True):
        base = df.copy()
        if TA_AVAILABLE:
            ema200 = EMAIndicator(base['close'], window=200).ema_indicator()
            rsi14 = RSIIndicator(base['close'], window=14).rsi()
            bb = BollingerBands(base['close'], window=20, window_dev=2)
            bb_mid = bb.bollinger_mavg()
            bb_up = bb.bollinger_hband()
            bb_lo = bb.bollinger_lband()
        else:
            ema200 = base['close'].ewm(span=200, adjust=False).mean()
            delta = base['close'].diff()
            gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
            loss = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi14 = 100 - (100 / (1 + rs))
            bb_mid = base['close'].rolling(20).mean()
            bb_std = base['close'].rolling(20).std()
            bb_up = bb_mid + 2 * bb_std
            bb_lo = bb_mid - 2 * bb_std

        feat = pd.DataFrame(index=base.index)
        feat['close'] = base['close']
        feat['ema200'] = ema200.shift(1)
        feat['ema_dist_pct'] = ((base['close'] / ema200) - 1.0).mul(100).shift(1)
        feat['rsi14'] = rsi14.shift(1)
        feat['rsi_slope_3'] = rsi14.diff().rolling(3).mean().shift(1)
        feat['bb_mid'] = bb_mid.shift(1)
        feat['bb_upper'] = bb_up.shift(1)
        feat['bb_lower'] = bb_lo.shift(1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            width_pct = (bb_up - bb_lo) / bb_mid.replace(0, np.nan)
            percent_b = (base['close'] - bb_lo) / (bb_up - bb_lo)
        feat['bb_width_pct'] = width_pct.replace([np.inf, -np.inf], np.nan).shift(1)
        feat['bb_percent_b'] = percent_b.replace([np.inf, -np.inf], np.nan).shift(1)

        if include_sentiment and self.sentiment_features is not None:
            for key, value in self.sentiment_features.items():
                feat[key] = value

        feat = feat.fillna(method='ffill').fillna(method='bfill')
        feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0)
        for col in feat.columns:
            if col != 'close':
                feat[col] = pd.to_numeric(feat[col], errors='coerce').fillna(0)
        return feat

    def train(self, date_str, current_time):
        self.sentiment_features = self.sentiment_analyzer.analyze_market_sentiment(date_str, current_time)
        feat = self.engineer_features(self.history_df)
        feature_cols = [col for col in feat.columns if col != 'close']
        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(feat[col]):
                feat[col] = pd.to_numeric(feat[col], errors='coerce').fillna(0)
        X = feat[feature_cols]
        y = feat['close']
        valid_idx = ~(y.isna() | X.isna().any(axis=1))
        X = X[valid_idx]
        y = y[valid_idx]
        self.feature_columns = feature_cols
        if len(y) < 10: raise RuntimeError("Not enough valid data.")
        self.model = LGBMRegressor(**self.lgb_params)
        self.model.fit(X, y)

    def predict(self, date_str, current_time, mode='30'):
        if self.history_df is None: self.load_historical(date_str)
        if self.model is None: self.train(date_str, current_time)
        dt_cur = datetime.strptime(f"{date_str} {current_time}", "%Y-%m-%d %H:%M")
        df_hist = self.history_df.copy()
        
        try:
            dfm = yf.Ticker(self.symbol).history(start=dt_cur - timedelta(minutes=120), end=dt_cur, interval='1m')
            if not dfm.empty:
                if dfm.index.tz is not None: dfm.index = dfm.index.tz_localize(None)
                df5m = dfm['Close'].resample('5T').ohlc()
                df5m.columns = ['open', 'high', 'low', 'close']
                df5m['Volume'] = dfm['Volume'].resample('5T').sum()
                df5m = df5m.between_time('09:15', '15:30').dropna()
                df_hist = pd.concat([self.history_df, df5m], axis=0)
                df_hist = df_hist[~df_hist.index.duplicated(keep='last')]
        except: pass

        df_curr = df_hist.copy()
        preds, times = [], []

        if mode == '30':
            steps = max(1, 30 // 5)
            end_time = dt_cur + timedelta(minutes=30)
        else:
            market_close = dt_cur.replace(hour=15, minute=30)
            if dt_cur >= market_close: return pd.DataFrame({'timestamp': [], 'predicted_close': []})
            steps = int(np.ceil((market_close - dt_cur).total_seconds() / 300))
        
        for i in range(1, steps + 1):
            t = dt_cur + timedelta(minutes=5 * i)
            if t.hour < 9 or (t.hour == 9 and t.minute < 15) or t.hour > 15 or (t.hour == 15 and t.minute > 30): continue
            try:
                feat = self.engineer_features(df_curr)
                feature_cols = self.feature_columns if self.feature_columns else [c for c in feat.columns if c != 'close']
                for col in feature_cols: 
                    if col not in feat.columns: feat[col] = 0
                last = feat[feature_cols].iloc[-1].values.reshape(1, -1)
                p = self.model.predict(last)[0]
                preds.append(p)
                times.append(t)
                df_curr.loc[t] = {
                    'open': p, 'high': p * 1.001, 'low': p * 0.999, 'close': p,
                    'Volume': df_curr['Volume'].iloc[-1] if len(df_curr) > 0 else 1000000,
                    'vwap_12': df_curr['vwap_12'].iloc[-1] if 'vwap_12' in df_curr.columns else p,
                    'atr_14': df_curr['atr_14'].iloc[-1] if 'atr_14' in df_curr.columns else p * 0.01
                }
            except Exception:
                last_price = preds[-1] if preds else df_curr['close'].iloc[-1]
                preds.append(last_price)
                times.append(t)
        return pd.DataFrame({'timestamp': times, 'predicted_close': preds})

# ==========================================
# 4. ROUTING
# ==========================================

@app.route('/', methods=['GET', 'POST'])
def home():
    # Set default date to today, handle weekends later in logic
    default_date = datetime.today().strftime('%Y-%m-%d')
    default_time = "10:15"
    
    if request.method == 'GET':
        return render_template_string(HTML_TEMPLATE, date=default_date, time=default_time, mode="30")
    
    try:
        date = request.form['date']
        current_time = request.form['time']
        mode = request.form['mode']

        # Handling Weekend Logic:
        # If user selects Sat/Sun, we shift input date to Friday to prevent Yahoo errors.
        input_dt = datetime.strptime(date, "%Y-%m-%d")
        if input_dt.weekday() > 4:
            input_dt = adjust_to_last_weekday(input_dt)
            date = input_dt.strftime('%Y-%m-%d')

        m = IntradayNiftyPredictor(lookback_days=365, fft_window=75)
        m.load_historical(date)
        m.train(date, current_time)
        df_pred = m.predict(date, current_time, mode=mode)

        # Plot Generation
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 6))
        
        avg_price = 0
        if len(df_pred) > 0:
            # Main prediction line
            ax.plot(df_pred['timestamp'], df_pred['predicted_close'], linewidth=3, color='#00f0ff', label='Forecast Model', zorder=5)
            # Start point
            ax.scatter(df_pred['timestamp'].iloc[0], df_pred['predicted_close'].iloc[0], s=100, color='#ffff00', edgecolors='black', label='Analysis Start', zorder=10)
            avg_price = df_pred['predicted_close'].mean()
        
        ax.set_facecolor('#000000')
        fig.patch.set_facecolor('#000000')
        ax.grid(True, linestyle='--', alpha=0.15, color='white')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#333')
        ax.spines['left'].set_color('#333')
        ax.tick_params(colors='#888')
        plt.legend(facecolor='#111', edgecolor='#333', labelcolor='white')
        plt.tight_layout()

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)

        # Accuracy Metrics
        metrics = None
        dt_cur = datetime.strptime(f"{date} {current_time}", "%Y-%m-%d %H:%M")
        
        # Define end time for actuals fetch
        if mode == '30':
            end_act = dt_cur + timedelta(minutes=30)
        else:
            end_act = dt_cur.replace(hour=15, minute=30)
            
        df_act = yf.Ticker(m.symbol).history(start=dt_cur, end=end_act + timedelta(minutes=1), interval='5m')
        
        if not df_act.empty and len(df_pred) > 0:
            # Align lengths
            min_len = min(len(df_act), len(df_pred))
            actuals = df_act['Close'].values[:min_len]
            preds = df_pred['predicted_close'].values[:min_len]
            
            if min_len > 0:
                mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
                metrics = {
                    'mse': f"{mean_squared_error(actuals, preds):.2f}",
                    'mae': f"{mean_absolute_error(actuals, preds):.2f}",
                    'acc': f"{100 - mape:.2f}"
                }

        predictions_list = []
        if len(df_pred) > 0:
            for _, row in df_pred.iterrows():
                predictions_list.append({
                    'time': row['timestamp'].strftime('%H:%M'),
                    'price': f"{row['predicted_close']:.2f}"
                })

        sentiment_signal = m.sentiment_features['sentiment_signal']
        sentiment_text = "Bullish (Positive)" if sentiment_signal > 0 else "Bearish (Negative)" if sentiment_signal < 0 else "Neutral"
        sentiment_class = "sentiment-pos" if sentiment_signal > 0 else "sentiment-neg" if sentiment_signal < 0 else "sentiment-neu"

        return render_template_string(HTML_TEMPLATE, 
                                      date=date, 
                                      time=current_time, 
                                      mode=mode,
                                      plot_url=plot_url,
                                      sentiment_score=f"{m.sentiment_features['sentiment_avg_score']:.4f}",
                                      sentiment_text=sentiment_text,
                                      sentiment_class=sentiment_class,
                                      avg_price=f"{avg_price:.2f}",
                                      metrics=metrics,
                                      predictions=predictions_list)

    except Exception as e:
        return render_template_string(HTML_TEMPLATE, date=request.form['date'], time=request.form['time'], mode=request.form['mode'], error=str(e))

if __name__ == '__main__':
    print("Initializing AI Core...")
    app.run(debug=True, port=5000)
