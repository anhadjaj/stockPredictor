import base64
import io
import time
import warnings
import re
from datetime import datetime, timedelta

# Core Data Science
import numpy as np
import pandas as pd
import yfinance as yf
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Visualization & Web
from flask import Flask, request, render_template_string
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Sentiment & Analysis
import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    import pandas_ta as ta
    from ta.trend import EMAIndicator
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("Warning: 'pandas_ta' library not found. Using manual calculation fallback.")

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ==========================================
# 1. UI TEMPLATE (Improved Dark Theme)
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nifty 50 AI | Institutional Analytics</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            /* Fintech Dark Palette */
            --bg-body: #050505;
            --bg-panel: #0e0e0e;
            --bg-card: #141414;
            
            --border-subtle: #27272a;
            --border-active: #3f3f46;
            
            --text-main: #e4e4e7;
            --text-muted: #a1a1aa;
            
            /* Accents */
            --accent-primary: #3b82f6; /* Blue */
            --accent-glow: rgba(59, 130, 246, 0.15);
            --signal-green: #10b981;
            --signal-red: #ef4444;
            --signal-neutral: #fbbf24;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-body);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
            background-image: 
                radial-gradient(circle at 50% 0%, #1e1e24 0%, transparent 40%),
                linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
            background-size: 100% 100%, 40px 40px, 40px 40px;
        }

        .container {
            width: 100%;
            max-width: 1400px;
            animation: fadeIn 0.6s ease-out;
        }

        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

        /* --- HEADER --- */
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding: 20px 0;
            border-bottom: 1px solid var(--border-subtle);
        }

        .brand {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: -1px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .brand span { color: var(--accent-primary); }
        .brand::before {
            content: '';
            width: 12px;
            height: 12px;
            background: var(--signal-green);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--signal-green);
        }

        .status-pill {
            font-size: 0.75rem;
            padding: 6px 12px;
            border-radius: 100px;
            background: var(--border-subtle);
            color: var(--text-muted);
            border: 1px solid var(--border-active);
            font-family: 'JetBrains Mono', monospace;
        }

        /* --- CONTROL PANEL (FORM) --- */
        .control-panel {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 25px;
            box-shadow: 0 20px 40px -10px rgba(0,0,0,0.5);
        }

        .form-row {
            display: flex;
            gap: 20px;
            align-items: flex-end;
        }

        .input-group {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .input-group label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            font-weight: 600;
        }

        .input-group input, .input-group select {
            background: var(--bg-panel);
            border: 1px solid var(--border-subtle);
            color: white;
            padding: 12px 16px;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.95rem;
            transition: all 0.2s;
            outline: none;
        }

        .input-group input:focus, .input-group select:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }

        .btn-predict {
            background: var(--accent-primary);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            font-size: 0.95rem;
            height: 46px; /* Match input height */
            transition: all 0.2s;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        }

        .btn-predict:hover {
            transform: translateY(-1px);
            filter: brightness(1.1);
        }

        /* --- DASHBOARD GRID --- */
        .dashboard {
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 25px;
        }

        @media (max-width: 900px) {
            .dashboard { grid-template-columns: 1fr; }
            .form-row { flex-direction: column; }
            .btn-predict { width: 100%; }
        }

        /* --- METRICS SIDEBAR --- */
        .metrics-col {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .metric-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 20px;
            position: relative;
            overflow: hidden;
        }

        .metric-card::after {
            content: '';
            position: absolute;
            top: 0; left: 0; width: 4px; height: 100%;
            background: var(--border-active);
        }

        .metric-card.bullish::after { background: var(--signal-green); }
        .metric-card.bearish::after { background: var(--signal-red); }
        .metric-card.primary::after { background: var(--accent-primary); }

        .metric-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
        }

        .metric-value {
            font-size: 1.75rem;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            color: #fff;
        }

        .metric-sub {
            font-size: 0.8rem;
            margin-top: 5px;
            opacity: 0.7;
        }

        .text-green { color: var(--signal-green); }
        .text-red { color: var(--signal-red); }
        .text-blue { color: var(--accent-primary); }

        /* --- MAIN CHART AREA --- */
        .chart-panel {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 20px;
            min-height: 400px;
            display: flex;
            flex-direction: column;
        }

        .panel-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid var(--border-subtle);
        }

        .panel-title {
            font-weight: 600;
            color: var(--text-main);
        }

        .chart-img-container {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #000;
            border-radius: 8px;
            border: 1px solid var(--border-subtle);
            padding: 10px;
        }
        
        .chart-img-container img {
            width: 100%;
            height: auto;
            max-height: 450px;
            object-fit: contain;
        }

        /* --- DATA TABLE --- */
        .table-container {
            margin-top: 25px;
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            overflow: hidden;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
        }

        th {
            background: var(--bg-panel);
            text-align: left;
            padding: 15px 20px;
            color: var(--text-muted);
            font-weight: 500;
            font-size: 0.8rem;
            text-transform: uppercase;
        }

        td {
            padding: 15px 20px;
            border-bottom: 1px solid var(--border-subtle);
            color: var(--text-main);
        }

        tr:last-child td { border-bottom: none; }
        tr:hover td { background: rgba(255,255,255,0.02); }

        .price-up { color: var(--signal-green); }
        
        /* --- LOADER --- */
        .loader-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.85);
            backdrop-filter: blur(5px);
            z-index: 999;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }

        .loader-bar {
            width: 300px;
            height: 4px;
            background: var(--border-active);
            border-radius: 2px;
            overflow: hidden;
            margin-top: 20px;
        }

        .loader-progress {
            height: 100%;
            width: 0%;
            background: var(--accent-primary);
            box-shadow: 0 0 10px var(--accent-primary);
            animation: load 2s ease-in-out infinite;
        }

        .loader-text {
            font-family: 'JetBrains Mono', monospace;
            color: var(--accent-primary);
            margin-top: 15px;
            font-size: 0.9rem;
        }

        @keyframes load { 0% { width: 0%; transform: translateX(-50%); } 100% { width: 100%; transform: translateX(0); } }

    </style>
    <script>
        function startLoading() {
            document.getElementById('loader').style.display = 'flex';
        }
    </script>
</head>
<body>

    <div id="loader" class="loader-overlay">
        <div style="font-size: 2rem; font-weight: 700; letter-spacing: -1px;">INITIALIZING <span style="color:var(--accent-primary)">AI MODELS</span></div>
        <div class="loader-bar"><div class="loader-progress"></div></div>
        <div class="loader-text">> Processing Market Microstructure...</div>
    </div>

    <div class="container">
        <nav class="navbar">
            <div class="brand">NIFTY<span>PRO</span>.AI</div>
            <div class="status-pill">● SYSTEM ONLINE</div>
        </nav>

        <div class="control-panel">
            <form action="/" method="post" onsubmit="startLoading()">
                <div class="form-row">
                    <div class="input-group">
                        <label>Target Date</label>
                        <input type="date" name="date" value="{{ date }}" required>
                    </div>
                    <div class="input-group">
                        <label>Execution Time</label>
                        <input type="time" name="time" value="{{ time }}" required>
                    </div>
                    <div class="input-group">
                        <label>Strategy Horizon</label>
                        <select name="mode">
                            <option value="30" {% if mode == '30' %}selected{% endif %}>Intraday (30m Scalp)</option>
                            <option value="EOD" {% if mode == 'EOD' %}selected{% endif %}>Intraday (End of Day)</option>
                        </select>
                    </div>
                    <button type="submit" class="btn-predict">RUN ANALYSIS</button>
                </div>
            </form>
        </div>

        {% if error %}
        <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid var(--signal-red); color: var(--signal-red); padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <strong>SYSTEM ERROR:</strong> {{ error }}
        </div>
        {% endif %}

        {% if plot_url %}
        <div class="dashboard">
            <div class="metrics-col">
                <div class="metric-card {% if sentiment_class == 'sentiment-pos' %}bullish{% elif sentiment_class == 'sentiment-neg' %}bearish{% else %}primary{% endif %}">
                    <div class="metric-label">Sentiment Analysis</div>
                    <div class="metric-value {% if sentiment_class == 'sentiment-pos' %}text-green{% elif sentiment_class == 'sentiment-neg' %}text-red{% endif %}">
                        {{ sentiment_text }}
                    </div>
                    <div class="metric-sub">Score: {{ sentiment_score }}</div>
                </div>

                <div class="metric-card primary">
                    <div class="metric-label">Target Avg Price</div>
                    <div class="metric-value">₹{{ avg_price }}</div>
                </div>

                {% if metrics %}
                <div class="metric-card primary">
                    <div class="metric-label">Model Confidence</div>
                    <div class="metric-value text-blue">{{ metrics.acc }}%</div>
                    <div class="metric-sub">Mean Abs Error: {{ metrics.mae }}</div>
                </div>
                {% endif %}
            </div>

            <div class="chart-panel">
                <div class="panel-header">
                    <div class="panel-title">Price Action Forecast</div>
                    <div style="font-size: 0.8rem; color: var(--text-muted);">LGBM REGRESSOR v2.4</div>
                </div>
                <div class="chart-img-container">
                    <img src="data:image/png;base64,{{ plot_url }}" alt="Prediction Chart">
                </div>
            </div>
        </div>

        {% if predictions %}
        <div class="table-container">
             <table>
                <thead>
                    <tr>
                        <th width="30%">Timestamp</th>
                        <th width="40%">Predicted Close</th>
                        <th width="30%">Status</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in predictions %}
                    <tr>
                        <td>{{ row.time }}</td>
                        <td class="price-up">₹{{ row.price }}</td>
                        <td><span style="color: var(--text-muted);">PREDICTED</span></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        {% endif %}
    </div>
</body>
</html>
"""

# ==========================================
# 2. SENTIMENT ENGINE
# ==========================================
class MarketSentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
    
    def clean_text(self, text):
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()
    
    def fetch_data(self):
        # 1. NewsAPI (Headlines)
        headlines = []
        try:
            api_key = "a2e3b4e6c2c147ad9308fd202b927fcd" # Free tier key, might need replacement if rate limited
            url = f"https://newsapi.org/v2/everything?q=nifty OR sensex OR indian economy&sortBy=publishedAt&language=en&apiKey={api_key}"
            resp = requests.get(url, timeout=3).json()
            headlines = [a['title'] for a in resp.get('articles', [])[:10]]
        except: pass

        # 2. Reddit (Retail Sentiment)
        reddit_texts = []
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            url = "https://www.reddit.com/r/IndianStockMarket/hot.json?limit=15"
            resp = requests.get(url, headers=headers, timeout=3).json()
            for post in resp['data']['children']:
                reddit_texts.append(post['data']['title'])
        except: pass
        
        return headlines + reddit_texts

    def analyze(self):
        texts = self.fetch_data()
        if not texts: return 0.0, 0.0 # Default neutral

        scores = []
        for t in texts:
            clean = self.clean_text(t)
            vs = self.vader.polarity_scores(clean)['compound']
            tb = TextBlob(clean).sentiment.polarity
            scores.append((vs + tb) / 2) # Hybrid Score
        
        avg_score = np.mean(scores)
        return avg_score, np.std(scores)

# ==========================================
# 3. PREDICTION ENGINE (IMPROVED)
# ==========================================
class AdvancedNiftyPredictor:
    def __init__(self, symbol="^NSEI"):
        self.symbol = symbol
        self.model = None
        self.feature_cols = None
        # Tuned Parameters for Noise Reduction
        self.params = {
            'n_estimators': 800,
            'learning_rate': 0.03,
            'num_leaves': 40,
            'max_depth': 8,
            'objective': 'regression',
            'random_state': 42,
            'n_jobs': 1,
            'verbose': -1
        }

    def fetch_data(self, date_str):
        # Adjust date if weekend
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        while dt.weekday() > 4: dt -= timedelta(days=1)
        
        # Get 1 year of context for training
        start_date = dt - timedelta(days=365)
        end_date = dt + timedelta(days=1) # Include target day for intraday fetching
        
        # 1. Daily Data (Context)
        df_daily = yf.Ticker(self.symbol).history(start=start_date, end=end_date, interval="1d")
        
        # 2. Intraday Data (Recent Pattern) - Yahoo allows last 60d for 5m interval
        df_intra = yf.Ticker(self.symbol).history(period="60d", interval="5m")
        
        if df_intra.empty: raise ValueError("Market data unavailable from source.")
        
        # Cleanup
        if df_intra.index.tz is not None: df_intra.index = df_intra.index.tz_localize(None)
        df_intra = df_intra.rename(columns={'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Volume':'volume'})
        
        # Filter Market Hours
        df_intra = df_intra.between_time('09:15', '15:30')
        return df_intra

    def engineer_features(self, df):
        df = df.copy()
        
        # --- A. TARGET TRANSFORMATION (Log Returns) ---
        # We predict movement (%), not price ($)
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        
        # --- B. LAG FEATURES (Temporal Memory) ---
        # The most important features for intraday
        for lag in [1, 2, 3, 5, 10]:
            df[f'ret_lag_{lag}'] = df['log_ret'].shift(lag)
            df[f'vol_lag_{lag}'] = df['volume'].shift(lag)
        
        # --- C. TIME CYCLICAL ENCODING ---
        # Teach model that 15:30 is "end of day"
        minutes = df.index.hour * 60 + df.index.minute
        df['time_sin'] = np.sin(2 * np.pi * minutes / 1440)
        df['time_cos'] = np.cos(2 * np.pi * minutes / 1440)
        
        # --- D. TECHNICAL INDICATORS ---
        if TA_AVAILABLE:
            # RSI
            df['rsi'] = RSIIndicator(df['close'], window=14).rsi().shift(1)
            # EMA Distance
            ema = EMAIndicator(df['close'], window=50).ema_indicator()
            df['ema_dist'] = (df['close'] - ema) / ema
            # Bollinger Width
            bb = BollingerBands(df['close'], window=20)
            df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        else:
            # Fallback (Manual Calc)
            df['rsi'] = 50 
            df['ema_dist'] = 0
            df['bb_width'] = 0

        return df.dropna()

    def train_and_predict(self, date_str, start_time, mode):
        # 1. Data Prep
        raw_df = self.fetch_data(date_str)
        
        # 2. Feature Engineering (for training)
        full_df = self.engineer_features(raw_df)
        
        # 3. Create Training Set
        X = full_df.drop(['open','high','low','close','volume','log_ret'], axis=1)
        y = full_df['log_ret'].shift(-1) # Target: Next Candle's Return
        
        # Align indexes
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X_train, y_train = X[valid_idx], y[valid_idx]
        
        # 4. Train Model
        self.model = LGBMRegressor(**self.params)
        self.model.fit(X_train, y_train)
        self.feature_cols = X_train.columns.tolist()
        
        # 5. Recursive Prediction Loop
        target_dt = datetime.strptime(f"{date_str} {start_time}", "%Y-%m-%d %H:%M")
        
        # Slice data up to the target time
        if target_dt < raw_df.index[-1]:
            current_state_df = raw_df[raw_df.index <= target_dt].copy()
        else:
            current_state_df = raw_df.copy()

        # --- FIX START: Calculate features FIRST so 'log_ret' exists ---
        current_state_df = self.engineer_features(current_state_df)
        
        # Calculate recent volatility (Standard Deviation of returns)
        recent_volatility = current_state_df['log_ret'].tail(20).std()
        
        # Fallback if volatility is NaN or 0 (e.g., flat market or not enough data)
        if np.isnan(recent_volatility) or recent_volatility == 0:
            recent_volatility = 0.0005 
        # --- FIX END ---
            
        preds = []
        steps = 6 if mode == '30' else 20 
        
        current_price = current_state_df['close'].iloc[-1]
        
        for i in range(steps):
            # Recalculate features based on updated history
            feat_df = self.engineer_features(current_state_df)
            last_row = feat_df[self.feature_cols].iloc[-1:].values
            
            # Predict Trend
            pred_log_ret = self.model.predict(last_row)[0]
            
            # --- Inject Noise (Monte Carlo) ---
            noise = np.random.normal(0, recent_volatility)
            final_log_ret = pred_log_ret + noise
            
            # Convert to Price
            next_price = current_price * np.exp(final_log_ret)
            
            # Timestamp
            next_time = current_state_df.index[-1] + timedelta(minutes=5)
            preds.append({'timestamp': next_time, 'price': next_price})
            
            # --- Simulate OHLC Candle for next step ---
            # Create a realistic high/low range so indicators (like Bollinger Bands) stay alive
            sim_high = next_price * (1 + (recent_volatility/2))
            sim_low = next_price * (1 - (recent_volatility/2))

            new_row = pd.DataFrame({
                'open': [next_price], 
                'high': [sim_high], 
                'low': [sim_low], 
                'close': [next_price], 
                'volume': [current_state_df['volume'].mean()],
                # Important: Calculate log_ret for the new row immediately
                'log_ret': [final_log_ret] 
            }, index=[next_time])
            
            # Append to history
            current_state_df = pd.concat([current_state_df, new_row])
            
            # Ensure we don't carry over NaNs in the new row
            current_state_df['log_ret'] = current_state_df['log_ret'].fillna(0)
            
            current_price = next_price

        return pd.DataFrame(preds), raw_df

# ==========================================
# 4. ROUTE HANDLER
# ==========================================
@app.route('/', methods=['GET', 'POST'])
def home():
    # Defaults
    def_date = datetime.now().strftime('%Y-%m-%d')
    def_time = "09:30"
    
    if request.method == 'GET':
        return render_template_string(HTML_TEMPLATE, date=def_date, time=def_time, mode='30')

    try:
        # Inputs
        date = request.form['date']
        time_input = request.form['time']
        mode = request.form['mode']
        
        # 1. Sentiment Analysis
        sent_analyzer = MarketSentimentAnalyzer()
        sent_score, sent_std = sent_analyzer.analyze()
        
        sent_class = "sentiment-pos" if sent_score > 0.05 else "sentiment-neg" if sent_score < -0.05 else "sentiment-neu"
        sent_text = "Bullish" if sent_score > 0.05 else "Bearish" if sent_score < -0.05 else "Neutral"

        # 2. Run Prediction
        predictor = AdvancedNiftyPredictor()
        pred_df, hist_df = predictor.train_and_predict(date, time_input, mode)

        # 3. Calculate Accuracy (If past data exists)
        metrics = None
        if not pred_df.empty:
            # Check if we have actual data for these timestamps
            act_prices = []
            pred_prices = []
            
            for _, row in pred_df.iterrows():
                try:
                    # Look for exact timestamp match in history (if user backtesting)
                    act = hist_df.loc[row['timestamp']]['close']
                    act_prices.append(act)
                    pred_prices.append(row['price'])
                except KeyError:
                    pass # Future data, can't validate
            
            if len(act_prices) > 2:
                mae = mean_absolute_error(act_prices, pred_prices)
                mape = np.mean(np.abs(np.array(act_prices) - np.array(pred_prices)) / np.array(act_prices)) * 100
                metrics = {'mae': f"{mae:.2f}", 'acc': f"{100-mape:.2f}"}

        # 4. Plotting
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot Context (Last 2 hours)
        context_data = hist_df.tail(24) # Last 24 candles (2 hrs)
        ax.plot(context_data.index, context_data['close'], color='#555', label='Historical')
        
        # Plot Prediction
        if not pred_df.empty:
            ax.plot(pred_df['timestamp'], pred_df['price'], color='#00f0ff', linewidth=2.5, marker='o', markersize=4, label='AI Forecast')
            # Connect the gap
            ax.plot([context_data.index[-1], pred_df['timestamp'].iloc[0]], 
                    [context_data['close'].iloc[-1], pred_df['price'].iloc[0]], 
                    color='#00f0ff', linestyle='--')

        # Styling
        ax.set_facecolor('#000')
        fig.patch.set_facecolor('#000')
        ax.grid(color='#333', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.tick_params(axis='x', colors='#888')
        ax.tick_params(axis='y', colors='#888')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend(facecolor='#111', labelcolor='#ccc', edgecolor='#333')
        plt.tight_layout()

        # Save Plot
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)

        # 5. Format Output
        formatted_preds = []
        prev_p = context_data['close'].iloc[-1]
        for _, row in pred_df.iterrows():
            curr = row['price']
            trend_arrow = "▲" if curr > prev_p else "▼"
            color = "#10b981" if curr > prev_p else "#ef4444"
            formatted_preds.append({
                'time': row['timestamp'].strftime('%H:%M'),
                'price': f"{curr:.2f}",
                'trend': trend_arrow,
                'color': color
            })
            prev_p = curr

        vol_calc = pred_df['price'].std() if len(pred_df) > 1 else 0

        return render_template_string(HTML_TEMPLATE,
                                      date=date, time=time_input, mode=mode,
                                      plot_url=plot_url,
                                      predictions=formatted_preds,
                                      sentiment_score=f"{sent_score:.3f}",
                                      sentiment_class=sent_class,
                                      sentiment_text=sent_text,
                                      volatility=f"{vol_calc:.2f}",
                                      metrics=metrics)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template_string(HTML_TEMPLATE, date=request.form.get('date'), time=request.form.get('time'), error=str(e))

if __name__ == '__main__':
    print("AI Nifty Predictor v2.0 Initialized...")
    app.run(debug=True, port=5000)
