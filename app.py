import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf

# --- Page Setup ---
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("STOCK PRICE PREDICTOR AGENT")
st.markdown("### Deep Learning & Sentiment-Driven Market Analysis")

# --- Security: API Key Management ---
if "GNEWS_API_KEY" in st.secrets:
    API_KEY = st.secrets["GNEWS_API_KEY"]
else:
    API_KEY = st.sidebar.text_input("GNews API Key (Manual Entry)", type="password")

# --- Helper Function: Ticker Search ---
def get_ticker_from_name(query):
    """Attempts to resolve a company name to a Yahoo Finance ticker."""
    query = query.strip()
    # If it already looks like a ticker (all caps, no spaces), return it
    if query.isupper() and " " not in query:
        return query
    
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5).json()
        # Grab the first symbol found in the 'quotes' list
        ticker = response['quotes'][0]['symbol']
        return ticker
    except:
        return query # Fallback to original input if search fails

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input(
    "Enter Company Names or Tickers", 
    value="Apple, Reliance, Tesla", 
    help="You can type names like 'Tata Motors' or tickers like 'AAPL'"
)

# Convert string input into a list of names/tickers
raw_inputs = [t.strip() for t in ticker_input.split(",") if t.strip()]

lookback = 60

class UltimateTradingBot:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    @st.cache_data(ttl=3600)
    def get_sentiment_details(_self, ticker, api_key):
        query = ticker.split('.')[0]
        url = f"https://gnews.io/api/v4/search?q={query}&lang=en&token={api_key}&max=5"
        try:
            res = requests.get(url, timeout=5).json()
            articles = res.get("articles", [])
            if not articles: return 0.0, "😐 Neutral"
            score = np.mean([_self.sia.polarity_scores(a['title'])['compound'] for a in articles])
            if score > 0.2: verdict = "🚀 Bullish"
            elif score > 0.05: verdict = "📈 Positive"
            elif score < -0.2: verdict = "📉 Panic/Fear"
            elif score < -0.05: verdict = "⚠️ Negative"
            else: verdict = "😐 Neutral"
            return round(float(score), 2), verdict
        except:
            return 0.0, "Limit/Error"
            
    def run_analysis(self, ticker):
        # 1. Fetch Data
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
        if df.empty: return None

        # --- FIX FOR nan PRICE ---
        # Force the columns to be simple (strips the ticker name from the header)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Ensure we are using the 'Close' column and remove any empty rows
        df = df[['Close']].ffill().dropna()
        # -------------------------
        
        # 2. Get Sentiment
        score, word = self.get_sentiment_details(ticker, API_KEY)
        
        # 3. Preprocessing
        data = df.values
        scaled_data = self.scaler.fit_transform(data)
        sent_feat = np.full((len(scaled_data), 1), score)
        combined = np.hstack((scaled_data, sent_feat))

        X, y = [], []
        for i in range(lookback, len(combined)):
            X.append(combined[i-lookback:i])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)

        # 4. Model Training
        tf.keras.backend.clear_session()
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=8, batch_size=32, verbose=0) 

        # 5. Prediction & Final Value Cleaning
        last_win = combined[-lookback:].reshape(1, lookback, 2)
        pred = model.predict(last_win, verbose=0)
        
        # CLEANING STEP: Force everything to be a standard Python float
        final_pred = float(self.scaler.inverse_transform(pred)[0][0])
        
        # Extract the last valid price. We use .item() to ensure it's a number, not a list.
        last_price = float(df['Close'].iloc[-1])
        
        # Calculate Move - now that last_price isn't nan, this will work!
        move = ((final_pred - last_price) / last_price) * 100
        
        # Verdict Logic
        if move > 1.2 and score > 0.1: advice = "STRONG BUY"
        elif move > 0.5: advice = "BUY / HOLD"
        elif move < -1.2: advice = "SELL / EXIT"
        else: advice = "NEUTRAL"

        return {
            "Ticker": ticker, 
            "Price": last_price, 
            "Target": final_pred,
            "Move": move, 
            "Sent_Score": score, 
            "Sent_Mood": word,
            "Advice": advice, 
            "History": df['Close'].tail(100)
        }

     

# --- Execution ---
if st.sidebar.button("Run Global Analysis"):
    if not API_KEY:
        st.error("Missing API Key! Please add it to Streamlit Secrets or sidebar.")
    elif not raw_inputs:
        st.error("Please enter at least one company or ticker.")
    else:
        bot = UltimateTradingBot()
        all_results = []
        progress_bar = st.progress(0)
        
        # RESOLUTION STEP: Convert names to tickers
        resolved_tickers = []
        with st.spinner("Resolving company names to tickers..."):
            for item in raw_inputs:
                ticker = get_ticker_from_name(item)
                resolved_tickers.append(ticker)
        
        # Main Analysis Loop
        for idx, s in enumerate(resolved_tickers):
            with st.spinner(f"AI is processing {s}..."):
                res = bot.run_analysis(s)
                if res: 
                    all_results.append(res)
                else:
                    st.warning(f"Could not find data for '{s}'.")
            progress_bar.progress((idx + 1) / len(resolved_tickers))

        if all_results:
            st.subheader("Final Decision Dashboard")
            summary_data = []
            for r in all_results:
                risk = "🔴 High" if abs(r['Move']) > 3 else "🟡 Mod" if abs(r['Move']) > 1.5 else "🟢 Low"
                summary_data.append([r['Ticker'], f"{r['Price']:.2f}", f"{r['Target']:.2f}", f"{r['Move']:+.2f}%", r['Sent_Mood'], risk, r['Advice']])

            df_final = pd.DataFrame(summary_data, columns=["Ticker", "Price", "Target", "Move", "Sentiment", "Risk", "Verdict"])
            st.table(df_final)

            st.subheader("Technical Analysis Graphs")
            graph_cols = st.columns(2)
            for idx, r in enumerate(all_results):
                with graph_cols[idx % 2]:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(r['History'].values, color='#2c3e50', linewidth=2, label="Price History")
                    ax.axhline(y=r['Target'], color='#e74c3c', linestyle='--', label=f"AI Target: {r['Target']:.2f}")
                    ax.set_title(f"{r['Ticker']} Forecast")
                    ax.set_facecolor('#fdfdfd')
                    ax.legend()
                    st.pyplot(fig)
        else:
            st.error("No data could be retrieved.")
