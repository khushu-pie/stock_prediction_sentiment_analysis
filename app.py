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

# --- Page Setup ---
st.set_page_config(page_title="AI Stock Oracle", layout="wide")
st.title("🚀 PRO-TRADER AI COMMAND DASHBOARD")
st.markdown("### Deep Learning & Sentiment-Driven Market Analysis")

# --- API Key: Read from Streamlit Secrets (NOT shown in sidebar) ---
# To set this up on Streamlit Cloud:
#   1. Go to your app's Settings → Secrets
#   2. Add:  GNEWS_API_KEY = "your_actual_key_here"
try:
    API_KEY = st.secrets["GNEWS_API_KEY"]
except (KeyError, FileNotFoundError):
    # Fallback for local development: set env variable or hardcode temporarily
    import os
    API_KEY = os.environ.get("GNEWS_API_KEY", "")
    if not API_KEY:
        st.sidebar.warning("⚠️ GNews API key not found in secrets. Add `GNEWS_API_KEY` to your Streamlit secrets.")

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Configuration")

# Preset popular tickers for the dropdown
PRESET_TICKERS = [
    # US Stocks
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
    "JPM", "BAC", "WMT", "DIS",
    # Indian Stocks (NSE)
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "WIPRO.NS", "SBIN.NS", "BHARTIARTL.NS", "TATAMOTORS.NS", "LT.NS",
    # Crypto
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD",
    # Commodities / ETFs
    "GLD", "SPY", "QQQ",
]

# --- Multiselect Dropdown ---
selected_from_dropdown = st.sidebar.multiselect(
    "📋 Select Tickers from List",
    options=PRESET_TICKERS,
    default=["AAPL", "TSLA", "RELIANCE.NS"],
    help="Choose one or more tickers from the list. You can also type to search."
)

# --- Custom Ticker Input ---
st.sidebar.markdown("---")
custom_input = st.sidebar.text_input(
    "➕ Add Custom Tickers (comma-separated)",
    value="",
    placeholder="e.g. AMZN, HDFCBANK.NS, BTC-USD",
    help="Type any ticker not in the list above. Separate multiple tickers with commas."
)

# Merge dropdown selections + custom input into one clean list (deduplicated)
custom_tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
selected_tickers = list(dict.fromkeys(selected_from_dropdown + custom_tickers))  # preserves order, removes dupes

# Show final ticker list as a preview
if selected_tickers:
    st.sidebar.markdown("**✅ Tickers to Analyze:**")
    st.sidebar.code(", ".join(selected_tickers))
else:
    st.sidebar.info("Select at least one ticker to continue.")

lookback = 60


class UltimateTradingBot:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def get_sentiment_details(self, ticker):
        query = ticker.split('.')[0]
        url = f"https://gnews.io/api/v4/search?q={query}&lang=en&token={API_KEY}&max=5"
        try:
            res = requests.get(url, timeout=5).json()
            articles = res.get("articles", [])
            if not articles:
                return 0.0, "😐 Neutral"
            score = np.mean([self.sia.polarity_scores(a['title'])['compound'] for a in articles])
            if score > 0.2:
                verdict = "🚀 Bullish"
            elif score > 0.05:
                verdict = "📈 Positive"
            elif score < -0.2:
                verdict = "📉 Panic/Fear"
            elif score < -0.05:
                verdict = "⚠️ Negative"
            else:
                verdict = "😐 Neutral"
            return round(score, 2), verdict
        except Exception:
            return 0.0, "Limit/Error"

    def run_analysis(self, ticker):
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        score, word = self.get_sentiment_details(ticker)
        data = df[['Close']].values
        scaled_data = self.scaler.fit_transform(data)

        sent_feat = np.full((len(scaled_data), 1), score)
        combined = np.hstack((scaled_data, sent_feat))

        X, y = [], []
        for i in range(lookback, len(combined)):
            X.append(combined[i - lookback:i])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=8, batch_size=32, verbose=0)

        last_win = combined[-lookback:].reshape(1, lookback, 2)
        pred = model.predict(last_win, verbose=0)
        final_pred = self.scaler.inverse_transform(pred)[0][0]
        last_price = df['Close'].iloc[-1].item()
        move = ((final_pred - last_price) / last_price) * 100

        if move > 1.2 and score > 0.1:
            advice = "STRONG BUY"
        elif move > 0.5:
            advice = "BUY / HOLD"
        elif move < -1.2:
            advice = "SELL / EXIT"
        else:
            advice = "NEUTRAL"

        return {
            "Ticker": ticker, "Price": last_price, "Target": final_pred,
            "Move": move, "Sent_Score": score, "Sent_Mood": word,
            "Advice": advice, "History": df['Close'].tail(100)
        }


# --- Execution ---
if st.sidebar.button("🚀 Run Global Analysis"):
    if not selected_tickers:
        st.error("Please select or enter at least one ticker symbol.")
    elif not API_KEY:
        st.error("GNews API key is missing. Add it to Streamlit Secrets before running.")
    else:
        bot = UltimateTradingBot()
        all_results = []

        progress_bar = st.progress(0)

        for idx, s in enumerate(selected_tickers):
            with st.spinner(f"Analyzing {s}..."):
                res = bot.run_analysis(s)
                if res:
                    all_results.append(res)
                else:
                    st.warning(f"Ticker '{s}' not found or no data available.")
            progress_bar.progress((idx + 1) / len(selected_tickers))

        if all_results:
            st.subheader("📊 Final Decision Dashboard")
            summary_data = []
            for r in all_results:
                risk = "🔴 High" if abs(r['Move']) > 3 else "🟡 Mod" if abs(r['Move']) > 1.5 else "🟢 Low"
                summary_data.append([
                    r['Ticker'], f"{r['Price']:.2f}", f"{r['Target']:.2f}",
                    f"{r['Move']:+.2f}%", r['Sent_Mood'], risk, r['Advice']
                ])

            df_final = pd.DataFrame(
                summary_data,
                columns=["Ticker", "Price", "Target", "Move", "Sentiment", "Risk", "Verdict"]
            )
            st.table(df_final)

            st.subheader("📈 Technical Analysis Graphs")
            graph_cols = st.columns(2)
            for idx, r in enumerate(all_results):
                with graph_cols[idx % 2]:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(r['History'].values, color='#2c3e50', linewidth=2, label="History")
                    ax.axhline(y=r['Target'], color='#e74c3c', linestyle='--',
                               label=f"Target: {r['Target']:.2f}")
                    ax.set_title(f"{r['Ticker']} Price Action")
                    ax.legend()
                    st.pyplot(fig)
        else:
            st.error("No valid data retrieved for the tickers provided.")
