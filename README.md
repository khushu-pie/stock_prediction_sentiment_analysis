# 📈 Stock Price Predictor — Deep Learning & Sentiment-Driven Market Analysis

A real-time stock price prediction web application that combines **LSTM Deep Learning** with **News Sentiment Analysis** to generate BUY / SELL / HOLD trading signals — all inside a browser-based Streamlit app.

---

## 🚀 Live Demo

Run locally with:

```bash
streamlit run app.py
```

---

## 🧠 What It Does

1. You type company names or tickers (e.g., `Apple, Reliance, Tesla`)
2. The app resolves names to official ticker symbols (e.g., `AAPL`, `RELIANCE.NS`, `TSLA`)
3. It downloads **2 years of daily closing price data** from Yahoo Finance
4. It fetches **5 latest news headlines** for each company via GNews API
5. **VADER Sentiment Analysis** scores each headline (Bullish / Bearish / Neutral)
6. An **LSTM Neural Network** is trained on the combined price + sentiment data
7. The model predicts the **next-day closing price**
8. A final **BUY / SELL / HOLD** verdict is generated with a risk level
9. Results are shown in a **dashboard table + price forecast graphs**

---

## 🏗️ System Architecture

```
User Input (Browser Sidebar)
         │
         ▼
  Name → Ticker Resolution
  (Yahoo Finance Search API)
         │
         ├──────────────────────────────────┐
         ▼                                  ▼
  Yahoo Finance (yfinance)           GNews API
  2 Years Daily Prices               Latest 5 Headlines
         │                                  │
         │                                  ▼
         │                     VADER Sentiment Analyzer
         │                     Compound Score: -1 to +1
         │                                  │
         └──────────────┬───────────────────┘
                        ▼
           Data Preprocessing
           MinMaxScaler → Normalize prices to [0, 1]
           Combine: [Scaled Price | Sentiment Score]
                        │
                        ▼
           Sliding Window (Lookback = 60 days)
           X = last 60 days, y = next day price
                        │
                        ▼
           LSTM Neural Network (TensorFlow/Keras)
           LSTM(50) → Dropout(0.2) → LSTM(50) → Dense(1)
                        │
                        ▼
           Inverse Scale → Real Price Prediction
                        │
                        ▼
           BUY / SELL / HOLD Verdict + Risk Level
                        │
                        ▼
           Streamlit Dashboard
           Table + Price Forecast Graph
```

---

## 🔄 Step-by-Step Workflow

| Step | What Happens |
|------|-------------|
| **1. Input** | User enters company names/tickers in the sidebar |
| **2. Resolve** | Names converted to tickers via Yahoo Finance Search |
| **3. Price Data** | 2 years of daily OHLCV data fetched via `yfinance` |
| **4. News Fetch** | 5 recent headlines fetched via GNews API |
| **5. Sentiment** | VADER scores each headline → average compound score |
| **6. Preprocess** | Prices normalized (0–1), sentiment appended as 2nd feature |
| **7. Sequence** | 60-day sliding windows created for LSTM input |
| **8. Train** | LSTM model trained for 8 epochs (fresh per run) |
| **9. Predict** | Next-day price predicted and inverse-scaled to real value |
| **10. Verdict** | Move % + sentiment score → BUY / SELL / HOLD + Risk badge |

---

## 🤖 LSTM Model Architecture

```
Input Shape: (60 days × 2 features)
      ↓
LSTM Layer — 50 neurons, return_sequences=True
      ↓
Dropout — 20% (prevents overfitting)
      ↓
LSTM Layer — 50 neurons
      ↓
Dense Layer — 1 neuron (output: predicted price)

Optimizer : Adam
Loss      : Mean Squared Error (MSE)
Epochs    : 8
Batch Size: 32
```

---

## 📋 Verdict Logic

```
Predicted Move = ((Target Price - Current Price) / Current Price) × 100

STRONG BUY   → Move > +1.2%  AND  Sentiment > 0.1
BUY / HOLD   → Move > +0.5%
SELL / EXIT  → Move < -1.2%
NEUTRAL      → Everything else

Risk Level:
  🔴 High     → |Move| > 3%
  🟡 Moderate → |Move| > 1.5%
  🟢 Low      → |Move| ≤ 1.5%
```

---

## 📊 Sentiment Verdict Mapping

| Score Range | Verdict |
|---|---|
| > 0.2 | 🚀 Bullish |
| 0.05 – 0.2 | 📈 Positive |
| -0.05 – 0.05 | 😐 Neutral |
| -0.2 – -0.05 | ⚠️ Negative |
| < -0.2 | 📉 Panic / Fear |

---

## 📦 Tech Stack

| Technology | Role |
|---|---|
| **Streamlit** | Web UI framework — entire app in one Python file |
| **yfinance** | Fetches real-time & historical stock data from Yahoo Finance |
| **GNews API** | Fetches latest financial news headlines |
| **VADER Sentiment** | NLP rule-based sentiment scoring for news headlines |
| **TensorFlow / Keras** | Builds and trains the LSTM neural network |
| **scikit-learn** | MinMaxScaler for data normalization |
| **Pandas / NumPy** | Data manipulation and array operations |
| **Matplotlib** | Price history + prediction line charts |

---

## 📁 Project Structure

```
stock_prediction_sentiment_analysis/
│
├── app.py               # Main application — entire logic lives here (201 lines)
├── requirements.txt     # All Python dependencies
├── README.md            # This file
├── .python-version      # Python version pin
└── .streamlit/          # Streamlit configuration folder
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+ (see `.python-version`)
- A free [GNews API Key](https://gnews.io/) (100 requests/day on free tier)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/stock_prediction_sentiment_analysis.git
cd stock_prediction_sentiment_analysis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the API Key

**Option A — Streamlit Secrets (Recommended)**

Create `.streamlit/secrets.toml`:

```toml
GNEWS_API_KEY = "your_gnews_api_key_here"
```

**Option B — Manual Entry**

Leave the secrets file empty — the app will show a password field in the sidebar at runtime.

### 4. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🖥️ Usage

1. Open the app in your browser
2. In the **sidebar**, enter comma-separated company names or tickers:
   ```
   Apple, Reliance, Tesla, INFY, Microsoft
   ```
3. Click **"Run Global Analysis"**
4. Wait for the AI to process each stock (~30–60 sec per stock depending on hardware)
5. View the **Decision Dashboard** table and **forecast graphs**

---

## 📉 Example Output

| Ticker | Price | Target | Move | Sentiment | Risk | Verdict |
|---|---|---|---|---|---|---|
| AAPL | 189.30 | 192.15 | +1.51% | 🚀 Bullish | 🟡 Mod | STRONG BUY |
| TSLA | 175.20 | 171.80 | -1.94% | 😐 Neutral | 🔴 High | SELL / EXIT |
| RELIANCE.NS | 2850.00 | 2863.40 | +0.47% | 📈 Positive | 🟢 Low | BUY / HOLD |

---

## ⚠️ Limitations & Disclaimer

| Limitation | Details |
|---|---|
| **No saved model** | Model is retrained fresh every run — slow but always current |
| **8 epochs only** | Trade-off between speed and accuracy |
| **Close price only** | Does not use volume, open, high, or low as features |
| **VADER is rule-based** | Not as powerful as transformer models (BERT, FinBERT) |
| **Next-day prediction** | Not suitable for long-term investment decisions |
| **GNews free tier** | Limited to 100 API calls/day; may hit limits with many stocks |

> ⚠️ **This is an educational project and NOT financial advice. Do not use these predictions for real trading decisions.**

---

## 🔮 Future Improvements

- [ ] Save trained models to disk (`.h5`) to avoid retraining every session
- [ ] Replace VADER with **FinBERT** (finance-specific BERT model) for better sentiment
- [ ] Add more features: volume, RSI, MACD, moving averages
- [ ] Implement multi-day forecasting (predict 7 / 30 days ahead)
- [ ] Add portfolio-level analysis (track multiple stocks together)
- [ ] Deploy on **Streamlit Cloud** or **Hugging Face Spaces**
- [ ] Add backtesting module to validate predictions against historical data

---

## 👤 Author

Built as an academic/learning project demonstrating the fusion of **Time-Series Deep Learning** and **Natural Language Processing (NLP)** for financial market analysis.

---

## 📜 License

This project is licensed under the **MIT License** — free to use and modify for educational purposes.