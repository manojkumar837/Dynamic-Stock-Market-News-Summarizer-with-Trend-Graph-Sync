import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from textblob import TextBlob
from datetime import datetime, timedelta
import json

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# ─── Page config 
st.set_page_config(
    page_title="AI Stock Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS 
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  * { font-family: 'Inter', sans-serif; }
  code, .mono { font-family: 'JetBrains Mono', monospace; }

  [data-testid="stAppViewContainer"] { background: #0d0f14; }
  [data-testid="stSidebar"] { background: #12151c; border-right: 1px solid #1e2230; }

  .metric-card {
    background: #12151c;
    border: 1px solid #1e2230;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 4px;
  }
  .metric-label { color: #6b7280; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; }
  .metric-value { color: #f1f5f9; font-size: 24px; font-weight: 700; margin: 4px 0 2px; }
  .metric-delta-up   { color: #22c55e; font-size: 12px; font-weight: 600; }
  .metric-delta-down { color: #ef4444; font-size: 12px; font-weight: 600; }

  .eval-card {
    background: linear-gradient(135deg, #12151c 0%, #161a24 100%);
    border: 1px solid #1e2230;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
  }
  .eval-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #6366f1, #22c55e);
  }
  .eval-metric-label { color: #6b7280; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; }
  .eval-metric-value { color: #f1f5f9; font-size: 28px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
  .eval-metric-desc  { color: #475569; font-size: 11px; margin-top: 4px; }

  .news-card {
    background: #12151c;
    border: 1px solid #1e2230;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
  }
  .news-card:hover { border-color: #2e3450; }
  .news-title { color: #e2e8f0; font-size: 14px; font-weight: 600; margin-bottom: 6px; }
  .news-meta  { color: #6b7280; font-size: 12px; }

  .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    margin-left: 6px;
  }
  .badge-pos  { background: #14532d; color: #4ade80; }
  .badge-neg  { background: #450a0a; color: #f87171; }
  .badge-neu  { background: #1e2230; color: #94a3b8; }

  .section-header {
    color: #94a3b8; font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin: 24px 0 12px;
    border-bottom: 1px solid #1e2230;
    padding-bottom: 6px;
  }

  .insight-box {
    background: #161a24;
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 13px;
    color: #cbd5e1;
  }
  .insight-box strong { color: #a5b4fc; }

  h1, h2, h3 { color: #f1f5f9 !important; }
  .stTabs [data-baseweb="tab-list"] { background: #12151c; border-radius: 10px; }
  .stTabs [data-baseweb="tab"] { color: #6b7280; }
  .stTabs [aria-selected="true"] { color: #f1f5f9 !important; }

  .signal-buy  { color: #22c55e; font-weight: 700; font-size: 20px; }
  .signal-sell { color: #ef4444; font-weight: 700; font-size: 20px; }
  .signal-hold { color: #f59e0b; font-weight: 700; font-size: 20px; }

  .progress-bar-bg { background: #1e2230; border-radius: 4px; height: 6px; margin-top: 8px; }
  .progress-bar-fill { height: 6px; border-radius: 4px; background: linear-gradient(90deg, #6366f1, #22c55e); }
</style>
""", unsafe_allow_html=True)

# ─── Constants & Helpers 
STOCKS = ["TSLA", "AAPL", "MSFT", "GOOG", "AMZN", "NVDA"]

PLOT_THEME = dict(
    paper_bgcolor="#0d0f14",
    plot_bgcolor="#0d0f14",
    font_color="#94a3b8",
    font_family="Inter",
    xaxis=dict(gridcolor="#1e2230", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#1e2230", showgrid=True, zeroline=False),
    margin=dict(l=10, r=10, t=40, b=10),
)

def get_sentiment(text):
    p = TextBlob(str(text)).sentiment.polarity
    if p > 0.05:  return "Positive"
    elif p < -0.05: return "Negative"
    else: return "Neutral"

def badge_html(sentiment):
    cls  = {"Positive": "badge-pos", "Negative": "badge-neg", "Neutral": "badge-neu"}.get(sentiment, "badge-neu")
    icon = {"Positive": "▲", "Negative": "▼", "Neutral": "●"}.get(sentiment, "●")
    return f'<span class="badge {cls}">{icon} {sentiment}</span>'

@st.cache_data(ttl=300)
def load_stock(ticker, period):
    return yf.download(ticker, period=period, auto_adjust=True, progress=False)

@st.cache_data(ttl=600)
def load_all_stocks(period="1mo"):
    data = {}
    for s in STOCKS:
        df = yf.download(s, period=period, auto_adjust=True, progress=False)
        if not df.empty:
            data[s] = df["Close"].squeeze()
    return pd.DataFrame(data).dropna()

def add_indicators(df):
    close = df["Close"].squeeze()
    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]
    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_upper"] = sma20 + 2 * std20
    df["BB_lower"] = sma20 - 2 * std20
    df["BB_mid"]   = sma20
    # EMAs
    df["EMA50"]  = close.ewm(span=50).mean()
    df["EMA200"] = close.ewm(span=200).mean()
    # ATR (Average True Range)
    high = df["High"].squeeze()
    low  = df["Low"].squeeze()
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    # OBV (On-Balance Volume)
    direction = np.sign(close.diff()).fillna(0)
    df["OBV"] = (direction * df["Volume"].squeeze()).cumsum()
    # Stochastic %K
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["Stoch_K"] = 100 * (close - low14) / (high14 - low14 + 1e-9)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()
    # Daily returns & rolling vol
    df["Returns"]    = close.pct_change()
    df["RollingVol"] = df["Returns"].rolling(20).std() * np.sqrt(252) * 100
    return df

def linear_forecast(close_series, horizon=5):
    """Simple linear regression forecast."""
    y = close_series.dropna().values
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    fut_x = np.arange(len(y), len(y) + horizon).reshape(-1, 1)
    return model.predict(fut_x), model.score(x, y)

# ─── Sidebar 
with st.sidebar:
    st.markdown("## ⚡ Stock Dashboard")
    st.markdown('<div class="section-header">Stock & Period</div>', unsafe_allow_html=True)
    ticker      = st.selectbox("Ticker", STOCKS)
    period      = st.selectbox("Period", ["5d","1mo","3mo","6mo","1y"], index=1)
    sent_filter = st.selectbox("Sentiment Filter", ["All","Positive","Negative","Neutral"])

    st.markdown('<div class="section-header">Indicators</div>', unsafe_allow_html=True)
    show_bb    = st.toggle("Bollinger Bands", value=True)
    show_rsi   = st.toggle("RSI", value=True)
    show_macd  = st.toggle("MACD", value=True)
    show_vol   = st.toggle("Volume", value=True)
    show_stoch = st.toggle("Stochastic", value=False)
    show_atr   = st.toggle("ATR", value=False)
    show_obv   = st.toggle("OBV", value=False)

    st.markdown('<div class="section-header">Price Alert</div>', unsafe_allow_html=True)
    alert_price = st.number_input("Alert when price hits ($)", min_value=0.0, value=0.0, step=1.0)

    st.markdown('<div class="section-header">Portfolio</div>', unsafe_allow_html=True)
    port_ticker = st.selectbox("Stock", STOCKS, key="port_ticker")
    port_shares = st.number_input("Shares owned", min_value=0.0, value=0.0, step=1.0)
    port_cost   = st.number_input("Avg buy price ($)", min_value=0.0, value=0.0, step=1.0)

# ─── Data Loading 
stock_data = load_stock(ticker, period)
if not stock_data.empty:
    stock_data = add_indicators(stock_data)

# ─── News Data 
try:
    df_news = pd.read_csv("news_data.csv")
except FileNotFoundError:
    df_news = pd.DataFrame({
        "title":     ["NVDA earnings beat expectations","Apple unveils new chip","Tesla cuts prices again",
                      "Microsoft Azure growth slows","Google launches AI search","Amazon AWS record quarter"],
        "source":    ["Reuters","Bloomberg","CNBC","WSJ","TechCrunch","Forbes"],
        "published": pd.date_range(end=datetime.today(), periods=6, freq="D").strftime("%Y-%m-%d").tolist(),
        "url":       ["#"] * 6,
    })

df_news["Sentiment"] = df_news["title"].apply(get_sentiment)
df_news["Summary"]   = df_news["title"].apply(lambda t: " ".join(str(t).split()[:12]) + "...")
if sent_filter != "All":
    df_news = df_news[df_news["Sentiment"] == sent_filter]

# ─── Header 
st.markdown(f"# {ticker} — AI Stock Dashboard")
st.caption(f"Last updated: {datetime.now().strftime('%b %d, %Y  %H:%M')}")

# ─── KPI Metrics 
if not stock_data.empty:
    close   = stock_data["Close"].squeeze()
    current = float(close.iloc[-1])
    prev    = float(close.iloc[-2]) if len(close) > 1 else current
    chg_abs = current - prev
    chg_pct = (chg_abs / prev) * 100
    high    = float(stock_data["High"].max().squeeze())
    low     = float(stock_data["Low"].min().squeeze())
    volume  = int(stock_data["Volume"].iloc[-1].squeeze())
    rsi_val = float(stock_data["RSI"].iloc[-1]) if "RSI" in stock_data.columns else 0
    atr_val = float(stock_data["ATR"].iloc[-1]) if "ATR" in stock_data.columns else 0
    vol_val = float(stock_data["RollingVol"].iloc[-1]) if "RollingVol" in stock_data.columns else 0

    # Linear regression forecast
    fcast_vals, fcast_r2 = linear_forecast(close, horizon=5)
    forecast_price = float(fcast_vals[-1])

    signal = "BUY" if rsi_val < 30 else "SELL" if rsi_val > 70 else "HOLD"

    if alert_price > 0 and current >= alert_price:
        st.toast(f"🔔 {ticker} hit your target of ${alert_price:.2f}!", icon="🚨")

    delta_color = "metric-delta-up" if chg_abs >= 0 else "metric-delta-down"
    delta_arrow = "▲" if chg_abs >= 0 else "▼"

    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    cards = [
        (c1, "Current Price",   f"${current:,.2f}", f'{delta_arrow} {chg_abs:+.2f} ({chg_pct:+.2f}%)', delta_color),
        (c2, "Period High",     f"${high:,.2f}",    "", ""),
        (c3, "Period Low",      f"${low:,.2f}",     "", ""),
        (c4, "Volume",          f"{volume:,}",       "", ""),
        (c5, "RSI (14)",        f"{rsi_val:.1f}",
             "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral"),
             "metric-delta-down" if rsi_val > 70 else "metric-delta-up"),
        (c6, "ATR (14)",        f"${atr_val:.2f}",  "Daily volatility range", "metric-delta-up"),
        (c7, "AI Signal",       signal, "", ""),
        (c8, "5d Forecast",     f"${forecast_price:.2f}",
             f"{'▲' if forecast_price > current else '▼'} {((forecast_price-current)/current)*100:+.1f}%",
             "metric-delta-up" if forecast_price > current else "metric-delta-down"),
    ]
    for col, label, val, delta, dcls in cards:
        with col:
            d_html = f'<div class="{dcls}">{delta}</div>' if delta else ""
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{val}</div>
              {d_html}
            </div>""", unsafe_allow_html=True)

    # Portfolio P&L
    if port_shares > 0 and port_cost > 0:
        port_data  = load_stock(port_ticker, "1d")
        port_price = float(port_data["Close"].iloc[-1].squeeze()) if not port_data.empty else port_cost
        pnl        = (port_price - port_cost) * port_shares
        pnl_pct    = ((port_price - port_cost) / port_cost) * 100
        pnl_color  = "#22c55e" if pnl >= 0 else "#ef4444"
        st.markdown(f"""
        <div class="metric-card" style="border-color:{pnl_color}33; margin-top:8px;">
          <div class="metric-label">Portfolio — {port_ticker} ({port_shares:.0f} shares @ ${port_cost:.2f})</div>
          <div class="metric-value" style="color:{pnl_color};">${pnl:+,.2f} ({pnl_pct:+.2f}%)</div>
          <div class="metric-label">Current: ${port_price:.2f} · Total value: ${port_price*port_shares:,.2f}</div>
        </div>""", unsafe_allow_html=True)

# ─── Tabs 
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Price & Indicators",
    "🤖 Prediction Analytics",
    "📰 News Feed",
    "📊 Comparison",
    "🔥 Correlation",
    "📉 Returns & Risk",
])



# TAB 1 — Price & Indicators

with tab1:
    if not stock_data.empty:
        extra_rows = int(show_rsi) + int(show_macd) + int(show_stoch) + int(show_atr) + int(show_obv)
        rows       = 1 + extra_rows
        row_heights = [max(0.4, 1 - extra_rows * 0.12)] + [0.12] * extra_rows
        subplot_titles = [f"{ticker} Price"] \
            + (["RSI"] if show_rsi else []) \
            + (["MACD"] if show_macd else []) \
            + (["Stochastic"] if show_stoch else []) \
            + (["ATR"] if show_atr else []) \
            + (["OBV"] if show_obv else [])

        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                            vertical_spacing=0.03,
                            row_heights=row_heights,
                            subplot_titles=subplot_titles)

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=stock_data.index,
            open=stock_data["Open"].squeeze(),
            high=stock_data["High"].squeeze(),
            low=stock_data["Low"].squeeze(),
            close=stock_data["Close"].squeeze(),
            name="Price",
            increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
            increasing_fillcolor="#22c55e",  decreasing_fillcolor="#ef4444",
        ), row=1, col=1)

        # Volume bars
        if show_vol:
            vol_colors = ["#22c55e" if c >= o else "#ef4444"
                          for c, o in zip(stock_data["Close"].squeeze(), stock_data["Open"].squeeze())]
            fig.add_trace(go.Bar(
                x=stock_data.index, y=stock_data["Volume"].squeeze(),
                name="Volume", marker_color=vol_colors, opacity=0.20, yaxis="y2"
            ), row=1, col=1)
            fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False,
                                          tickformat=".2s", title="Volume", title_font_color="#6b7280"))

        # EMAs
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["EMA50"],
                                 name="EMA 50",  line=dict(color="#00D4FF", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["EMA200"],
                                 name="EMA 200", line=dict(color="#FFB703", width=1.5)), row=1, col=1)

        # Bollinger Bands
        if show_bb and "BB_upper" in stock_data.columns:
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["BB_upper"].squeeze(),
                                     name="BB Upper", line=dict(color="#6366f1", width=1, dash="dot"), opacity=0.7), row=1, col=1)
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["BB_lower"].squeeze(),
                                     name="BB Lower", line=dict(color="#6366f1", width=1, dash="dot"), opacity=0.7,
                                     fill="tonexty", fillcolor="rgba(99,102,241,0.05)"), row=1, col=1)
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["BB_mid"].squeeze(),
                                     name="BB Mid", line=dict(color="#818cf8", width=1, dash="dash"), opacity=0.5), row=1, col=1)

        # Linear regression forecast line overlay
        if len(close) >= 5:
            fcast_vals5, _ = linear_forecast(close, horizon=10)
            last_date = stock_data.index[-1]
            freq = pd.infer_freq(stock_data.index) or "B"
            try:
                fut_dates = pd.date_range(start=last_date, periods=11, freq=freq)[1:]
            except Exception:
                fut_dates = pd.date_range(start=last_date, periods=11, freq="B")[1:]
            fig.add_trace(go.Scatter(
                x=list(stock_data.index[-1:]) + list(fut_dates),
                y=[float(close.iloc[-1])] + list(fcast_vals5),
                name="LR Forecast",
                line=dict(color="#f0abfc", width=2, dash="dash"),
                opacity=0.8,
            ), row=1, col=1)

        # News annotations
        if not df_news.empty and "published" in df_news.columns:
            try:
                for _, row in df_news.iterrows():
                    date = pd.to_datetime(row["published"], errors="coerce")
                    if pd.isna(date): continue
                    price = float(stock_data["Close"].squeeze().iloc[-1])
                    if date in stock_data.index:
                        price = float(stock_data.loc[date, "Close"].squeeze())
                    color = {"Positive":"#22c55e","Negative":"#ef4444","Neutral":"#f59e0b"}.get(row["Sentiment"],"#94a3b8")
                    fig.add_annotation(x=date, y=price, text="📰", showarrow=True,
                                       arrowhead=2, arrowcolor=color, arrowwidth=1.5,
                                       bgcolor="#12151c", bordercolor=color, font_size=10,
                                       row=1, col=1)
            except Exception:
                pass

        cur_row = 2

        # RSI
        if show_rsi and "RSI" in stock_data.columns:
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["RSI"].squeeze(),
                                     name="RSI", line=dict(color="#f59e0b", width=1.5)), row=cur_row, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="#ef4444", opacity=0.5, row=cur_row, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="#22c55e", opacity=0.5, row=cur_row, col=1)
            fig.add_hrect(y0=70, y1=100, fillcolor="#ef4444", opacity=0.05, row=cur_row, col=1)
            fig.add_hrect(y0=0,  y1=30,  fillcolor="#22c55e", opacity=0.05, row=cur_row, col=1)
            cur_row += 1

        # MACD
        if show_macd and "MACD" in stock_data.columns:
            hist = stock_data["MACD_hist"].squeeze()
            fig.add_trace(go.Bar(x=stock_data.index, y=hist, name="MACD Hist",
                                 marker_color=["#22c55e" if v >= 0 else "#ef4444" for v in hist], opacity=0.7), row=cur_row, col=1)
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["MACD"].squeeze(),
                                     name="MACD",   line=dict(color="#6366f1", width=1.5)), row=cur_row, col=1)
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["MACD_signal"].squeeze(),
                                     name="Signal", line=dict(color="#f59e0b", width=1.5, dash="dot")), row=cur_row, col=1)
            cur_row += 1

        # Stochastic
        if show_stoch and "Stoch_K" in stock_data.columns:
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["Stoch_K"].squeeze(),
                                     name="%K", line=dict(color="#ec4899", width=1.5)), row=cur_row, col=1)
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["Stoch_D"].squeeze(),
                                     name="%D", line=dict(color="#f97316", width=1.5, dash="dot")), row=cur_row, col=1)
            fig.add_hline(y=80, line_dash="dot", line_color="#ef4444", opacity=0.4, row=cur_row, col=1)
            fig.add_hline(y=20, line_dash="dot", line_color="#22c55e", opacity=0.4, row=cur_row, col=1)
            cur_row += 1

        # ATR
        if show_atr and "ATR" in stock_data.columns:
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["ATR"].squeeze(),
                                     name="ATR", line=dict(color="#14b8a6", width=1.5),
                                     fill="tozeroy", fillcolor="rgba(20,184,166,0.08)"), row=cur_row, col=1)
            cur_row += 1

        # OBV
        if show_obv and "OBV" in stock_data.columns:
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["OBV"].squeeze(),
                                     name="OBV", line=dict(color="#a78bfa", width=1.5),
                                     fill="tozeroy", fillcolor="rgba(167,139,250,0.08)"), row=cur_row, col=1)

        fig.update_layout(
            height=700, xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                        bgcolor="rgba(0,0,0,0)", font_color="#94a3b8"),
            **PLOT_THEME
        )
        fig.update_xaxes(gridcolor="#1e2230")
        fig.update_yaxes(gridcolor="#1e2230")
        st.plotly_chart(fig, use_container_width=True)

        # Earnings Calendar
        try:
            info = yf.Ticker(ticker)
            cal  = info.calendar
            if cal is not None and not cal.empty:
                st.markdown('<div class="section-header">Earnings Calendar</div>', unsafe_allow_html=True)
                st.dataframe(cal.T, use_container_width=True)
        except Exception:
            pass
    else:
        st.error("No stock data available for the selected ticker/period.")



# TAB 2 — Prediction Analytics

with tab2:
    if not stock_data.empty and len(stock_data) >= 30:
        close = stock_data["Close"].squeeze()

        # ── Compute metrics 
        actual    = close.shift(-1).dropna()
        predicted = close.rolling(3).mean().dropna()
        min_len   = min(len(actual), len(predicted))
        actual    = actual.iloc[:min_len]
        predicted = predicted.iloc[:min_len]

        mae  = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2   = r2_score(actual, predicted)
        mape = float(np.mean(np.abs((actual - predicted) / actual)) * 100)
        accuracy = float(np.clip(r2 * 100, 0, 100))

        # Directional accuracy
        actual_dir    = np.sign(actual.diff().dropna())
        predicted_dir = np.sign(predicted.diff().dropna())
        da_len = min(len(actual_dir), len(predicted_dir))
        dir_acc = float(np.mean(actual_dir.values[:da_len] == predicted_dir.values[:da_len]) * 100)

        # ─ Evaluation cards 
        st.markdown('<div class="section-header">Model Evaluation Metrics</div>', unsafe_allow_html=True)

        ec1, ec2, ec3, ec4, ec5, ec6 = st.columns(6)

        def eval_card(col, label, value, desc, color="#f1f5f9"):
            with col:
                st.markdown(f"""
                <div class="eval-card">
                  <div class="eval-metric-label">{label}</div>
                  <div class="eval-metric-value" style="color:{color};">{value}</div>
                  <div class="eval-metric-desc">{desc}</div>
                </div>""", unsafe_allow_html=True)

        eval_card(ec1, "MAE",  f"${mae:.2f}",  "Mean absolute error",  "#f59e0b")
        eval_card(ec2, "RMSE", f"${rmse:.2f}", "Root mean sq error",   "#f97316")
        eval_card(ec3, "R²",   f"{r2:.3f}",    "Variance explained",
                  "#22c55e" if r2 > 0.7 else "#f59e0b" if r2 > 0.4 else "#ef4444")
        eval_card(ec4, "MAPE", f"{mape:.2f}%", "Mean abs % error",
                  "#22c55e" if mape < 2 else "#f59e0b" if mape < 5 else "#ef4444")
        eval_card(ec5, "Dir. Acc", f"{dir_acc:.1f}%", "Directional accuracy",
                  "#22c55e" if dir_acc > 55 else "#ef4444")
        eval_card(ec6, "Model Score", f"{accuracy:.1f}%", "Overall R² based score",
                  "#22c55e" if accuracy > 70 else "#f59e0b" if accuracy > 40 else "#ef4444")

        # Insight box
        def model_insight(r2, mape, dir_acc):
            if r2 > 0.8 and dir_acc > 60:
                return "✅ <strong>Strong model performance.</strong> High R² and directional accuracy suggest the 3-day MA predictor tracks price movement well for this period."
            elif r2 > 0.5:
                return "⚠️ <strong>Moderate fit.</strong> The model captures some price trend but may miss short-term reversals. Consider adding momentum features."
            else:
                return "❌ <strong>Weak fit.</strong> High MAPE or low R² suggests significant prediction error — the 3-day MA lags too much. A more sophisticated model is recommended."

        st.markdown(f'<div class="insight-box">{model_insight(r2, mape, dir_acc)}</div>', unsafe_allow_html=True)

        # Gauge chart
        g1, g2 = st.columns(2)
        with g1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=accuracy,
                delta={"reference": 50, "increasing": {"color": "#22c55e"}},
                title={"text": "Model Accuracy (%)", "font": {"color": "#94a3b8", "size": 14}},
                number={"font": {"color": "#f1f5f9", "size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#6b7280"},
                    "bar":  {"color": "#6366f1", "thickness": 0.3},
                    "bgcolor": "#1e2230",
                    "steps": [
                        {"range": [0,  40], "color": "#450a0a"},
                        {"range": [40, 70], "color": "#422006"},
                        {"range": [70, 100],"color": "#14532d"},
                    ],
                    "threshold": {"line": {"color": "#f1f5f9", "width": 2}, "value": accuracy},
                }
            ))
            fig_gauge.update_layout(height=280, **PLOT_THEME)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with g2:
            fig_dir = go.Figure(go.Indicator(
                mode="gauge+number",
                value=dir_acc,
                title={"text": "Directional Accuracy (%)", "font": {"color": "#94a3b8", "size": 14}},
                number={"font": {"color": "#f1f5f9", "size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#6b7280"},
                    "bar":  {"color": "#22c55e" if dir_acc > 55 else "#ef4444", "thickness": 0.3},
                    "bgcolor": "#1e2230",
                    "steps": [
                        {"range": [0,  50], "color": "#450a0a"},
                        {"range": [50, 65], "color": "#422006"},
                        {"range": [65, 100],"color": "#14532d"},
                    ],
                    "threshold": {"line": {"color": "#f1f5f9", "width": 2}, "value": 50},
                }
            ))
            fig_dir.update_layout(height=280, **PLOT_THEME)
            st.plotly_chart(fig_dir, use_container_width=True)

        #  Actual vs Predicted 
        st.markdown('<div class="section-header">Actual vs Predicted Price</div>', unsafe_allow_html=True)
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(x=actual.index, y=actual,
                                      name="Actual", line=dict(color="#22c55e", width=2)))
        fig_eval.add_trace(go.Scatter(x=actual.index, y=predicted,
                                      name="Predicted (3d MA)", line=dict(color="#6366f1", width=2, dash="dot")))
        # Error fill
        fig_eval.add_trace(go.Scatter(
            x=list(actual.index) + list(actual.index[::-1]),
            y=list(predicted) + list(actual)[::-1],
            fill="toself", fillcolor="rgba(99,102,241,0.07)",
            line=dict(color="rgba(0,0,0,0)"), name="Error Band", showlegend=True
        ))
        fig_eval.update_layout(title="Rolling 3-day MA Prediction vs Actual Close",
                                height=400, **PLOT_THEME)
        st.plotly_chart(fig_eval, use_container_width=True)

        #  Residuals 
        st.markdown('<div class="section-header">Residuals Analysis</div>', unsafe_allow_html=True)
        residuals = (actual.values - predicted.values)

        res1, res2 = st.columns(2)
        with res1:
            fig_res = go.Figure()
            fig_res.add_trace(go.Bar(
                x=actual.index, y=residuals,
                marker_color=["#22c55e" if v >= 0 else "#ef4444" for v in residuals],
                name="Residual", opacity=0.8
            ))
            fig_res.add_hline(y=0, line_color="#94a3b8", line_width=1)
            fig_res.update_layout(title="Prediction Residuals Over Time",
                                   height=300, **PLOT_THEME)
            st.plotly_chart(fig_res, use_container_width=True)

        with res2:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=residuals, nbinsx=20,
                marker_color="#6366f1", opacity=0.8, name="Residual Dist."
            ))
            fig_hist.update_layout(title="Residual Distribution",
                                    height=300, **PLOT_THEME)
            st.plotly_chart(fig_hist, use_container_width=True)

        #  Forecast (Linear Regression) 
        st.markdown('<div class="section-header">Linear Regression Forecast (Next 10 Days)</div>', unsafe_allow_html=True)
        fcast10, lr_r2 = linear_forecast(close, horizon=10)
        last_date = stock_data.index[-1]
        try:
            freq = pd.infer_freq(stock_data.index) or "B"
            fut_dates = pd.date_range(start=last_date, periods=11, freq=freq)[1:]
        except Exception:
            fut_dates = pd.date_range(start=last_date, periods=11, freq="B")[1:]

        # Confidence interval (±1 std of residuals)
        res_std = float(np.std(residuals))
        upper = fcast10 + 1.96 * res_std
        lower = fcast10 - 1.96 * res_std

        fig_fcast = go.Figure()
        fig_fcast.add_trace(go.Scatter(x=stock_data.index[-30:], y=close.values[-30:],
                                        name="Historical", line=dict(color="#22c55e", width=2)))
        fig_fcast.add_trace(go.Scatter(x=fut_dates, y=fcast10,
                                        name="Forecast", line=dict(color="#f0abfc", width=2, dash="dash"),
                                        mode="lines+markers", marker=dict(size=6, color="#f0abfc")))
        fig_fcast.add_trace(go.Scatter(
            x=list(fut_dates) + list(fut_dates[::-1]),
            y=list(upper) + list(lower[::-1]),
            fill="toself", fillcolor="rgba(240,171,252,0.1)",
            line=dict(color="rgba(0,0,0,0)"), name="95% CI"
        ))
        fig_fcast.add_vline(x=str(last_date), line_dash="dot", line_color="#6b7280", opacity=0.5)
        fig_fcast.update_layout(title=f"10-Day Forecast (LR R²={lr_r2:.3f})",
                                  height=380, **PLOT_THEME)
        st.plotly_chart(fig_fcast, use_container_width=True)

        # Feature Importance (rolling correlation) 
        st.markdown('<div class="section-header">Feature Correlation with Next-Day Close</div>', unsafe_allow_html=True)
        features = {}
        next_close = close.shift(-1)
        for feat in ["RSI","MACD","ATR","EMA50","BB_mid","Stoch_K","RollingVol","OBV"]:
            if feat in stock_data.columns:
                s = stock_data[feat].squeeze()
                corr_val = float(s.corr(next_close))
                if not np.isnan(corr_val):
                    features[feat] = corr_val

        if features:
            feat_df = pd.DataFrame({"Feature": list(features.keys()),
                                     "Correlation": list(features.values())})
            feat_df = feat_df.sort_values("Correlation", key=abs, ascending=True)
            fig_feat = go.Figure(go.Bar(
                x=feat_df["Correlation"], y=feat_df["Feature"],
                orientation="h",
                marker_color=["#22c55e" if v > 0 else "#ef4444" for v in feat_df["Correlation"]],
                text=[f"{v:.3f}" for v in feat_df["Correlation"]],
                textposition="outside",
            ))
            fig_feat.add_vline(x=0, line_color="#6b7280")
            feat_theme = {**PLOT_THEME,
                          "xaxis": {**PLOT_THEME.get("xaxis", {}), "range": [-1, 1]}}
            fig_feat.update_layout(title="Pearson Correlation of Indicators vs Next-Day Close",
                                    height=350, **feat_theme)
            st.plotly_chart(fig_feat, use_container_width=True)
    else:
        st.warning("Not enough data for evaluation. Try a longer period (1mo or more).")


# TAB 3 — News Feed

with tab3:
    if not df_news.empty:
        c_feed, c_gauge = st.columns([2, 1])

        with c_gauge:
            st.markdown('<div class="section-header">Sentiment Breakdown</div>', unsafe_allow_html=True)
            counts = df_news["Sentiment"].value_counts()
            fig_donut = go.Figure(go.Pie(
                labels=counts.index, values=counts.values, hole=0.65,
                marker_colors=["#22c55e" if l=="Positive" else "#ef4444" if l=="Negative" else "#f59e0b"
                               for l in counts.index],
                textinfo="label+percent", textfont_size=12,
            ))
            fig_donut.update_layout(
                showlegend=False, height=220,
                annotations=[dict(text=f"{len(df_news)}<br>articles", x=0.5, y=0.5,
                                  font_size=14, showarrow=False, font_color="#f1f5f9")],
                **PLOT_THEME
            )
            st.plotly_chart(fig_donut, use_container_width=True)

            if "published" in df_news.columns:
                try:
                    df_time = df_news.copy()
                    df_time["date"] = pd.to_datetime(df_time["published"], errors="coerce")
                    df_time = df_time.dropna(subset=["date"])
                    daily = df_time.groupby(["date","Sentiment"]).size().unstack(fill_value=0).reset_index()
                    fig_bar = go.Figure()
                    for col in [c for c in ["Positive","Neutral","Negative"] if c in daily.columns]:
                        fig_bar.add_trace(go.Bar(x=daily["date"], y=daily[col], name=col,
                                                  marker_color={"Positive":"#22c55e","Neutral":"#f59e0b","Negative":"#ef4444"}[col]))
                    fig_bar.update_layout(barmode="stack", height=200, title="Sentiment Over Time", **PLOT_THEME)
                    st.plotly_chart(fig_bar, use_container_width=True)
                except Exception:
                    pass

        with c_feed:
            st.markdown('<div class="section-header">News Articles</div>', unsafe_allow_html=True)
            for _, row in df_news.head(30).iterrows():
                b   = badge_html(row["Sentiment"])
                src = row.get("source", "Unknown")
                pub = row.get("published", "")
                st.markdown(f"""
                <div class="news-card">
                  <div class="news-title">{row['title']} {b}</div>
                  <div class="news-meta">📌 {src} &nbsp;·&nbsp; 📅 {pub}</div>
                  <div class="news-meta" style="margin-top:4px;color:#475569;">{row['Summary']}</div>
                </div>""", unsafe_allow_html=True)

        csv = df_news.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download News CSV", data=csv, file_name="stock_news.csv", mime="text/csv")
    else:
        st.info("No news articles match the current filter.")



# TAB 4 — Multi-Stock Comparison

with tab4:
    compare_stocks = st.multiselect("Select stocks to compare", STOCKS, default=[ticker, "AAPL"])
    compare_period = st.radio("Period", ["1mo","3mo","6mo","1y"], horizontal=True, index=0)
    normalize      = st.toggle("Normalize to 100 (indexed performance)", value=True)

    if compare_stocks:
        colors = ["#6366f1","#22c55e","#f59e0b","#ef4444","#ec4899","#14b8a6"]

        fig_cmp = go.Figure()
        for i, s in enumerate(compare_stocks):
            d = load_stock(s, compare_period)
            if d.empty: continue
            c2 = d["Close"].squeeze()
            y  = (c2 / c2.iloc[0]) * 100 if normalize else c2
            fig_cmp.add_trace(go.Scatter(x=d.index, y=y, name=s,
                                          line=dict(color=colors[i % len(colors)], width=2)))
        fig_cmp.update_layout(
            title="Normalized Performance (base = 100)" if normalize else "Closing Prices",
            height=450,
            legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"),
            **PLOT_THEME
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Summary table
        rows_t = []
        for s in compare_stocks:
            d = load_stock(s, compare_period)
            if d.empty: continue
            c2  = d["Close"].squeeze()
            ret = ((c2.iloc[-1] - c2.iloc[0]) / c2.iloc[0]) * 100
            vol = c2.pct_change().std() * np.sqrt(252) * 100
            sharpe = (c2.pct_change().mean() / c2.pct_change().std()) * np.sqrt(252)
            max_dd = ((c2 / c2.cummax()) - 1).min() * 100
            rows_t.append({
                "Ticker": s,
                "Return (%)": f"{ret:+.2f}%",
                "Ann. Vol (%)": f"{vol:.1f}%",
                "Sharpe (approx)": f"{sharpe:.2f}",
                "Max Drawdown": f"{max_dd:.1f}%",
                "Current ($)": f"${c2.iloc[-1]:.2f}",
            })
        if rows_t:
            st.dataframe(pd.DataFrame(rows_t).set_index("Ticker"), use_container_width=True)

        # Volume comparison
        st.markdown('<div class="section-header">Average Daily Volume Comparison</div>', unsafe_allow_html=True)
        vol_data = {}
        for s in compare_stocks:
            d = load_stock(s, compare_period)
            if not d.empty:
                vol_data[s] = float(d["Volume"].squeeze().mean())
        if vol_data:
            fig_vol = go.Figure(go.Bar(
                x=list(vol_data.keys()), y=list(vol_data.values()),
                marker_color=colors[:len(vol_data)], opacity=0.85,
                text=[f"{v/1e6:.1f}M" for v in vol_data.values()], textposition="outside"
            ))
            fig_vol.update_layout(title="Avg Daily Volume", height=300, **PLOT_THEME)
            st.plotly_chart(fig_vol, use_container_width=True)



# TAB 5 — Correlation Heatmap

with tab5:
    st.markdown("Correlation of daily returns across all tracked stocks (1 month)")
    all_data = load_all_stocks("1mo")
    if not all_data.empty:
        returns = all_data.pct_change().dropna()
        corr    = returns.corr()

        fig_heat = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale=[[0,"#ef4444"],[0.5,"#1e2230"],[1,"#22c55e"]],
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            textfont=dict(size=14, color="#f1f5f9"), hoverongaps=False,
        ))
        fig_heat.update_layout(title="Return Correlation Matrix", height=450, **PLOT_THEME)
        st.plotly_chart(fig_heat, use_container_width=True)

        st.caption("Values close to **+1** move together · **-1** move opposite · **0** uncorrelated")

        # Rolling correlation
        if len(compare_stocks if 'compare_stocks' in dir() else []) >= 2:
            s1, s2 = compare_stocks[0], compare_stocks[1]
            if s1 in returns.columns and s2 in returns.columns:
                roll = returns[s1].rolling(10).corr(returns[s2])
                fig_roll = go.Figure(go.Scatter(
                    x=roll.index, y=roll, fill="tozeroy",
                    line=dict(color="#6366f1", width=2),
                    fillcolor="rgba(99,102,241,0.1)"
                ))
                fig_roll.add_hline(y=0, line_color="#6b7280", line_dash="dot")
                fig_roll.update_layout(title=f"10-day Rolling Correlation: {s1} vs {s2}",
                                        height=300, **PLOT_THEME)
                st.plotly_chart(fig_roll, use_container_width=True)
    else:
        st.warning("Could not load data for correlation analysis.")



# TAB 6 — Returns & Risk Analysis

with tab6:
    if not stock_data.empty and "Returns" in stock_data.columns:
        rets = stock_data["Returns"].dropna()
        close = stock_data["Close"].squeeze()

        st.markdown('<div class="section-header">Return Distribution</div>', unsafe_allow_html=True)
        rd1, rd2 = st.columns(2)

        with rd1:
            fig_retdist = go.Figure()
            fig_retdist.add_trace(go.Histogram(
                x=rets * 100, nbinsx=30,
                marker_color="#6366f1", opacity=0.8, name="Daily Returns (%)"
            ))
            # Normal overlay
            mu, sigma = float(rets.mean() * 100), float(rets.std() * 100)
            x_norm = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
            y_norm = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_norm - mu)/sigma)**2)
            y_norm = y_norm * len(rets) * (rets.std() * 100) * 2
            fig_retdist.add_trace(go.Scatter(x=x_norm, y=y_norm, name="Normal Fit",
                                              line=dict(color="#22c55e", width=2)))
            fig_retdist.add_vline(x=0, line_color="#6b7280", line_dash="dot")
            fig_retdist.update_layout(title="Daily Returns Distribution", height=340, **PLOT_THEME)
            st.plotly_chart(fig_retdist, use_container_width=True)

        with rd2:
            # Q-Q Plot approximation via sorted returns vs theoretical
            sorted_rets = np.sort(rets.values)
            n = len(sorted_rets)
            theoretical = np.array([float(np.percentile(np.random.normal(0,1,10000), 100*i/n)) for i in range(1, n+1)])
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(x=theoretical, y=sorted_rets * 100,
                                         mode="markers", marker=dict(color="#6366f1", size=4, opacity=0.7), name="Returns"))
            lim = max(abs(theoretical.min()), abs(theoretical.max()))
            fig_qq.add_trace(go.Scatter(x=[-lim, lim], y=[-lim*sigma, lim*sigma],
                                         line=dict(color="#22c55e", dash="dot"), name="Normal Line"))
            fig_qq.update_layout(title="Q-Q Plot (Normality Check)",
                                  xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles (%)",
                                  height=340, **PLOT_THEME)
            st.plotly_chart(fig_qq, use_container_width=True)

        # Drawdown
        st.markdown('<div class="section-header">Drawdown Analysis</div>', unsafe_allow_html=True)
        cum_ret  = (1 + rets).cumprod()
        roll_max = cum_ret.cummax()
        drawdown = (cum_ret / roll_max - 1) * 100

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.values,
            fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
            line=dict(color="#ef4444", width=1.5), name="Drawdown (%)"
        ))
        max_dd_val = float(drawdown.min())
        fig_dd.add_hline(y=max_dd_val, line_dash="dot", line_color="#f87171",
                          annotation_text=f"Max DD: {max_dd_val:.1f}%",
                          annotation_position="bottom right")
        fig_dd.update_layout(title="Portfolio Drawdown (%)", height=300, **PLOT_THEME)
        st.plotly_chart(fig_dd, use_container_width=True)

        # Rolling Volatility
        st.markdown('<div class="section-header">Annualized Rolling Volatility</div>', unsafe_allow_html=True)
        if "RollingVol" in stock_data.columns:
            rvol = stock_data["RollingVol"].dropna()
            fig_rvol = go.Figure()
            fig_rvol.add_trace(go.Scatter(
                x=rvol.index, y=rvol.values,
                fill="tozeroy", fillcolor="rgba(99,102,241,0.1)",
                line=dict(color="#6366f1", width=2), name="20d Ann. Vol (%)"
            ))
            fig_rvol.add_hline(y=float(rvol.mean()), line_dash="dot", line_color="#f59e0b",
                                annotation_text=f"Avg: {rvol.mean():.1f}%")
            fig_rvol.update_layout(title="20-Day Annualized Rolling Volatility (%)",
                                    height=280, **PLOT_THEME)
            st.plotly_chart(fig_rvol, use_container_width=True)

        # Risk metrics summary
        st.markdown('<div class="section-header">Risk Summary</div>', unsafe_allow_html=True)
        var_95  = float(np.percentile(rets, 5) * 100)
        cvar_95 = float(rets[rets <= np.percentile(rets, 5)].mean() * 100)
        sharpe  = float((rets.mean() / rets.std()) * np.sqrt(252))
        ann_ret = float(((1 + rets.mean()) ** 252 - 1) * 100)

        rm1, rm2, rm3, rm4, rm5 = st.columns(5)
        risk_cards = [
            (rm1, "Ann. Return",    f"{ann_ret:+.1f}%",  "#22c55e" if ann_ret > 0 else "#ef4444"),
            (rm2, "Sharpe Ratio",   f"{sharpe:.2f}",     "#22c55e" if sharpe > 1 else "#f59e0b" if sharpe > 0 else "#ef4444"),
            (rm3, "Max Drawdown",   f"{max_dd_val:.1f}%","#ef4444"),
            (rm4, "VaR 95%",        f"{var_95:.2f}%",    "#f59e0b"),
            (rm5, "CVaR 95%",       f"{cvar_95:.2f}%",   "#ef4444"),
        ]
        for col, label, val, color in risk_cards:
            with col:
                st.markdown(f"""
                <div class="eval-card">
                  <div class="eval-metric-label">{label}</div>
                  <div class="eval-metric-value" style="color:{color};">{val}</div>
                </div>""", unsafe_allow_html=True)
    else:
        st.warning("Not enough data for return/risk analysis.")

#Footer 
st.markdown("---")
st.caption("AI-Powered Financial News & Stock Trend Analysis · Data via Yahoo Finance · For informational purposes only.")