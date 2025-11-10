# app.py — Streamlit Stock Analysis (ML + TA)
# pip install -q streamlit yfinance pandas-ta lightgbm optuna plotly google-generativeai

import os
import warnings
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import lightgbm as lgb
import optuna
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

import streamlit as st

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Helper function to get API key from various sources
def get_gemini_api_key():
    """Get Gemini API key from Streamlit secrets, env var, or return empty string"""
    # Try Streamlit secrets first (for cloud deployment)
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except (FileNotFoundError, KeyError):
        # Fallback to environment variable (for local development)
        return os.getenv("GEMINI_API_KEY", "")

# ============== Streamlit Setup ==============
st.set_page_config(page_title="Brutal Stock Tool v10 (ML + TA)", layout="wide")
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.ERROR)
pd.options.display.float_format = "{:.5f}".format

# ============== Optional Gemini ==============
try:
    import google.generativeai as genai
except Exception:
    genai = None


# =========================
# Util umum: sanitasi OHLCV
# =========================
def _flatten_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        if set(["Open", "High", "Low", "Close", "Volume", "Adj Close"]).issubset(
            set(out.columns.get_level_values(0))
        ):
            out.columns = out.columns.get_level_values(0)
        else:
            out.columns = [
                "_".join([str(x) for x in tup if x is not None]).strip("_")
                for tup in out.columns
            ]
    return out


def _sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = _flatten_ohlcv_columns(df)
    rename_map = {c: c.title() for c in df.columns}
    df = df.rename(columns=rename_map)

    cols = [
        c
        for c in ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
        if c in df.columns
    ]
    df = df[cols].copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, errors="coerce")
        except Exception:
            pass
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    for c in df.columns:
        ser = df[c]
        if isinstance(ser, pd.DataFrame):
            ser = ser.iloc[:, 0]
        df[c] = pd.to_numeric(ser, errors="coerce")

    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0)

    need = [x for x in ["Open", "High", "Low", "Close"] if x in df.columns]
    if need:
        df = df.dropna(subset=need)

    return df


# ==============
# Util: Tick BEI
# ==============
def round_to_idx_tick_size(price: float) -> int:
    price = float(price)
    if price < 200:
        tick = 1
    elif price < 500:
        tick = 2
    elif price < 2000:
        tick = 5
    elif price < 5000:
        tick = 10
    else:
        tick = 25
    return int(round(price / tick) * tick)


# ==========================
# 1.1) ML Feature Engineering (ENHANCED)
# ==========================
def make_features_ml(
    df_merged_lower: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = df_merged_lower.copy()

    df.ta.rsi(14, append=True)
    df.ta.macd(append=True)
    df.ta.bbands(20, append=True)
    df.ta.atr(14, append=True)
    df.ta.mfi(14, append=True)
    df.ta.adx(14, append=True)
    df.ta.stoch(append=True)
    df.ta.ema(20, append=True)
    df.ta.ema(50, append=True)

    for lag in [1, 2, 3, 5, 10, 21]:
        df[f"ret_{lag}d"] = df["close"].pct_change(lag)

    for win in [5, 10, 20]:
        df[f"roll_mean_{win}"] = df["close"].rolling(win).mean()
        df[f"roll_std_{win}"] = df["close"].rolling(win).std()
        df[f"roll_min_{win}"] = df["close"].rolling(win).min()
        df[f"roll_max_{win}"] = df["close"].rolling(win).max()
        df[f"zscore_{win}"] = (df["close"] - df[f"roll_mean_{win}"]) / (
            df[f"roll_std_{win}"] + 1e-9
        )

    if "close_jkse" in df.columns:
        df["rel_str_jkse"] = df["close"] / (df["close_jkse"] + 1e-9)
        df["ret_jkse_1d"] = df["close_jkse"].pct_change(1)
        df["ret_jkse_5d"] = df["close_jkse"].pct_change(5)
    if "close_vix" in df.columns:
        df["ret_vix_1d"] = df["close_vix"].pct_change(1)

    df["target_return"] = df["close"].pct_change().shift(-1)

    df_for_pred = df.iloc[-2:].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    X = df.drop(columns=["target_return"])
    y = df["target_return"]
    if X.empty or y.empty:
        raise ValueError("Data tidak cukup setelah feature engineering ML.")
    return X, y, df_for_pred


# =====================================================
# 1.2) ML Walk-forward CV + Embargo + Optuna tuning
# =====================================================
def tune_and_backtest_ml(
    X: pd.DataFrame,
    y: pd.Series,
    embargo: int = 5,
    n_splits: int = 5,
    n_trials: int = 40,
):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scaler = StandardScaler()

    splits = []
    idx = np.arange(len(X))
    for tr, va in tscv.split(idx):
        if embargo > 0:
            va = va[embargo:] if len(va) > embargo else va
            if len(va) == 0:
                continue
        splits.append((tr, va))
    if not splits:
        raise ValueError("Tidak ada split validasi yang tersisa setelah embargo.")

    def objective(trial):
        params = {
            "objective": "regression_l1",
            "metric": "mae",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 120),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
            "seed": 42,
            "verbose": -1,
            "n_jobs": -1,
        }
        thr = trial.suggest_float("dir_threshold", -0.002, 0.002)

        wr = []
        for tr, va in splits:
            X_tr, X_va = X.iloc[tr], X.iloc[va]
            y_tr, y_va = y.iloc[tr], y.iloc[va]

            Xtr = scaler.fit_transform(X_tr)
            Xva = scaler.transform(X_va)

            dtr = lgb.Dataset(Xtr, label=y_tr)
            dva = lgb.Dataset(Xva, label=y_va, reference=dtr)

            model = lgb.train(
                params,
                dtr,
                valid_sets=[dva],
                num_boost_round=1500,
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            pred = model.predict(Xva, num_iteration=model.best_iteration)
            pred_dir = (pred > thr).astype(int)
            actual_dir = (y_va > 0).astype(int)
            wr.append(accuracy_score(actual_dir, pred_dir))
        return float(np.mean(wr))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params.copy()
    best_thr = best_params.pop("dir_threshold")
    best_wr = study.best_value
    return best_params, best_thr, best_wr, splits


# ==========================================
# 1.3) ML Train final, tables & plots
# ==========================================
def train_final_and_report_ml(X, y, df_for_pred, best_params, best_thr, splits, ticker):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    dtrain = lgb.Dataset(X_scaled, label=y)

    final_params = {
        "objective": "regression_l1",
        "metric": "mae",
        "seed": 42,
        "verbose": -1,
        "n_jobs": -1,
        **best_params,
    }
    model = lgb.train(final_params, dtrain, num_boost_round=800)

    tr, va = splits[-1]
    X_hold, y_hold = X.iloc[va], y.iloc[va]
    Xh = scaler.transform(X_hold)
    pred_ret = model.predict(Xh)
    close_t = X_hold["close"].values
    actual_price_t1 = close_t * (1 + y_hold.values)
    pred_price_t1_raw = close_t * (1 + pred_ret)
    pred_price_t1 = np.array([round_to_idx_tick_size(p) for p in pred_price_t1_raw])

    df_bt = pd.DataFrame(
        {
            "Date_t": X_hold.index,
            "close_t": close_t,
            "Pred_close_Tplus1": pred_price_t1,
            "Actual_close_Tplus1": actual_price_t1,
            "Pred_return_Tplus1": pred_ret,
            "Actual_return_Tplus1": y_hold.values,
        }
    )
    df_bt["Selisih"] = df_bt["Pred_close_Tplus1"] - df_bt["Actual_close_Tplus1"]
    df_bt["ArahPred"] = np.where(
        df_bt["Pred_return_Tplus1"] > best_thr, "NAIK", "TURUN"
    )
    df_bt["ArahActual"] = np.where(df_bt["Actual_return_Tplus1"] > 0, "NAIK", "TURUN")
    df_bt["Benar?"] = np.where(df_bt["ArahPred"] == df_bt["ArahActual"], "✅", "❌")
    wr_hold = (df_bt["Benar?"] == "✅").mean()

    recent = df_bt.tail(100)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent["Date_t"],
            y=recent["Actual_close_Tplus1"],
            mode="lines+markers",
            name="Harga Asli (T+1)",
            line=dict(width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=recent["Date_t"],
            y=recent["Pred_close_Tplus1"],
            mode="lines+markers",
            name="Prediksi Harga ML (T+1)",
            line=dict(width=2, dash="dot"),
        )
    )
    fig.update_layout(
        title=f"📈 {ticker} | Prediksi ML vs Aktual (100 titik terbaru)  |  Holdout Win-Rate: {wr_hold:.2%}",
        xaxis_title="Tanggal",
        yaxis_title="Harga (Rp)",
        template="plotly_dark",
        hovermode="x unified",
        height=420,
    )

    X_live = df_for_pred.drop(columns=["target_return"], errors="ignore").iloc[[-1]]
    X_live = X_live.ffill().bfill()
    Xl = scaler.transform(X_live)
    live_ret = model.predict(Xl)[0]
    last_close = X_live["close"].values[0]
    live_pred_price = round_to_idx_tick_size(last_close * (1 + live_ret))
    arah_live = "NAIK" if live_ret > best_thr else "TURUN"

    ml_metrics = {
        "Ticker": ticker,
        "winrate_holdout": wr_hold,
        "mae_holdout": mean_absolute_error(
            df_bt["Actual_close_Tplus1"], df_bt["Pred_close_Tplus1"]
        ),
        "r2_holdout": r2_score(
            df_bt["Actual_close_Tplus1"], df_bt["Pred_close_Tplus1"]
        ),
    }
    ml_prediction = {
        "Ticker": ticker,
        "ML_Close_T": last_close,
        "ML_Pred_Return_T+1": live_ret,
        "ML_Threshold_Arah": best_thr,
        "ML_Pred_Harga_T+1": live_pred_price,
        "ML_Arah": arah_live,
    }
    return ml_metrics, ml_prediction, df_bt, fig


# ==========================================================
# BLOK 2: FUNGSI-FUNGSI TA-AI
# ==========================================================
def compute_indicators_ta(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = _sanitize_ohlcv(df_raw)
    if df.empty:
        raise ValueError("Data kosong setelah sanitasi OHLC di compute_indicators_ta")

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    fast, slow, signal = 12, 26, 9
    min_bars = max(slow + signal, 35)
    if len(df) < min_bars:
        raise ValueError(
            f"Data terlalu pendek untuk MACD: butuh ≥{min_bars} bar, baru ada {len(df)}."
        )

    macd_df = ta.macd(close, fast=fast, slow=slow, signal=signal, talib=False)
    have_cols = False
    if macd_df is not None:
        need = {"MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9"}
        have_cols = need.issubset(set(macd_df.columns))

    if have_cols:
        df["MACD"] = macd_df["MACD_12_26_9"]
        df["MACD_signal"] = macd_df["MACDs_12_26_9"]
        df["MACD_hist"] = macd_df["MACDh_12_26_9"]
    else:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
        macd_hist = macd_line - macd_signal
        df["MACD"], df["MACD_signal"], df["MACD_hist"] = (
            macd_line,
            macd_signal,
            macd_hist,
        )

    df["RSI14"] = ta.rsi(close, length=14)
    df["ATR14"] = ta.atr(high, low, close, length=14)
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["VolSMA20"] = df["Volume"].rolling(20, min_periods=1).mean()

    return df.dropna()


def detect_breakout_levels_ta(df_raw, lookback=40):
    base = _sanitize_ohlcv(df_raw)
    if base.empty:
        raise ValueError("Data kosong di detect_breakout_levels_ta")

    base = base[["High", "Low"]].dropna()
    win = min(lookback, len(base))
    recent = base.tail(win)
    resistance = float(recent["High"].max())
    support = float(recent["Low"].min())
    height = resistance - support
    return resistance, support, height


def ta_generate_trade_plan(df_raw, ticker="TICKER", use_ai=False, gemini_key: str = ""):
    df_ind = compute_indicators_ta(df_raw)
    last = df_ind.dropna().iloc[-1]

    close_t = float(last["Close"])
    rsi = float(last["RSI14"])
    macd = float(last["MACD"])
    macd_signal = float(last["MACD_signal"])
    macd_hist = float(last["MACD_hist"])
    atr = float(last["ATR14"])
    ema20 = float(last["EMA20"])
    ema50 = float(last["EMA50"])
    vol = float(last["Volume"])
    vol_sma20 = float(last["VolSMA20"])

    resistance, support, height = detect_breakout_levels_ta(df_raw, lookback=40)

    bull_votes = 0
    bull_votes += 1 if ema20 > ema50 else 0
    bull_votes += 1 if macd > macd_signal else 0
    bull_votes += 1 if rsi > 50 else 0
    view = "BULLISH" if bull_votes >= 2 else "BEARISH"

    breakout_level = resistance
    tp1 = round_to_idx_tick_size(breakout_level + height)
    tp2 = round_to_idx_tick_size(breakout_level + 1.5 * height)
    entry_low = round_to_idx_tick_size(breakout_level - 0.3 * atr)
    entry_high = round_to_idx_tick_size(breakout_level + 0.3 * atr)
    stop_loss = round_to_idx_tick_size(max(1.0, breakout_level - 1.0 * atr))
    vol_note = (
        "Volume di atas rata-rata (SMA20)"
        if vol > vol_sma20
        else "Volume di bawah/sekitar rata-rata"
    )

    plan = {
        "Ticker": ticker,
        "Close_T": round_to_idx_tick_size(close_t),
        "View": view,
        "RSI14": round(rsi, 2),
        "MACD_hist": round(macd_hist, 4),
        "ATR14": round(atr, 2),
        "Volume_Note": vol_note,
        "Support": round_to_idx_tick_size(support),
        "Resistance/Breakout": round_to_idx_tick_size(breakout_level),
        "Entry_Zone": f"{entry_low} – {entry_high}",
        "Stop_Loss": stop_loss,
        "TP1": tp1,
        "TP2": tp2,
    }

    ai_comment = None
    if use_ai and genai is not None and gemini_key:
        try:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            prompt = f"""
Kamu seorang Certified Technical Analyst. Buat ringkasan padat & actionable untuk {ticker}.
Data:
- Close: {plan['Close_T']}, View: {plan['View']}
- RSI14: {plan['RSI14']} (70/30 rule)
- MACD Hist: {plan['MACD_hist']}
- ATR14: {plan['ATR14']}
- Volume: {plan['Volume_Note']}
- Support: {plan['Support']}, Breakout: {plan['Resistance/Breakout']}
- Entry zone: {plan['Entry_Zone']}, SL: {plan['Stop_Loss']}, TP1: {plan['TP1']}, TP2: {plan['TP2']}
Instruksi:
1) Jelaskan kondisi RSI/MACD/EMA & volume.
2) Jelaskan skenario breakout di {plan['Resistance/Breakout']}, validasi volume.
3) Beri alasan Entry Zone/SL/TP berdasarkan ATR/rectangle height.
4) Tutup dengan view {plan['View']} jangka pendek (1-2 minggu), disclaimers singkat.
Gunakan bahasa Indonesia, singkat, jelas, poin-poin.
"""
            resp = model.generate_content(prompt)
            ai_comment = resp.text.strip()
        except Exception as e:
            ai_comment = f"(AI ERROR) {e}. Pastikan API key valid & internet aktif."
    elif use_ai and (genai is None or not gemini_key):
        ai_comment = "(AI OFF) Library gemini tidak tersedia atau API Key kosong."

    return plan, ai_comment


def ta_plot_trade_plan(df_raw, plan):
    recent = _sanitize_ohlcv(df_raw).tail(120).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=recent.index,
            open=recent["Open"],
            high=recent["High"],
            low=recent["Low"],
            close=recent["Close"],
            name="Price",
        )
    )
    res = plan["Resistance/Breakout"]
    sup = plan["Support"]
    sl = plan["Stop_Loss"]
    tp1 = plan["TP1"]
    tp2 = plan["TP2"]
    try:
        e_low, e_high = [int(x.strip()) for x in plan["Entry_Zone"].split("–")]
    except ValueError:
        parts = plan["Entry_Zone"].replace("-", "–").split("–")
        e_low, e_high = int(parts[0].strip()), int(parts[1].strip())

    for level, name in [
        (res, "Breakout"),
        (sup, "Support"),
        (sl, "Stop-Loss"),
        (tp1, "TP1"),
        (tp2, "TP2"),
    ]:
        fig.add_hline(
            y=level,
            line_dash="dot" if name != "Breakout" else "solid",
            annotation_text=name,
            annotation_position="top right",
        )

    fig.add_hrect(
        y0=e_low,
        y1=e_high,
        fillcolor="rgba(0,200,0,0.10)",
        line_width=0,
        annotation_text="ENTRY ZONE",
        annotation_position="left",
    )

    fig.update_layout(
        title=f"Trade Plan {plan['Ticker']} | View: {plan['View']}",
        template="plotly_dark",
        hovermode="x unified",
        height=520,
        yaxis_title="Harga (Rp)",
    )
    return fig


# ==========================================================
# Data Fetchers (cached)
# ==========================================================
@st.cache_data(show_spinner=False)
def fetch_prices(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=pd.Timestamp.now() + pd.Timedelta(days=1),
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    return df


@st.cache_data(show_spinner=False)
def fetch_external_series(
    start: str, include_ihsg: bool, include_vix: bool
) -> pd.DataFrame:
    tickers = []
    names = []
    if include_ihsg:
        tickers.append("^JKSE")
        names.append("close_jkse")
    if include_vix:
        tickers.append("^VIX")
        names.append("close_vix")

    if not tickers:
        return pd.DataFrame()

    ext_raw = yf.download(
        tickers,
        start=start,
        end=pd.Timestamp.now() + pd.Timedelta(days=1),
        auto_adjust=False,
        progress=False,
    )
    if "Close" in ext_raw.columns:
        ext = ext_raw["Close"]
    else:
        ext = ext_raw.copy()
    if isinstance(ext, pd.Series):
        ext = ext.to_frame()
    # Keep only tickers in same order
    cols = []
    for t in tickers:
        if t in ext.columns:
            cols.append(t)
        elif (t,) in ext.columns:  # edge MultiIndex collapsed
            cols.append((t,))
    ext = ext[cols]
    ext.columns = names
    ext = ext.ffill()
    return ext


# ==========================================================
# Runners per ticker
# ==========================================================
def run_ml_for_ticker(
    ticker: str,
    df_capital: pd.DataFrame,
    df_ext: pd.DataFrame,
    n_trials: int,
    embargo: int,
    n_splits: int,
):
    df_lower = df_capital[["Open", "High", "Low", "Close", "Volume"]].copy()
    df_lower.columns = ["open", "high", "low", "close", "volume"]
    df_merged_std = df_lower.join(df_ext, how="left").ffill()

    X, y, df_for_pred = make_features_ml(df_merged_std)
    best_params, best_thr, best_wr, splits = tune_and_backtest_ml(
        X, y, embargo=embargo, n_splits=n_splits, n_trials=n_trials
    )
    metrics, prediction, df_bt, fig = train_final_and_report_ml(
        X, y, df_for_pred, best_params, best_thr, splits, ticker
    )
    return metrics, prediction, df_bt, fig, best_wr


def run_ta_for_ticker(
    ticker: str, df_capital: pd.DataFrame, use_ai: bool, gemini_key: str
):
    plan, ai_comment = ta_generate_trade_plan(
        df_capital, ticker=ticker, use_ai=use_ai, gemini_key=gemini_key
    )
    fig = ta_plot_trade_plan(df_capital, plan)
    return plan, ai_comment, fig


# ==========================================================
# =============== UI ==================
# ==========================================================
st.title("🤖 Brutal Stock Tool v10 — ML Enhanced + TA AI")
with st.sidebar:
    st.header("Pengaturan")
    tickers_input = st.text_input(
        "Ticker (pisah spasi/koma)",
        value="BBCA, BBRI, BMRI",
        help="Contoh: BBCA, BBRI, BREN",
    )
    auto_append_jk = st.checkbox("Auto-append .JK", value=True)
    start_date = st.date_input(
        "Start Date", value=pd.to_datetime("2015-01-01")
    ).strftime("%Y-%m-%d")

    st.markdown("---")
    run_ml = st.checkbox("Jalankan ML Quant", value=True)
    include_ihsg = st.checkbox("Tambahkan fitur IHSG (^JKSE)", value=True)
    include_vix = st.checkbox("Tambahkan fitur VIX (^VIX)", value=False)
    n_trials = st.slider("Optuna Trials (ML)", 5, 60, 30, step=5)
    n_splits = st.slider("TimeSeries Splits", 3, 10, 5)
    embargo = st.slider("Embargo (bar)", 0, 10, 5)

    st.markdown("---")
    run_ta = st.checkbox("Jalankan TA + Trade Plan", value=True)
    use_ai = st.checkbox("Komentar AI Gemini", value=False)

    # Get API key from Streamlit secrets, env var, or user input
    if use_ai:
        gemini_key = st.text_input(
            "GEMINI_API_KEY (opsional)",
            value="",
            type="password",
            help="Masukkan API key Gemini Anda, atau kosongkan untuk auto-load dari secrets/env"
        )
        # Jika user tidak input, gunakan helper function untuk auto-load
        if not gemini_key:
            gemini_key = get_gemini_api_key()
    else:
        gemini_key = ""

    st.markdown("---")
    run_btn = st.button("🚀 RUN")

if run_btn:
    raw_tokens = [
        t.strip().upper() for t in tickers_input.replace(",", " ").split() if t.strip()
    ]
    tickers: List[str] = []
    for t in raw_tokens:
        if auto_append_jk and not t.endswith(".JK"):
            tickers.append(f"{t}.JK")
        else:
            tickers.append(t)

    st.write(f"**Akan memproses:** {', '.join(tickers)}")

    # External series (once)
    df_ext = pd.DataFrame()
    if run_ml and (include_ihsg or include_vix):
        with st.status("Mengunduh data eksternal (IHSG/VIX)...", expanded=False):
            try:
                df_ext = fetch_external_series(
                    start=start_date, include_ihsg=include_ihsg, include_vix=include_vix
                )
                st.write("OK.")
            except Exception as e:
                st.warning(
                    f"Gagal mengunduh data eksternal: {e}. ML jalan tanpa konteks."
                )
                df_ext = pd.DataFrame()

    ml_predictions: List[Dict[str, Any]] = []
    ml_holdout_metrics: List[Dict[str, Any]] = []

    for t in tickers:
        st.subheader(f"==> {t}")
        with st.status(f"Mengunduh data harga {t} ...", expanded=False) as s:
            df_raw = fetch_prices(t, start=start_date)
            if df_raw.empty:
                s.update(state="error", label=f"Data yfinance kosong untuk {t}. Skip.")
                continue
            df_capital = _sanitize_ohlcv(df_raw)
            s.update(state="complete", label=f"Data {t} siap.")

        col1, col2 = st.columns(2)

        if run_ml:
            with col1:
                st.write("### ML Quant")
                try:
                    with st.spinner("Training & backtest ML ..."):
                        metrics, pred, df_bt, fig_ml, cv_wr = run_ml_for_ticker(
                            t,
                            df_capital,
                            df_ext,
                            n_trials=n_trials,
                            embargo=embargo,
                            n_splits=n_splits,
                        )
                    st.plotly_chart(fig_ml, use_container_width=True)
                    st.dataframe(
                        df_bt.tail(15)[
                            [
                                "Date_t",
                                "close_t",
                                "Pred_close_Tplus1",
                                "Actual_close_Tplus1",
                                "Selisih",
                                "ArahPred",
                                "ArahActual",
                                "Benar?",
                            ]
                        ],
                        use_container_width=True,
                    )
                    ml_predictions.append(pred)
                    ml_holdout_metrics.append(metrics)
                    st.success(
                        f"CV Best WR: **{cv_wr:.2%}** | Holdout WR: **{metrics['winrate_holdout']:.2%}** | MAE: {metrics['mae_holdout']:.0f}"
                    )
                except Exception as e:
                    st.error(f"ML gagal untuk {t}: {e}")

        if run_ta:
            with col2:
                st.write("### TA + Trade Plan")
                try:
                    plan, ai_comment, fig_ta = run_ta_for_ticker(
                        t, df_capital, use_ai=use_ai, gemini_key=gemini_key
                    )
                    st.plotly_chart(fig_ta, use_container_width=True)
                    st.dataframe(
                        pd.DataFrame([plan]).T.rename(columns={0: "Value"}),
                        use_container_width=True,
                    )
                    if ai_comment:
                        st.info(ai_comment)
                except Exception as e:
                    st.error(f"TA gagal untuk {t}: {e}")

    st.markdown("---")
    if run_ml and ml_predictions:
        st.write("## Ringkasan ML — Prediksi T+1")
        df_pred = pd.DataFrame(ml_predictions).set_index("Ticker")
        fmt_pred = df_pred.copy()
        if "ML_Close_T" in fmt_pred.columns:
            fmt_pred["ML_Close_T"] = fmt_pred["ML_Close_T"].round(0)
        if "ML_Pred_Harga_T+1" in fmt_pred.columns:
            fmt_pred["ML_Pred_Harga_T+1"] = fmt_pred["ML_Pred_Harga_T+1"].round(0)
        if "ML_Pred_Return_T+1" in fmt_pred.columns:
            fmt_pred["ML_Pred_Return_T+1"] = (
                fmt_pred["ML_Pred_Return_T+1"] * 100
            ).round(2)
        if "ML_Threshold_Arah" in fmt_pred.columns:
            fmt_pred["ML_Threshold_Arah"] = (fmt_pred["ML_Threshold_Arah"] * 100).round(
                2
            )
        st.dataframe(fmt_pred, use_container_width=True)

        st.write("## Ringkasan ML — Metrik Holdout")
        df_metrics = pd.DataFrame(ml_holdout_metrics).set_index("Ticker")
        fmt_met = df_metrics.copy()
        if "winrate_holdout" in fmt_met.columns:
            fmt_met["winrate_holdout"] = (fmt_met["winrate_holdout"] * 100).round(2)
        if "mae_holdout" in fmt_met.columns:
            fmt_met["mae_holdout"] = fmt_met["mae_holdout"].round(0)
        if "r2_holdout" in fmt_met.columns:
            fmt_met["r2_holdout"] = fmt_met["r2_holdout"].round(3)
        st.dataframe(fmt_met, use_container_width=True)

else:
    st.info("Masukkan ticker di sidebar, pilih opsi, lalu klik **RUN**.")
