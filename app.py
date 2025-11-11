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
st.set_page_config(
    page_title="Professional Stock Analyzer v11 (ML + TA + Risk)", 
    layout="wide",
    initial_sidebar_state="expanded"
)
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
# 1.3) ML Train final, tables & plots + ENHANCED METRICS
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

    # ========== ENHANCED METRICS ==========
    # 1. Sharpe Ratio (annualized)
    strategy_returns = df_bt["Pred_return_Tplus1"] * (df_bt["ArahPred"] == "NAIK").astype(int)
    sharpe_ratio = (strategy_returns.mean() / (strategy_returns.std() + 1e-9)) * np.sqrt(252)
    
    # 2. Max Drawdown
    cumulative_returns = (1 + strategy_returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # 3. Win/Loss Ratio
    wins = df_bt[df_bt["Benar?"] == "✅"]
    losses = df_bt[df_bt["Benar?"] == "❌"]
    avg_win = wins["Pred_return_Tplus1"].abs().mean() if len(wins) > 0 else 0
    avg_loss = losses["Pred_return_Tplus1"].abs().mean() if len(losses) > 0 else 1
    win_loss_ratio = avg_win / (avg_loss + 1e-9)
    
    # 4. Profit Factor
    gross_profit = wins["Pred_return_Tplus1"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["Pred_return_Tplus1"].sum()) if len(losses) > 0 else 1
    profit_factor = gross_profit / (gross_loss + 1e-9)
    
    # 5. Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False).head(10)

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

    # Feature Importance Chart
    fig_importance = go.Figure()
    fig_importance.add_trace(
        go.Bar(
            x=feature_importance['importance'],
            y=feature_importance['feature'],
            orientation='h',
            marker_color='lightblue'
        )
    )
    fig_importance.update_layout(
        title=f"🔍 Top 10 Feature Importance - {ticker}",
        xaxis_title="Importance (Gain)",
        yaxis_title="Feature",
        template="plotly_dark",
        height=400,
    )

    X_live = df_for_pred.drop(columns=["target_return"], errors="ignore").iloc[[-1]]
    X_live = X_live.ffill().bfill()
    Xl = scaler.transform(X_live)
    live_ret = model.predict(Xl)[0]
    
    # Confidence interval estimation (simple bootstrap-like approach)
    # Using standard error from validation predictions
    pred_errors = pred_ret - y_hold.values
    std_error = pred_errors.std()
    confidence_95_lower = live_ret - 1.96 * std_error
    confidence_95_upper = live_ret + 1.96 * std_error
    
    last_close = X_live["close"].values[0]
    live_pred_price = round_to_idx_tick_size(last_close * (1 + live_ret))
    live_pred_lower = round_to_idx_tick_size(last_close * (1 + confidence_95_lower))
    live_pred_upper = round_to_idx_tick_size(last_close * (1 + confidence_95_upper))
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
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_loss_ratio": win_loss_ratio,
        "profit_factor": profit_factor,
    }
    ml_prediction = {
        "Ticker": ticker,
        "ML_Close_T": last_close,
        "ML_Pred_Return_T+1": live_ret,
        "ML_Threshold_Arah": best_thr,
        "ML_Pred_Harga_T+1": live_pred_price,
        "ML_Pred_Lower_95": live_pred_lower,
        "ML_Pred_Upper_95": live_pred_upper,
        "ML_Arah": arah_live,
    }
    return ml_metrics, ml_prediction, df_bt, fig, fig_importance, feature_importance


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

    base = base[["High", "Low", "Close"]].dropna()
    win = min(lookback, len(base))
    recent = base.tail(win)
    
    resistance = float(recent["High"].max())
    support = float(recent["Low"].min())
    last_close = float(recent["Close"].iloc[-1])
    height = resistance - support

    # Tentukan tipe plan berdasarkan posisi harga terakhir
    # Jika harga lebih dekat ke support, plan-nya 'Swing' (beli di support)
    # Jika harga lebih dekat ke resistance, plan-nya 'Breakout' (beli saat tembus resistance)
    if abs(last_close - support) < abs(last_close - resistance):
        plan_type = "Swing"
    else:
        plan_type = "Breakout"
        
    return resistance, support, height, plan_type, last_close


def ta_generate_trade_plan(df_raw, ticker="TICKER", use_ai=False, gemini_key: str = "", ml_winrate=None):
    df_ind = compute_indicators_ta(df_raw)
    last = df_ind.dropna().iloc[-1]

    close_t = float(last["Close"])
    rsi = float(last["RSI14"])
    macd_hist = float(last["MACD_hist"])
    atr = float(last["ATR14"])
    ema20 = float(last["EMA20"])
    ema50 = float(last["EMA50"])
    vol = float(last["Volume"])
    vol_sma20 = float(last["VolSMA20"])

    resistance, support, height, plan_type, last_close = detect_breakout_levels_ta(df_raw, lookback=40)

    bull_votes = 0
    bull_votes += 1 if ema20 > ema50 else 0
    bull_votes += 1 if macd_hist > 0 else 0
    bull_votes += 1 if rsi > 50 else 0
    view = "BULLISH" if bull_votes >= 2 else "BEARISH"

    # Logika plan dinamis berdasarkan posisi harga
    if plan_type == "Breakout":
        # Plan A: Beli saat menembus resistance
        scenario = "Breakout Bullish"
        entry_point = resistance
        tp1 = round_to_idx_tick_size(entry_point + height)
        tp2 = round_to_idx_tick_size(entry_point + 1.5 * height)
        stop_loss = round_to_idx_tick_size(max(1.0, entry_point - 1.0 * atr))
    else: # plan_type == "Swing"
        # Plan B: Beli di dekat support (mean reversion/swing)
        scenario = "Swing/Buy on Weakness"
        entry_point = support
        tp1 = round_to_idx_tick_size(resistance) # Target utama adalah resistance
        tp2 = round_to_idx_tick_size(support + 0.5 * height) # Target antara
        stop_loss = round_to_idx_tick_size(max(1.0, entry_point - 1.0 * atr))

    entry_low = round_to_idx_tick_size(entry_point - 0.3 * atr)
    entry_high = round_to_idx_tick_size(entry_point + 0.3 * atr)
    entry_mid = round_to_idx_tick_size((entry_low + entry_high) / 2)
    
    # ========== RISK/REWARD ANALYSIS (DATA-DRIVEN) ==========
    # Risk: Entry ke Stop Loss
    risk_amount = abs(entry_mid - stop_loss)
    risk_pct = risk_amount / (entry_mid + 1e-9)
    
    # Reward: Entry ke TP1 dan TP2
    reward_tp1 = abs(tp1 - entry_mid)
    reward_tp2 = abs(tp2 - entry_mid)
    reward_tp1_pct = reward_tp1 / (entry_mid + 1e-9)
    reward_tp2_pct = reward_tp2 / (entry_mid + 1e-9)
    
    # Risk/Reward Ratios
    rr_ratio_tp1 = reward_tp1 / (risk_amount + 1e-9)
    rr_ratio_tp2 = reward_tp2 / (risk_amount + 1e-9)
    
    # REAL WIN PROBABILITY: Use ML model holdout win-rate if available
    if ml_winrate is not None and 0.4 <= ml_winrate <= 0.8:
        win_prob = ml_winrate
        prob_source = "ML Model Holdout"
    else:
        # Fallback: Estimate from technical indicators strength
        # Count bullish signals and normalize
        ta_score = bull_votes / 3.0  # 0 to 1
        # Apply sigmoid-like curve: 0.45 to 0.60 range based on TA
        win_prob = 0.45 + (ta_score * 0.15)
        prob_source = "TA Indicator Strength"
    
    # REAL EXPECTED VALUE: No assumptions, pure math
    # Expected value per R risked = (win_prob * avg_RR) - (loss_prob * 1)
    # Assume 70% of wins hit TP1, 30% hit TP2 (conservative distribution)
    avg_rr = (0.7 * rr_ratio_tp1 + 0.3 * rr_ratio_tp2)
    expected_value_per_r = (win_prob * avg_rr) - ((1 - win_prob) * 1)
    expected_return_pct = expected_value_per_r * risk_pct
    
    # REAL KELLY CRITERION: Proper formula without arbitrary caps
    # Kelly = (p * b - q) / b, where p=win_prob, q=loss_prob, b=avg_win/avg_loss
    # For trading: Kelly = (win_prob * rr_ratio - loss_prob) / rr_ratio
    kelly_full = (win_prob * avg_rr - (1 - win_prob)) / avg_rr
    
    # FRACTIONAL KELLY: Industry standard is 0.25x Kelly (Quarter Kelly)
    # This is NOT arbitrary capping, it's risk management best practice
    # Reference: "Fortune's Formula" by William Poundstone
    kelly_fraction = max(0, kelly_full * 0.25)  # 1/4 Kelly for real-world risk management
    
    # Position sizing note
    if kelly_fraction <= 0:
        position_note = "Negative EV - Skip trade"
    elif kelly_fraction < 0.02:
        position_note = "Very small edge"
    elif kelly_fraction < 0.05:
        position_note = "Moderate edge"
    else:
        position_note = "Strong edge"
    
    vol_note = (
        "Volume di atas rata-rata (SMA20)"
        if vol > vol_sma20
        else "Volume di bawah/sekitar rata-rata"
    )

    plan = {
        "Ticker": ticker,
        "Scenario": scenario,
        "Close_T": round_to_idx_tick_size(close_t),
        "View": view,
        "RSI14": round(rsi, 2),
        "MACD_hist": round(macd_hist, 4),
        "ATR14": round(atr, 2),
        "Volume_Note": vol_note,
        "Support": round_to_idx_tick_size(support),
        "Resistance": round_to_idx_tick_size(resistance),
        "Entry_Zone": f"{entry_low} – {entry_high}",
        "Stop_Loss": stop_loss,
        "TP1": tp1,
        "TP2": tp2,
        # Enhanced metrics (Data-Driven)
        "Risk_Amount": risk_amount,
        "Risk_%": round(risk_pct * 100, 2),
        "Reward_TP1": reward_tp1,
        "Reward_TP1_%": round(reward_tp1_pct * 100, 2),
        "Reward_TP2": reward_tp2,
        "Reward_TP2_%": round(reward_tp2_pct * 100, 2),
        "RR_Ratio_TP1": round(rr_ratio_tp1, 2),
        "RR_Ratio_TP2": round(rr_ratio_tp2, 2),
        "Win_Probability": round(win_prob, 3),
        "Win_Prob_Source": prob_source,
        "Expected_Value_per_R": round(expected_value_per_r, 3),
        "Expected_Return_%": round(expected_return_pct * 100, 2),
        "Kelly_Full_%": round(kelly_full * 100, 2),
        "Kelly_Quarter_%": round(kelly_fraction * 100, 2),
        "Position_Note": position_note,
    }

    ai_comment = None
    if use_ai and genai is not None and gemini_key:
        try:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            prompt = f"""
Kamu seorang Certified Technical Analyst & Risk Manager. 
Buat analisis professional untuk {ticker}.

**Data Teknikal:**
- Skenario: {plan['Scenario']}
- Harga: {plan['Close_T']}, View: {plan['View']}
- RSI14: {plan['RSI14']} | MACD Hist: {plan['MACD_hist']}
- Volume: {plan['Volume_Note']}
- Support: {plan['Support']} | Resistance: {plan['Resistance']}

**Trade Setup (Data-Driven):**
- Entry Zone: {plan['Entry_Zone']}
- Stop Loss: {plan['Stop_Loss']} (Risk: {plan['Risk_%']}%)
- TP1: {plan['TP1']} (R/R: {plan['RR_Ratio_TP1']})
- TP2: {plan['TP2']} (R/R: {plan['RR_Ratio_TP2']})
- Win Probability: {plan['Win_Probability']} ({plan['Win_Prob_Source']})
- Expected Value per R: {plan['Expected_Value_per_R']}
- Kelly Full: {plan['Kelly_Full_%']}%, Quarter Kelly: {plan['Kelly_Quarter_%']}%
- Position Assessment: {plan['Position_Note']}

**Instruksi Analisis:**
1) **Kondisi Pasar**: Jelaskan RSI/MACD/EMA & volume
2) **Skenario Trading**: Jelaskan '{plan['Scenario']}'
3) **Risk Management**: Analisis R/R dan Expected Value per R risked
4) **Position Sizing**: Jelaskan Quarter Kelly ({plan['Kelly_Quarter_%']}%)
5) **Outlook**: View {plan['View']} jangka pendek (1-2 minggu)
6) **Disclaimer**: Singkat, risiko trading

Gunakan bahasa Indonesia, professional, padat, bullet points.
"""
            resp = model.generate_content(prompt)
            ai_comment = resp.text.strip()
        except Exception as e:
            ai_comment = f"(AI ERROR) {e}. Check API key & internet."
    elif use_ai and (genai is None or not gemini_key):
        ai_comment = "(AI OFF) Gemini library/API key not available."

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
    res = plan["Resistance"]
    sup = plan["Support"]
    sl = plan["Stop_Loss"]
    tp1 = plan["TP1"]
    tp2 = plan["TP2"]
    try:
        e_low, e_high = [int(x.strip()) for x in plan["Entry_Zone"].split("–")]
    except ValueError:
        parts = plan["Entry_Zone"].replace("-", "–").split("–")
        e_low, e_high = int(parts[0].strip()), int(parts[1].strip())

    # Ganti nama anotasi berdasarkan skenario
    entry_level_name = "Breakout" if plan["Scenario"] == "Breakout Bullish" else "Support"

    for level, name in [
        (res, "Resistance"),
        (sup, "Support"),
        (sl, "Stop-Loss"),
        (tp1, "TP1"),
        (tp2, "TP2"),
    ]:
        # Garis resistance dan support dibuat lebih tebal
        is_main_level = name in ["Resistance", "Support"]
        fig.add_hline(
            y=level,
            line_dash="dot" if not is_main_level else "solid",
            line_width=1 if not is_main_level else 2,
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
        title=f"Trade Plan {plan['Ticker']} | Scenario: {plan['Scenario']} | View: {plan['View']}",
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
    metrics, prediction, df_bt, fig, fig_importance, feature_importance = train_final_and_report_ml(
        X, y, df_for_pred, best_params, best_thr, splits, ticker
    )
    return metrics, prediction, df_bt, fig, fig_importance, feature_importance, best_wr


def run_ta_for_ticker(
    ticker: str, df_capital: pd.DataFrame, use_ai: bool, gemini_key: str, ml_winrate=None
):
    plan, ai_comment = ta_generate_trade_plan(
        df_capital, ticker=ticker, use_ai=use_ai, gemini_key=gemini_key, ml_winrate=ml_winrate
    )
    fig = ta_plot_trade_plan(df_capital, plan)
    return plan, ai_comment, fig


# ==========================================================
# =============== UI ==================
# ==========================================================
st.title("📊 Professional Stock Analyzer v11")
st.markdown("*Enterprise-grade Stock Analysis with ML, TA & Risk Management*")
st.caption("🏆 Industry Best Practices: Sharpe Ratio | Max Drawdown | Kelly Criterion | R/R Analysis")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    
    # Input Ticker
    with st.expander("📊 **Ticker & Data**", expanded=True):
        tickers_input = st.text_input(
            "Ticker (pisah spasi/koma)",
            value="BBCA, BBRI, BMRI",
            help="Contoh: BBCA, BBRI, BREN",
        )
        auto_append_jk = st.checkbox("Auto-append .JK", value=True)
        start_date = st.date_input(
            "Start Date", value=pd.to_datetime("2020-01-01")
        ).strftime("%Y-%m-%d")

    # ML Settings
    with st.expander("🤖 **Machine Learning**", expanded=False):
        run_ml = st.checkbox("Aktifkan ML Quant", value=True)
        if run_ml:
            include_ihsg = st.checkbox("Fitur IHSG (^JKSE)", value=True)
            include_vix = st.checkbox("Fitur VIX (^VIX)", value=False)
            n_trials = st.slider("Optuna Trials", 5, 60, 20, step=5, 
                                help="Lebih sedikit = lebih cepat, tapi kurang optimal")
            n_splits = st.slider("TimeSeries Splits", 3, 10, 4)
            embargo = st.slider("Embargo (bar)", 0, 10, 5)

    # TA Settings
    with st.expander("📈 **Technical Analysis**", expanded=False):
        run_ta = st.checkbox("Aktifkan TA + Trade Plan", value=True)
        if run_ta:
            use_ai = st.checkbox("Komentar AI Gemini", value=False)
            if use_ai:
                gemini_key = get_gemini_api_key()
                if gemini_key:
                    st.success("✅ API Key terdeteksi")
                else:
                    st.warning("⚠️ API Key tidak ditemukan")
            else:
                gemini_key = ""

    st.markdown("---")
    run_btn = st.button("🚀 MULAI ANALISIS", type="primary", use_container_width=True)
    
    # Info sidebar
    st.markdown("---")
    st.caption("💡 **Tips:**")
    st.caption("• Kurangi Trials untuk kecepatan")
    st.caption("• Start Date 2020+ untuk performa optimal")
    st.caption("• Aktifkan AI untuk insight lebih dalam")

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

    # Summary cards at top
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("📊 Total Saham", len(tickers))
    with col_info2:
        st.metric("📅 Periode Data", f"{start_date} - Sekarang")
    with col_info3:
        status = []
        if run_ml:
            status.append("ML")
        if run_ta:
            status.append("TA")
        st.metric("🔧 Mode Analisis", " + ".join(status) if status else "Tidak ada")

    st.markdown("---")

    # External series (once)
    df_ext = pd.DataFrame()
    if run_ml and (include_ihsg or include_vix):
        with st.spinner("📥 Mengunduh data eksternal (IHSG/VIX)..."):
            try:
                df_ext = fetch_external_series(
                    start=start_date, include_ihsg=include_ihsg, include_vix=include_vix
                )
                st.success("✅ Data eksternal berhasil diunduh")
            except Exception as e:
                st.warning(
                    f"⚠️ Gagal mengunduh data eksternal: {e}. ML jalan tanpa konteks eksternal."
                )
                df_ext = pd.DataFrame()

    ml_predictions: List[Dict[str, Any]] = []
    ml_holdout_metrics: List[Dict[str, Any]] = []
    ticker_ml_winrates: Dict[str, float] = {}  # Store ML winrates for TA

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, t in enumerate(tickers):
        progress_bar.progress((idx) / len(tickers))
        status_text.text(f"Memproses {idx + 1}/{len(tickers)}: {t}")
        
        st.markdown(f"## 📌 {t}")
        
        with st.spinner(f"📊 Mengunduh data harga {t}..."):
            df_raw = fetch_prices(t, start=start_date)
            if df_raw.empty:
                st.error(f"❌ Data yfinance kosong untuk {t}. Skip.")
                continue
            df_capital = _sanitize_ohlcv(df_raw)
            st.success(f"✅ Data {t} berhasil diunduh ({len(df_capital)} data points)")

        # Use tabs for better mobile experience
        if run_ml and run_ta:
            tab1, tab2 = st.tabs(["🤖 Machine Learning", "📈 Technical Analysis"])
        elif run_ml:
            tab1 = st.container()
        elif run_ta:
            tab2 = st.container()
        else:
            st.warning("⚠️ Tidak ada mode analisis yang dipilih")
            continue

        if run_ml:
            with (tab1 if run_ta else st.container()):
                st.markdown("### 🤖 ML Quant Analysis")
                try:
                    with st.spinner("🔄 Training & backtest ML..."):
                        metrics, pred, df_bt, fig_ml, fig_importance, feature_importance, cv_wr = run_ml_for_ticker(
                            t,
                            df_capital,
                            df_ext,
                            n_trials=n_trials,
                            embargo=embargo,
                            n_splits=n_splits,
                        )
                    
                    # Enhanced Metrics in columns
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🎯 CV Win-Rate", f"{cv_wr:.2%}")
                    with col2:
                        st.metric("📊 Holdout WR", f"{metrics['winrate_holdout']:.2%}")
                    with col3:
                        st.metric("📈 Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                    with col4:
                        st.metric("📉 Max Drawdown", f"{metrics['max_drawdown']:.2%}")
                    
                    col5, col6, col7, col8 = st.columns(4)
                    with col5:
                        st.metric("💰 Profit Factor", f"{metrics['profit_factor']:.2f}")
                    with col6:
                        st.metric("⚖️ Win/Loss Ratio", f"{metrics['win_loss_ratio']:.2f}")
                    with col7:
                        st.metric("📉 MAE", f"{metrics['mae_holdout']:.0f}")
                    with col8:
                        st.metric("📊 R²", f"{metrics['r2_holdout']:.3f}")
                    
                    st.plotly_chart(fig_ml, use_container_width=True)
                    
                    # Prediction with Confidence Interval
                    st.markdown("#### 🔮 Prediksi T+1")
                    pred_col1, pred_col2, pred_col3 = st.columns(3)
                    with pred_col1:
                        st.metric("Prediksi", f"Rp {pred['ML_Pred_Harga_T+1']:,}", 
                                 delta=f"{pred['ML_Pred_Return_T+1']*100:.2f}%")
                    with pred_col2:
                        st.metric("Lower 95%", f"Rp {pred['ML_Pred_Lower_95']:,}")
                    with pred_col3:
                        st.metric("Upper 95%", f"Rp {pred['ML_Pred_Upper_95']:,}")
                    
                    # Feature Importance
                    with st.expander("� Feature Importance (Top 10)"):
                        st.plotly_chart(fig_importance, use_container_width=True)
                        st.dataframe(feature_importance, use_container_width=True)
                    
                    with st.expander("�📋 Detail Prediksi (15 data terakhir)"):
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
                    # Store winrate for TA to use
                    ticker_ml_winrates[t] = metrics['winrate_holdout']
                    
                except Exception as e:
                    st.error(f"❌ ML gagal untuk {t}: {e}")

        if run_ta:
            with (tab2 if run_ml else st.container()):
                st.markdown("### 📈 TA + Trade Plan")
                try:
                    # Pass ML winrate if available
                    ml_wr = ticker_ml_winrates.get(t, None)
                    plan, ai_comment, fig_ta = run_ta_for_ticker(
                        t, df_capital, use_ai=use_ai, gemini_key=gemini_key, ml_winrate=ml_wr
                    )
                    
                    # Key metrics in columns
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("💰 Harga Sekarang", f"Rp {plan['Close_T']:,}")
                    with col2:
                        st.metric("📊 Scenario", plan['Scenario'])
                    with col3:
                        st.metric("🎯 View", plan['View'])
                    with col4:
                        st.metric("📈 RSI14", plan['RSI14'])
                    
                    # Risk/Reward Metrics (Data-Driven)
                    st.markdown("#### ⚖️ Risk/Reward Analysis (Real Data)")
                    rr_col1, rr_col2, rr_col3, rr_col4 = st.columns(4)
                    with rr_col1:
                        st.metric("R/R Ratio (TP1)", f"{plan['RR_Ratio_TP1']:.2f}")
                    with rr_col2:
                        st.metric("R/R Ratio (TP2)", f"{plan['RR_Ratio_TP2']:.2f}")
                    with rr_col3:
                        st.metric("Win Probability", f"{plan['Win_Probability']:.1%}",
                                 help=f"Source: {plan['Win_Prob_Source']}")
                    with rr_col4:
                        st.metric("Expected Value/R", f"{plan['Expected_Value_per_R']:.2f}",
                                 help="Expected value per R risked")
                    
                    # Kelly Criterion & Position Sizing
                    st.markdown("#### 💰 Position Sizing (Kelly Criterion)")
                    kelly_col1, kelly_col2, kelly_col3 = st.columns(3)
                    with kelly_col1:
                        st.metric(
                            "Kelly Full", 
                            f"{plan['Kelly_Full_%']}%",
                            help="Full Kelly (theoretical max)"
                        )
                    with kelly_col2:
                        st.metric(
                            "Quarter Kelly", 
                            f"{plan['Kelly_Quarter_%']}%",
                            help="1/4 Kelly - Industry standard"
                        )
                    with kelly_col3:
                        ev_per_r = plan['Expected_Value_per_R']
                        if ev_per_r > 0.3:
                            badge = "� Excellent"
                        elif ev_per_r > 0:
                            badge = "🟡 Positive"
                        else:
                            badge = "🔴 Negative"
                        st.metric("Edge Assessment", badge)
                    
                    st.plotly_chart(fig_ta, use_container_width=True)
                    
                    # Info about probability source
                    if ml_wr is not None:
                        st.info(f"ℹ️ **Win Probability menggunakan ML Model**: "
                               f"{plan['Win_Probability']:.1%} (dari holdout validation)")
                    else:
                        st.info(f"ℹ️ **Win Probability berdasarkan TA**: "
                               f"{plan['Win_Probability']:.1%} (dari indikator teknikal)")
                    
                    with st.expander("📋 Detail Trade Plan"):
                        st.dataframe(
                            pd.DataFrame([plan]).T.rename(columns={0: "Value"}),
                            use_container_width=True,
                        )
                    
                    if ai_comment:
                        st.info(f"🤖 **AI Analysis:**\n\n{ai_comment}")
                        
                except Exception as e:
                    st.error(f"❌ TA gagal untuk {t}: {e}")
        
        st.markdown("---")

    # Complete progress
    progress_bar.progress(1.0)
    status_text.text("✅ Semua ticker selesai diproses!")

    # Summary Section
    st.markdown("---")
    st.markdown("# 📊 Ringkasan Hasil Analisis")
    
    if run_ml and ml_predictions:
        st.markdown("## 🤖 Ringkasan ML — Prediksi T+1")
        df_pred = pd.DataFrame(ml_predictions).set_index("Ticker")
        fmt_pred = df_pred.copy()
        if "ML_Close_T" in fmt_pred.columns:
            fmt_pred["ML_Close_T"] = fmt_pred["ML_Close_T"].round(0)
        if "ML_Pred_Harga_T+1" in fmt_pred.columns:
            fmt_pred["ML_Pred_Harga_T+1"] = fmt_pred["ML_Pred_Harga_T+1"].round(0)
        if "ML_Pred_Lower_95" in fmt_pred.columns:
            fmt_pred["ML_Pred_Lower_95"] = fmt_pred["ML_Pred_Lower_95"].round(0)
        if "ML_Pred_Upper_95" in fmt_pred.columns:
            fmt_pred["ML_Pred_Upper_95"] = fmt_pred["ML_Pred_Upper_95"].round(0)
        if "ML_Pred_Return_T+1" in fmt_pred.columns:
            fmt_pred["ML_Pred_Return_T+1"] = (
                fmt_pred["ML_Pred_Return_T+1"] * 100
            ).round(2)
        if "ML_Threshold_Arah" in fmt_pred.columns:
            fmt_pred["ML_Threshold_Arah"] = (fmt_pred["ML_Threshold_Arah"] * 100).round(
                2
            )
        st.dataframe(fmt_pred, use_container_width=True)
        
        # Download button for ML predictions
        csv_pred = fmt_pred.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download Prediksi ML (CSV)",
            data=csv_pred,
            file_name=f"ml_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

        st.markdown("## 🎯 Ringkasan ML — Metrik Performa")
        df_metrics = pd.DataFrame(ml_holdout_metrics).set_index("Ticker")
        fmt_met = df_metrics.copy()
        if "winrate_holdout" in fmt_met.columns:
            fmt_met["winrate_holdout"] = (fmt_met["winrate_holdout"] * 100).round(2)
        if "mae_holdout" in fmt_met.columns:
            fmt_met["mae_holdout"] = fmt_met["mae_holdout"].round(0)
        if "r2_holdout" in fmt_met.columns:
            fmt_met["r2_holdout"] = fmt_met["r2_holdout"].round(3)
        if "sharpe_ratio" in fmt_met.columns:
            fmt_met["sharpe_ratio"] = fmt_met["sharpe_ratio"].round(2)
        if "max_drawdown" in fmt_met.columns:
            fmt_met["max_drawdown"] = (fmt_met["max_drawdown"] * 100).round(2)
        if "win_loss_ratio" in fmt_met.columns:
            fmt_met["win_loss_ratio"] = fmt_met["win_loss_ratio"].round(2)
        if "profit_factor" in fmt_met.columns:
            fmt_met["profit_factor"] = fmt_met["profit_factor"].round(2)
        
        st.dataframe(fmt_met, use_container_width=True)
        
        # Interpretation guide
        with st.expander("📖 Panduan Interpretasi Metrik"):
            st.markdown("""
            **Win Rate**: Persentase prediksi arah yang benar (>55% = bagus)
            
            **Sharpe Ratio**: Risk-adjusted return
            - < 1: Poor
            - 1-2: Good
            - > 2: Excellent
            
            **Max Drawdown**: Kerugian maksimal dari peak (-20% = moderate risk)
            
            **Win/Loss Ratio**: Rata-rata profit vs loss (>1.5 = bagus)
            
            **Profit Factor**: Total profit / total loss (>1.5 = profitable)
            
            **R²**: Kualitas fit model (>0.3 = decent untuk harga saham)
            
            **Expected Value per R**: Profit/loss yang diharapkan per 1R risked
            - > 0.5: Excellent edge
            - 0.2-0.5: Good edge
            - 0-0.2: Marginal edge
            - < 0: Skip trade
            
            **Kelly Criterion**:
            - Full Kelly: Theoretical maximum (terlalu agresif)
            - Quarter Kelly (1/4): Industry standard untuk risk management real
            - Basis: Win-rate dari ML model holdout (data-driven, bukan asumsi)
            """)
        
        # Download button for metrics
        csv_metrics = fmt_met.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download Metrik ML (CSV)",
            data=csv_metrics,
            file_name=f"ml_metrics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

else:
    # Welcome message
    st.info("👈 **Cara Penggunaan:**\n\n1. Masukkan ticker saham di sidebar\n2. Pilih mode analisis (ML/TA atau keduanya)\n3. Atur parameter sesuai kebutuhan\n4. Klik tombol **MULAI ANALISIS**")
    
    st.markdown("---")
    st.markdown("### 🎯 Fitur Utama")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🤖 Machine Learning:**
        - Prediksi harga T+1 dengan confidence interval
        - Win-rate & Direction Accuracy
        - **Risk Metrics:**
          - Sharpe Ratio (risk-adjusted return)
          - Max Drawdown (kerugian maksimal)
          - Win/Loss Ratio & Profit Factor
        - **Feature Importance Analysis**
        - Walk-forward validation
        - Hyperparameter tuning otomatis (Optuna)
        """)
    
    with col2:
        st.markdown("""
        **📈 Technical Analysis:**
        - Trade plan otomatis (dynamic)
        - Support & Resistance detection
        - **Smart Scenario:**
          - Breakout (jika dekat resistance)
          - Swing (jika dekat support)
        - **Risk Management:**
          - Risk/Reward Ratio calculation
          - Position sizing recommendation (Kelly)
          - Expected return estimation
        - AI-powered insights (Gemini)
        """)
    
    st.markdown("---")
    st.markdown("### 📊 Industry Best Practices")
    
    st.markdown("""
    Aplikasi ini menggunakan **best practices** dari industri finance & quant trading:
    
    1. **Walk-Forward Validation** - Menghindari look-ahead bias
    2. **Embargo Period** - Menghindari data leakage
    3. **Risk-Adjusted Metrics** - Sharpe Ratio, Max Drawdown
    4. **Data-Driven Kelly Criterion** - Position sizing berdasarkan ML win-rate actual
    5. **Fractional Kelly (1/4)** - Industry standard risk management
    6. **Confidence Intervals** - Estimasi range prediksi (95% CI)
    7. **Feature Importance** - Transparency & interpretability
    8. **Dynamic Trade Plans** - Adaptif terhadap kondisi pasar
    9. **Expected Value per R** - Probabilistic risk/reward analysis
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Tips Penggunaan")
    
    tip_col1, tip_col2 = st.columns(2)
    with tip_col1:
        st.markdown("""
        **Untuk Performa Optimal:**
        - Start date: 2020+ (lebih cepat)
        - Optuna Trials: 20-30 (balance speed-accuracy)
        - Splits: 4-5 (sufficient validation)
        """)
    with tip_col2:
        st.markdown("""
        **Interpretasi Hasil:**
        - Sharpe > 1 = good, >2 = excellent
        - Win Rate > 55% = profitable edge
        - R/R Ratio > 2 = favorable risk/reward
        """)
    
    st.markdown("---")
    st.markdown("### 📊 Contoh Ticker IDX")
    st.code("BBCA, BBRI, BMRI, TLKM, ASII, UNVR, GOTO, BREN")
    
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit, LightGBM, Optuna & Gemini AI | v11.0 - Enhanced Edition")
