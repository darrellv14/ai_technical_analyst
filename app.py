import os
import json
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime

import streamlit as st
import yfinance as yf
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ML
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    log_loss,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ====== Gemini (opsional) ======
try:
    import google.generativeai as genai

    if os.getenv("GEMINI_API_KEY"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        _HAS_GEMINI = True
    else:
        _HAS_GEMINI = False
except Exception:
    _HAS_GEMINI = False

# ====== Optuna default ON (20 trials) ======
_USE_OPTUNA = True
OPTUNA_TRIALS = 20
try:
    import optuna
    from optuna.pruners import MedianPruner
except Exception:
    _USE_OPTUNA = False

# =====================
# Konstanta Global
# =====================
LOOKBACK = "3y"        # tetap 3y
HORIZON = 1
LABEL_METHOD = "fixed"
UP_TH = 0.0025
DOWN_TH = -0.0025
TRIPLE_PT = 1.5
TRIPLE_SL = 1.0
TX_COST = 0.0005
WINDOW = 48
N_SPLITS = 3
GAP = 3

IDX2LAB = {0: "Down", 1: "Flat", 2: "Up"}


# =====================
# Helpers & Core Logic
# =====================
def fetch_ticker_data(ticker: str, lookback: str) -> pd.DataFrame:
    periods = [lookback, "2y", "5y", "max"]
    for p in periods:
        try:
            df = yf.download(
                ticker,
                period=p,
                interval="1d",
                auto_adjust=False,
                progress=False,
            )
            if df is not None and not df.empty and len(df) > 252:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.index = pd.to_datetime(df.index)
                for c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

                if p != lookback:
                    cutoff = pd.Timestamp.now() - pd.DateOffset(
                        days=int(float(lookback[:-1]) * 365.25)
                    )
                    df = df.loc[df.index >= cutoff]

                return df.dropna()
        except Exception:
            pass
    raise ValueError(f"Data kosong untuk {ticker}")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    # Returns & realized vol
    X["ret1"] = X["Close"].pct_change()
    X["ret2"] = X["Close"].pct_change(2)
    X["ret5"] = X["Close"].pct_change(5)
    X["ret10"] = X["Close"].pct_change(10)
    X["logret1"] = np.log(X["Close"]).diff()
    X["rv5"] = X["ret1"].rolling(5).std()
    X["rv20"] = X["ret1"].rolling(20).std()
    X["rv60"] = X["ret1"].rolling(60).std()

    # Volume z-score
    vol_ma20 = X["Volume"].rolling(20).mean()
    vol_std20 = X["Volume"].rolling(20).std()
    X["vol_z"] = (X["Volume"] - vol_ma20) / (vol_std20 + 1e-9)

    # VWAP distance
    vwap = (X["High"] + X["Low"] + X["Close"]) / 3
    X["vwap_dist"] = X["Close"] / vwap - 1

    # Indicators (ta)
    X["rsi14"] = ta.momentum.RSIIndicator(X["Close"], 14).rsi()
    stoch = ta.momentum.StochasticOscillator(X["High"], X["Low"], X["Close"])
    X["stoch_k"] = stoch.stoch()
    X["stoch_d"] = stoch.stoch_signal()

    macd = ta.trend.MACD(X["Close"])
    X["macd"] = macd.macd()
    X["macd_signal"] = macd.macd_signal()
    X["macd_hist"] = macd.macd_diff()

    ppo = ta.momentum.PercentagePriceOscillator(X["Close"])
    X["ppo"] = ppo.ppo()
    X["ppos"] = ppo.ppo_signal()
    X["ppoh"] = ppo.ppo_hist()

    bb = ta.volatility.BollingerBands(X["Close"])
    X["bbl"] = bb.bollinger_lband()
    X["bbm"] = bb.bollinger_mavg()
    X["bbu"] = bb.bollinger_hband()
    X["bbb"] = bb.bollinger_pband()
    X["bbp"] = bb.bollinger_wband()

    atr = ta.volatility.AverageTrueRange(X["High"], X["Low"], X["Close"])
    X["atr14"] = atr.average_true_range()

    X["mfi14"] = ta.volume.MFIIndicator(
        X["High"], X["Low"], X["Close"], X["Volume"]
    ).money_flow_index()
    X["obv"] = ta.volume.OnBalanceVolumeIndicator(
        X["Close"], X["Volume"]
    ).on_balance_volume()
    X["cmf20"] = ta.volume.ChaikinMoneyFlowIndicator(
        X["High"], X["Low"], X["Close"], X["Volume"]
    ).chaikin_money_flow()

    X["adx14"] = ta.trend.ADXIndicator(
        X["High"], X["Low"], X["Close"]
    ).adx()
    X["cci20"] = ta.trend.CCIIndicator(
        X["High"], X["Low"], X["Close"]
    ).cci()
    X["willr14"] = ta.momentum.WilliamsRIndicator(
        X["High"], X["Low"], X["Close"]
    ).williams_r()

    for w in [8, 21, 50, 200]:
        X[f"ema{w}"] = ta.trend.EMAIndicator(X["Close"], w).ema_indicator()
        X[f"ema{w}_slope"] = X[f"ema{w}"].pct_change()
        X[f"ema{w}_gap"] = X["Close"] / X[f"ema{w}"] - 1

    X["skew20"] = X["ret1"].rolling(20).skew()
    X["kurt20"] = X["ret1"].rolling(20).kurt()

    X["dow"] = X.index.dayofweek
    X["dom"] = X.index.day
    X["mon"] = X.index.month
    X["is_month_end"] = X.index.is_month_end.astype(int)
    return X


def make_labels_fixed(df: pd.DataFrame, h: int, up: float, down: float) -> pd.Series:
    fwd = df["Close"].shift(-h) / df["Close"] - 1
    lab = np.where(fwd > up, 2, np.where(fwd < down, 0, 1))
    return pd.Series(lab, index=df.index, name=f"y_{h}d")


def make_labels_triple_barrier(df: pd.DataFrame, h: int, pt: float, sl: float) -> pd.Series:
    if "ret1" not in df.columns:
        df["ret1"] = df["Close"].pct_change()
    vol = df["ret1"].rolling(20).std().shift(1).bfill()
    labels = []
    idx = list(df.index)
    for i in range(len(idx)):
        if i >= len(idx) - h:
            labels.append(np.nan)
            continue
        entry = df["Close"].iloc[i]
        v = vol.iloc[i] or 0.01
        up_bar = entry * (1 + pt * v)
        dn_bar = entry * (1 - sl * v)
        path = df["Close"].iloc[i + 1 : i + 1 + h]
        hit_up = (path >= up_bar).any()
        hit_dn = (path <= dn_bar).any()
        if hit_up and not hit_dn:
            labels.append(2)
        elif hit_dn and not hit_up:
            labels.append(0)
        else:
            ret = path.iloc[-1] / entry - 1
            labels.append(2 if ret > 0 else (0 if ret < 0 else 1))
    return pd.Series(labels, index=df.index, name=f"y_tb_{h}d")


def make_sequences(Z: np.ndarray, yy: np.ndarray, win: int):
    Xs, ys = [], []
    if len(Z) > win:
        for i in range(win, len(Z)):
            Xs.append(Z[i - win : i])
            ys.append(yy[i])
    return np.asarray(Xs, np.float32), np.asarray(ys, np.int64)


def temp_scale_probs(p, grid=np.linspace(0.5, 3.0, 26)):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    p = p / p.sum(axis=1, keepdims=True)
    best_T = 1.0
    best_ll = np.inf
    for T in grid:
        logits = np.log(p)
        sm = np.exp(logits / T)
        sm /= sm.sum(axis=1, keepdims=True)
        ll = -np.mean(
            np.log(np.clip(sm[np.arange(len(sm)), sm.argmax(1)], 1e-9, 1.0))
        )
        if ll < best_ll:
            best_ll = ll
            best_T = T
    sm = np.exp(np.log(p) / best_T)
    sm /= sm.sum(axis=1, keepdims=True)
    return sm, best_T


def weight_search(probs_list, y_true):
    grid = np.linspace(0.0, 1.0, 11)
    best_w = None
    best_ll = np.inf
    for w1 in grid:
        for w2 in grid:
            for w3 in grid:
                w4 = 1.0 - w1 - w2 - w3
                if w4 < 0:
                    continue
                P = (
                    w1 * probs_list[0]
                    + w2 * probs_list[1]
                    + w3 * probs_list[2]
                    + w4 * probs_list[3]
                )
                ll = log_loss(y_true, P, labels=[0, 1, 2])
                if ll < best_ll:
                    best_ll = ll
                    best_w = (w1, w2, w3, w4)
    return best_w


def fetch_marketaux_news_df(ticker: str, limit: int = 40) -> pd.DataFrame:
    key = os.getenv("MARKETAUX_API_KEY")
    if not key:
        return pd.DataFrame(columns=["date", "title", "desc"])

    import requests

    params = {
        "api_token": key,
        "limit": limit,
        "language": "en,id",
        "countries": "id",
        "filter_entities": "true",
    }

    if ticker == "^JKSE":
        params[
            "search"
        ] = 'IHSG OR "IDX Composite" OR "Jakarta Composite" OR "Bursa Efek Indonesia"'
    else:
        params["symbols"] = ticker

    base_url = "https://api.marketaux.com/v1/news/all"

    try:
        r = requests.get(base_url, params=params, timeout=20)
        r.raise_for_status()
        js = r.json()
        rows = []
        for it in js.get("data", []):
            dt = pd.to_datetime(
                it.get("published_at", it.get("published_on")), utc=True
            )
            rows.append(
                {
                    "date": dt.tz_convert(None).normalize(),
                    "title": it.get("title", ""),
                    "desc": it.get("description", ""),
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["date", "title", "desc"])
        return df
    except Exception:
        return pd.DataFrame(columns=["date", "title", "desc"])


def analyze_news_with_gemini(news_df: pd.DataFrame) -> pd.DataFrame:
    if not _HAS_GEMINI or news_df.empty:
        return pd.DataFrame(columns=["date", "score"])

    try:
        news_df_gemini = news_df.copy()
        news_df_gemini["date"] = news_df_gemini["date"].dt.strftime("%Y-%m-%d")
        payload = news_df_gemini.to_dict("records")

        prompt = (
            "Analyze each news item in the list. Provide: "
            "1. A numerical market impact score (-1.0 to 1.0). "
            "2. A categorical sentiment ('Positif', 'Negatif', 'Netral'). "
            "3. The original 'title'. "
            'Return STRICT JSON: {"analysis_results": ['
            '{"date": "YYYY-MM-DD", "title": "...", "score": number, "sentiment": "..."}'
            "]}"
        )

        # >>>>>> PENTING: Model tetap gemini-2.5-flash-preview-09-2025 <<<<<<
        model = genai.GenerativeModel("gemini-2.5-flash-preview-09-2025")
        resp = model.generate_content([json.dumps(payload), prompt])
        txt = getattr(resp, "text", "{}")
        if "```json" in txt:
            txt = txt.split("```json")[1].split("```")[0]

        analysis_results = json.loads(txt).get("analysis_results", [])
        scores_df = pd.DataFrame(analysis_results)
        if scores_df.empty or "score" not in scores_df.columns:
            return pd.DataFrame(columns=["date", "score"])

        scores_df["date"] = pd.to_datetime(scores_df["date"]).dt.normalize()
        sent = scores_df.groupby("date")["score"].mean().to_frame()
        sent = sent.rename(columns={"score": "sent_score"})
        return sent
    except Exception:
        return pd.DataFrame(columns=["sent_score"])


def _hline(fig, y, text, row, col, x0=None, x1=None, index=None):
    if index is None or len(index) == 0:
        return
    x0 = x0 or index.min()
    x1 = x1 or index.max()
    fig.add_shape(
        type="line",
        x0=x0,
        x1=x1,
        y0=y,
        y1=y,
        line=dict(dash="dot", width=1),
        row=row,
        col=col,
    )
    fig.add_annotation(
        x=x1,
        y=y,
        text=f"{text} {y:.2f}",
        showarrow=False,
        xanchor="left",
        xshift=8,
        row=row,
        col=col,
    )


def make_dashboard(
    data,
    p_up_s,
    p_dn_s,
    thr_opt,
    entry,
    sl,
    tp1,
    tp2,
    curve,
    bh,
    ticker,
    lookback,
    strat_name,
):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.68, 0.32]
    )

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )
    for name in ["ema21", "ema50", "ema200"]:
        if name in data:
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data[name], mode="lines", name=name.upper()
                ),
                row=1,
                col=1,
            )

    _hline(fig, entry, "ENTRY", 1, 1, index=data.index)
    _hline(fig, sl, "SL", 1, 1, index=data.index)
    _hline(fig, tp1, "TP1", 1, 1, index=data.index)
    _hline(fig, tp2, "TP2", 1, 1, index=data.index)

    fig.add_trace(
        go.Scatter(
            x=p_up_s.index, y=p_up_s, mode="lines", name="p(Up)-ema3"
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=p_dn_s.index, y=p_dn_s, mode="lines", name="p(Down)-ema3"
        ),
        row=2,
        col=1,
    )
    _hline(fig, thr_opt, f"thr={thr_opt:.2f}", 2, 1, index=p_up_s.index)

    fig.update_layout(
        title=f"{ticker} • {lookback.upper()}  |  Strategy: {strat_name}",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=700,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    eq = make_subplots(rows=1, cols=1)
    eq.add_trace(
        go.Scatter(x=curve.index, y=curve, mode="lines", name="Strategy")
    )
    eq.add_trace(
        go.Scatter(x=bh.index, y=bh, mode="lines", name="Buy&Hold")
    )
    eq.update_layout(
        title="Equity Curve (log scale)",
        yaxis_type="log",
        template="plotly_white",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig, eq


def run_full_pipeline(ticker: str, use_optuna: bool = True):
    # 1) Fetch + feature
    raw = fetch_ticker_data(ticker, LOOKBACK)
    feat = add_features(raw)

    # 2) News + sentiment
    news_df = fetch_marketaux_news_df(ticker, limit=40)
    sent = analyze_news_with_gemini(news_df)
    feat.index = pd.to_datetime(feat.index)
    sent.index = pd.to_datetime(sent.index)

    sent = sent.reindex(feat.index, method="ffill").fillna({"sent_score": 0.0})
    feat2 = feat.join(sent, how="left")
    feat2["sent_score_ma3"] = (
        feat2["sent_score"].rolling(3, min_periods=1).mean()
    )
    feat2["sent_score_ema7"] = feat2["sent_score"].ewm(
        span=7, adjust=False
    ).mean()

    # 3) Labels
    if LABEL_METHOD == "fixed":
        y_series = make_labels_fixed(feat2, HORIZON, UP_TH, DOWN_TH)
    else:
        y_series = make_labels_triple_barrier(
            feat2, HORIZON, TRIPLE_PT, TRIPLE_SL
        )

    data = pd.concat([feat2, y_series], axis=1).dropna()

    CAND_FEATS = [
        "ret1",
        "ret2",
        "ret5",
        "ret10",
        "logret1",
        "rv5",
        "rv20",
        "rv60",
        "vol_z",
        "vwap_dist",
        "rsi14",
        "stoch_k",
        "stoch_d",
        "macd",
        "macd_hist",
        "macd_signal",
        "bbl",
        "bbm",
        "bbu",
        "bbb",
        "bbp",
        "mfi14",
        "obv",
        "adx14",
        "atr14",
        "ema8",
        "ema21",
        "ema50",
        "ema200",
        "ema8_slope",
        "ema21_slope",
        "ema50_slope",
        "ema200_slope",
        "ema8_gap",
        "ema21_gap",
        "ema50_gap",
        "ema200_gap",
        "cci20",
        "willr14",
        "cmf20",
        "ppoh",
        "ppo",
        "ppos",
        "skew20",
        "kurt20",
        "dow",
        "dom",
        "mon",
        "is_month_end",
        "sent_score",
        "sent_score_ma3",
        "sent_score_ema7",
    ]
    FEATS = [c for c in CAND_FEATS if c in data.columns]
    X_all = data[FEATS].astype(float).values
    y_all = data[y_series.name].astype(int).values
    idx_all = data.index

    # Default params
    best_lgb_params = {
        "n_estimators": 800,
        "learning_rate": 0.02,
        "num_leaves": 63,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_samples": 18,
        "reg_lambda": 1.0,
    }
    best_xgb_params = {
        "eta": 0.03,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "num_boost_round": 600,
    }

    # 4) Optuna (ringan)
    if use_optuna and _USE_OPTUNA and len(X_all) > 400:
        ts_small = TimeSeriesSplit(n_splits=3)

        def obj_lgb(trial):
            params = {
                "objective": "multiclass",
                "num_class": 3,
                "learning_rate": trial.suggest_float("lr", 0.005, 0.05, log=True),
                "num_leaves": trial.suggest_int("leaves", 31, 255),
                "subsample": trial.suggest_float("sub", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("col", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("mcs", 10, 60),
                "reg_lambda": trial.suggest_float("l2", 0.0, 3.0),
                "n_estimators": trial.suggest_int("nest", 400, 1200),
                "random_state": SEED,
                "n_jobs": -1,
                "verbose": -1,
            }
            losses = []
            for tr, te in ts_small.split(X_all):
                Xtr, Xte = X_all[tr], X_all[te]
                ytr, yte = y_all[tr], y_all[te]
                lgbm = lgb.LGBMClassifier(**params)
                lgbm.fit(
                    Xtr,
                    ytr,
                    eval_set=[(Xte, yte)],
                    eval_metric="multi_logloss",
                    callbacks=[lgb.early_stopping(100, verbose=False)],
                )
                p = lgbm.predict_proba(Xte)
                losses.append(log_loss(yte, p, labels=[0, 1, 2]))
            return float(np.mean(losses))

        def obj_xgb(trial):
            params = {
                "objective": "multi:softprob",
                "num_class": 3,
                "eval_metric": "mlogloss",
                "eta": trial.suggest_float("eta", 0.01, 0.08, log=True),
                "max_depth": trial.suggest_int("depth", 3, 9),
                "subsample": trial.suggest_float("sub", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("col", 0.6, 1.0),
                "lambda": trial.suggest_float("l2", 0.0, 3.0),
                "num_boost_round": trial.suggest_int(
                    "nbr", 300, 1200
                ),
            }
            losses = []
            for tr, te in ts_small.split(X_all):
                dtr = xgb.DMatrix(X_all[tr], label=y_all[tr])
                dte = xgb.DMatrix(X_all[te], label=y_all[te])
                model = xgb.train(
                    params,
                    dtr,
                    params["num_boost_round"],
                    evals=[(dte, "val")],
                    early_stopping_rounds=100,
                    verbose_eval=False,
                )
                p = model.predict(
                    dte,
                    iteration_range=(0, model.best_iteration + 1),
                )
                losses.append(log_loss(y_all[te], p, labels=[0, 1, 2]))
            return float(np.mean(losses))

        study1 = optuna.create_study(
            direction="minimize", pruner=MedianPruner()
        )
        study1.optimize(
            obj_lgb, n_trials=min(OPTUNA_TRIALS, 30), show_progress_bar=False
        )
        bp = study1.best_params
        best_lgb_params.update(
            {
                "learning_rate": bp["lr"],
                "num_leaves": bp["leaves"],
                "subsample": bp["sub"],
                "colsample_bytree": bp["col"],
                "min_child_samples": bp["mcs"],
                "reg_lambda": bp["l2"],
                "n_estimators": bp["nest"],
            }
        )

        study2 = optuna.create_study(
            direction="minimize", pruner=MedianPruner()
        )
        study2.optimize(
            obj_xgb, n_trials=min(OPTUNA_TRIALS, 30), show_progress_bar=False
        )
        bx = study2.best_params
        best_xgb_params.update(
            {
                "eta": bx["eta"],
                "max_depth": bx["depth"],
                "subsample": bx["sub"],
                "colsample_bytree": bx["col"],
                "lambda": bx["l2"],
                "num_boost_round": bx["nbr"],
            }
        )

    # 5) Train CV + Ensemble
    results = []
    proba_oos = np.full((len(y_all), 3), np.nan, dtype=float)
    best_w = (0.45, 0.25, 0.20, 0.10)

    TS = TimeSeriesSplit(n_splits=N_SPLITS)
    for split_id, (tr, te) in enumerate(TS.split(X_all)):
        if len(tr) <= WINDOW or len(te) == 0:
            continue
        tr_end = tr[-1] - GAP if tr[-1] - GAP > tr[0] else tr[-1]
        tr_idx = np.arange(tr[0], tr_end + 1)
        if len(tr_idx) <= WINDOW:
            continue
        Xtr, Xte = X_all[tr_idx], X_all[te]
        ytr, yte = y_all[tr_idx], y_all[te]

        classes, counts = np.unique(ytr, return_counts=True)
        total = counts.sum()
        class_weight = {
            int(c): float(total / (len(classes) * cnt))
            for c, cnt in zip(classes, counts)
        }

        lgbm = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            random_state=SEED,
            class_weight=class_weight,
            n_jobs=-1,
            **best_lgb_params,
        )
        lgbm.fit(
            Xtr,
            ytr,
            eval_set=[(Xte, yte)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(120, verbose=False)],
        )
        p_lgb = lgbm.predict_proba(Xte)

        scaler = StandardScaler().fit(Xtr)
        Ztr, Zte = scaler.transform(Xtr), scaler.transform(Xte)
        Z_full_for_seq = np.concatenate(
            [Ztr[-(WINDOW - 1) :], Zte], axis=0
        )

        Xs_tr, ys_tr = make_sequences(Ztr, ytr, WINDOW)
        Xs_te = []
        if len(Z_full_for_seq) > WINDOW:
            for i in range(WINDOW, len(Z_full_for_seq)):
                Xs_te.append(Z_full_for_seq[i - WINDOW : i])
        Xs_te = np.asarray(Xs_te, np.float32)

        if (
            len(Xs_tr) == 0
            or len(Xs_te) == 0
            or len(Xs_te) != len(yte)
        ):
            p_gru = np.tile(
                np.array([1 / 3, 1 / 3, 1 / 3]), (len(yte), 1)
            )
            p_trf = p_gru.copy()
        else:
            tf.keras.backend.clear_session()
            gru = keras.Sequential(
                [
                    layers.Input(shape=(WINDOW, Ztr.shape[1])),
                    layers.GRU(64, return_sequences=True),
                    layers.Dropout(0.25),
                    layers.GRU(48),
                    layers.Dropout(0.25),
                    layers.Dense(3, activation="softmax"),
                ]
            )
            gru.compile(
                optimizer=keras.optimizers.Adam(1e-3),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            gru.fit(
                Xs_tr,
                ys_tr,
                epochs=30,
                batch_size=64,
                validation_split=0.1,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        "val_loss", patience=5, restore_best_weights=True
                    )
                ],
                verbose=0,
            )
            p_gru = gru.predict(Xs_te, verbose=0)

            inp = layers.Input(shape=(WINDOW, Ztr.shape[1]))
            x = layers.Dense(64)(inp)
            att = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
            x = layers.LayerNormalization(epsilon=1e-6)(x + att)
            ff = keras.Sequential(
                [
                    layers.Dense(128, activation="gelu"),
                    layers.Dense(64),
                ]
            )(x)
            x = layers.LayerNormalization(epsilon=1e-6)(x + ff)
            x = layers.GlobalAveragePooling1D()(x)
            x = layers.Dropout(0.3)(x)
            out = layers.Dense(3, activation="softmax")(x)
            trf = keras.Model(inp, out)
            trf.compile(
                optimizer=keras.optimizers.Adam(1e-3),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            trf.fit(
                Xs_tr,
                ys_tr,
                epochs=20,
                batch_size=64,
                validation_split=0.1,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        "val_loss", patience=4, restore_best_weights=True
                    )
                ],
                verbose=0,
            )
            p_trf = trf.predict(Xs_te, verbose=0)

        dtrain = xgb.DMatrix(Xtr, label=ytr)
        dtest = xgb.DMatrix(Xte, label=yte)
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "eta": best_xgb_params["eta"],
            "max_depth": best_xgb_params["max_depth"],
            "subsample": best_xgb_params["subsample"],
            "colsample_bytree": best_xgb_params["colsample_bytree"],
            "lambda": best_xgb_params["lambda"],
        }

        xgbm = xgb.train(
            params,
            dtrain,
            num_boost_round=best_xgb_params["num_boost_round"],
            evals=[(dtest, "val")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )
        p_xgb = xgbm.predict(
            dtest, iteration_range=(0, xgbm.best_iteration + 1)
        )

        p_lgb_cal, _ = temp_scale_probs(p_lgb)
        p_gru_cal, _ = temp_scale_probs(p_gru)
        p_trf_cal, _ = temp_scale_probs(p_trf)
        p_xgb_cal, _ = temp_scale_probs(p_xgb)

        w = weight_search(
            [p_lgb_cal, p_gru_cal, p_trf_cal, p_xgb_cal], yte
        )
        if w:
            best_w = w
        p_ens = (
            best_w[0] * p_lgb_cal
            + best_w[1] * p_gru_cal
            + best_w[2] * p_trf_cal
            + best_w[3] * p_xgb_cal
        )

        if p_ens.shape[0] == len(te):
            proba_oos[te] = p_ens
            y_pred = p_ens.argmax(axis=1)
            results.append(
                (
                    accuracy_score(yte, y_pred),
                    f1_score(yte, y_pred, average="macro"),
                )
            )
        else:
            p_gbdt_ens = 0.6 * p_lgb_cal + 0.4 * p_xgb_cal
            if p_gbdt_ens.shape[0] == len(te):
                proba_oos[te] = p_gbdt_ens
                y_pred = p_gbdt_ens.argmax(axis=1)
                results.append(
                    (
                        accuracy_score(yte, y_pred),
                        f1_score(yte, y_pred, average="macro"),
                    )
                )

    # CV metrics
    accs = [a for a, _ in results if not pd.isna(a)]
    f1s = [b for _, b in results if not pd.isna(b)]
    cv_acc = float(np.mean(accs)) if len(accs) else None
    cv_f1 = float(np.mean(f1s)) if len(f1s) else None

    idx_all = pd.to_datetime(idx_all)
    data.index = pd.to_datetime(data.index)

    p_up = pd.Series(proba_oos[:, 2], index=idx_all)
    p_dn = pd.Series(proba_oos[:, 0], index=idx_all)
    p_up_s = p_up.ewm(span=3, adjust=False).mean()
    p_dn_s = p_dn.ewm(span=3, adjust=False).mean()

    ret_daily = data["Close"].pct_change().reindex(index=idx_all)

    thr_opt = 0.58
    mg_opt = 0.06
    if use_optuna and _USE_OPTUNA and len(idx_all) > 200:
        def obj_thr(trial):
            thr = trial.suggest_float("thr", 0.50, 0.72)
            mg = trial.suggest_float("mg", 0.02, 0.15)
            sig = ((p_up_s > thr) & ((p_up_s - p_dn_s) > mg)).astype(int)
            sig = sig.shift(1).fillna(0)
            strat = sig * (
                ret_daily - TX_COST * sig.diff().abs().fillna(0)
            )
            if strat.std(ddof=1) == 0:
                return 1e6
            sr = (strat.mean() / strat.std(ddof=1)) * np.sqrt(252)
            return -float(sr)

        st_thr = optuna.create_study(direction="minimize")
        st_thr.optimize(
            obj_thr, n_trials=OPTUNA_TRIALS, show_progress_bar=False
        )
        thr_opt = float(st_thr.best_params["thr"])
        mg_opt = float(st_thr.best_params["mg"])
    else:
        best_sr = -1e9
        for thr in np.linspace(0.5, 0.72, 12):
            for mg in np.linspace(0.02, 0.15, 14):
                sig = ((p_up_s > thr) & ((p_up_s - p_dn_s) > mg)).astype(
                    int
                )
                sig = sig.shift(1).fillna(0)
                strat = sig * (
                    ret_daily - TX_COST * sig.diff().abs().fillna(0)
                )
                if strat.std(ddof=1) > 0:
                    sr = (
                        strat.mean() / strat.std(ddof=1)
                    ) * np.sqrt(252)
                    if sr > best_sr:
                        best_sr = sr
                        thr_opt = thr
                        mg_opt = mg

    signal = (
        (p_up_s > thr_opt) & ((p_up_s - p_dn_s) > mg_opt)
    ).astype(int)
    signal = (
        signal.reindex(data.index).fillna(0).shift(1)
    )
    ret = data["Close"].pct_change()
    strat_ret = (signal * (ret - TX_COST * signal.diff().abs().fillna(0))).fillna(0)
    curve = (1 + strat_ret).cumprod()
    bh = (1 + ret.fillna(0)).cumprod()

    if not curve.empty and curve.iloc[-1] != 0 and not pd.isna(curve.iloc[-1]):
        cagr = curve.iloc[-1] ** (252 / len(curve)) - 1
        sr = (
            (strat_ret.mean() / strat_ret.std(ddof=1)) * np.sqrt(252)
            if strat_ret.std(ddof=1) > 0
            else np.nan
        )
    else:
        cagr, sr = None, None

    mask = ~np.isnan(proba_oos).any(axis=1)
    yhat = None
    oos_df = None
    cls_report = None
    cm = None

    if mask.sum() > 0:
        yhat = proba_oos[mask].argmax(axis=1)
        cls_report = classification_report(
            y_all[mask],
            yhat,
            target_names=[IDX2LAB[i] for i in range(3)],
            zero_division=0,
        )
        cm = confusion_matrix(y_all[mask], yhat)

        oos_df = pd.DataFrame(
            {
                "Date": idx_all[mask],
                "Close": data.loc[idx_all[mask], "Close"].values,
                "Actual": y_all[mask],
                "Predicted": yhat,
                "Prob_Down": proba_oos[mask, 0],
                "Prob_Flat": proba_oos[mask, 1],
                "Prob_Up": proba_oos[mask, 2],
            }
        )
        oos_df["Actual"] = oos_df["Actual"].map(IDX2LAB)
        oos_df["Predicted"] = oos_df["Predicted"].map(IDX2LAB)
        oos_df["Date"] = oos_df["Date"].dt.strftime("%Y-%m-%d")

    # Trading plan levels
    ema21 = data["ema21"]
    ema50 = data["ema50"]
    ema200 = data["ema200"]
    atr = data["atr14"]

    last_close = float(data["Close"].iloc[-1])
    last_date = str(data.index[-1].date())
    atr_now = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else float(
        data["Close"].pct_change()
        .rolling(14)
        .std()
        .iloc[-1]
        * data["Close"].iloc[-1]
    )
    res_20 = (
        float(data["High"].rolling(20).max().iloc[-2])
        if len(data) >= 21
        else float(data["High"].max())
    )
    sup_20 = (
        float(data["Low"].rolling(20).min().iloc[-2])
        if len(data) >= 21
        else float(data["Low"].min())
    )

    trend_up = (last_close > ema50.iloc[-1]) and (
        ema50.iloc[-1] > ema200.iloc[-1]
    )
    prob_up = float(p_up_s.fillna(0.0).iloc[-1])
    prob_down = float(p_dn_s.fillna(0.0).iloc[-1])
    prob_flat = max(0.0, 1.0 - prob_up - prob_down)

    if trend_up and prob_up > 0.58:
        strat_name = "Buy on Breakout"
        entry = res_20 + 0.1 * atr_now
        sl = entry - 1.2 * atr_now
        tp1 = entry + 1.5 * atr_now
        tp2 = entry + 2.5 * atr_now
    else:
        strat_name = "Buy Pullback to EMA21"
        entry = float(ema21.iloc[-1])
        sl = entry - 1.5 * atr_now
        tp1 = entry + 1.2 * atr_now
        tp2 = entry + 2.0 * atr_now

    sl = min(sl, entry * 0.995)

    plan_text = (
        f"Strategy: {strat_name}\n"
        f"Entry: {entry:.2f}\n"
        f"Stop Loss: {sl:.2f}\n"
        f"TP1: {tp1:.2f}\n"
        f"TP2: {tp2:.2f}\n"
        f"(ProbUp={prob_up:.2f}, Close={last_close:.2f}, ATR={atr_now:.2f})"
    )

    if _HAS_GEMINI:
        try:
            model = genai.GenerativeModel("gemini-2.5-flash-preview-09-2025")
            prompt = (
                f"Buat trading plan singkat bahasa Indonesia untuk {ticker} berdasarkan data berikut. "
                f"Tanggal terakhir {last_date}. ProbUp={prob_up:.2f}. Close={last_close:.2f}. ATR14={atr_now:.2f}. "
                f"Resistance20={res_20:.2f}, Support20={sup_20:.2f}. "
                f"EMA21={float(ema21.iloc[-1]):.2f}, "
                f"EMA50={float(ema50.iloc[-1]):.2f}, EMA200={float(ema200.iloc[-1]):.2f}. "
                f"Rekomendasikan gaya: '{strat_name}'. Berikan level detail: Entry, Stop Loss, Take Profit 1 & 2, "
                f"logika singkat, dan manajemen risiko. Maks 6 kalimat."
            )
            resp = model.generate_content(prompt)
            plan_text = resp.text.strip()
        except Exception:
            pass

    pred_df = pd.DataFrame(
        {
            "Tanggal Prediksi": [last_date],
            "Tipe Strategi": [strat_name],
            "Prob. Naik": [f"{prob_up:.2%}"],
            "Prob. Datar": [f"{prob_flat:.2%}"],
            "Prob. Turun": [f"{prob_down:.2%}"],
            "Target Entry": [f"{entry:.2f}"],
            "Target SL": [f"{sl:.2f}"],
            "Target TP1": [f"{tp1:.2f}"],
            "Target TP2": [f"{tp2:.2f}"],
        }
    )

    outputs = {
        "ticker": ticker,
        "lookback": LOOKBACK,
        "cv_acc": cv_acc,
        "cv_f1": cv_f1,
        "thr_opt": thr_opt,
        "margin_opt": mg_opt,
        "last_close": last_close,
        "atr14": atr_now,
        "plan": plan_text,
        "entry": entry,
        "stop_loss": sl,
        "tp1": tp1,
        "tp2": tp2,
        "cagr": cagr,
        "sharpe": sr,
        "oos_df": oos_df,
        "p_up_s": p_up_s,
        "p_dn_s": p_dn_s,
        "curve": curve,
        "bh": bh,
        "data": data,
        "prob_up": prob_up,
        "prob_down": prob_down,
        "prob_flat": prob_flat,
        "cls_report": cls_report,
        "cm": cm,
        "strat_name": strat_name,
    }

    return outputs


# =====================
# STREAMLIT APP
# =====================
def main():
    st.set_page_config(
        page_title="ML Trading + Gemini Plan",
        page_icon="📈",
        layout="wide",
    )

    st.title("📈 ML Trading (3Y) + Optuna + Gemini Plan")
    st.caption("by Darrell + ChatGPT — versi Streamlit (responsive)")

    with st.sidebar:
        st.subheader("⚙️ Pengaturan")
        user_ticker = st.text_input(
            "Ticker (BBCA, BBCA.JK, ^JKSE, dll)",
            value="^JKSE",
        ).strip()

        # normalisasi .JK
        if user_ticker and user_ticker.isalpha() and not user_ticker.endswith(".JK") and not user_ticker.startswith("^"):
            ticker = user_ticker + ".JK"
        else:
            ticker = user_ticker or "^JKSE"

        use_optuna = st.checkbox(
            "Aktifkan Optuna (lebih lambat tapi lebih optimal)",
            value=True,
        )

        st.markdown("---")
        st.markdown("**API Keys**")
        st.write(
            "- `GEMINI_API_KEY`: diperlukan untuk trading plan AI & sentimen berita.\n"
            "- `MARKETAUX_API_KEY`: untuk berita pasar (opsional)."
        )

        run_btn = st.button("🚀 Jalankan Analisis", type="primary")

    if not run_btn:
        st.info("Masukkan ticker lalu klik **🚀 Jalankan Analisis**.")
        return

    try:
        with st.spinner("Mengambil data & menjalankan pipeline ML (harap tunggu)..."):
            outputs = run_full_pipeline(ticker, use_optuna=use_optuna)
    except Exception as e:
        st.error(f"Gagal menjalankan pipeline: {e}")
        return

    data = outputs["data"]
    if data is None or data.empty:
        st.warning("Data kosong setelah pemrosesan. Coba ticker lain.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Ticker", outputs["ticker"])
        st.metric("Lookback", outputs["lookback"])
    with col2:
        st.metric(
            "CV Accuracy",
            f"{outputs['cv_acc']:.3f}" if outputs["cv_acc"] is not None else "N/A",
        )
        st.metric(
            "CV F1",
            f"{outputs['cv_f1']:.3f}" if outputs["cv_f1"] is not None else "N/A",
        )
    with col3:
        st.metric(
            "CAGR (net)",
            f"{outputs['cagr']*100:.2f}%" if outputs["cagr"] is not None else "N/A",
        )
        st.metric(
            "Sharpe Ratio",
            f"{outputs['sharpe']:.2f}" if outputs["sharpe"] is not None else "N/A",
        )
    with col4:
        st.metric("Threshold", f"{outputs['thr_opt']:.2f}")
        st.metric("Margin Diff", f"{outputs['margin_opt']:.2f}")

    st.markdown("### 🧠 Trading Plan (AI)")
    st.write(outputs["plan"])

    tabs = st.tabs(
        [
            "📊 Chart & Equity Curve",
            "📋 Backtest Detail",
            "🎯 Prediksi T+1",
            "📑 Classification Report",
        ]
    )

    with tabs[0]:
        fig1, fig2 = make_dashboard(
            data=data,
            p_up_s=outputs["p_up_s"],
            p_dn_s=outputs["p_dn_s"],
            thr_opt=outputs["thr_opt"],
            entry=outputs["entry"],
            sl=outputs["stop_loss"],
            tp1=outputs["tp1"],
            tp2=outputs["tp2"],
            curve=outputs["curve"],
            bh=outputs["bh"],
            ticker=outputs["ticker"],
            lookback=outputs["lookback"],
            strat_name=outputs["strat_name"],
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)

    with tabs[1]:
        st.markdown("#### Detail Backtest (20 Hari Terakhir)")
        oos_df = outputs["oos_df"]
        if oos_df is not None and not oos_df.empty:
            st.dataframe(
                oos_df.tail(20),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Belum ada data OOS backtest yang valid.")

    with tabs[2]:
        st.markdown("#### Prediksi & Target Harga (T+1)")
        st.write(
            "Model ini **klasifikasi arah (Naik/Datar/Turun)**, bukan prediksi harga pasti.\n"
            "Tabel di bawah menunjukkan probabilitas arah T+1 dan level target dari strategi."
        )
        st.dataframe(
            outputs["pred_df"]
            if "pred_df" in outputs
            else pd.DataFrame(
                {
                    "Tanggal Prediksi": [data.index[-1].date()],
                    "Tipe Strategi": [outputs["strat_name"]],
                    "Prob. Naik": [f"{outputs['prob_up']:.2%}"],
                    "Prob. Datar": [f"{outputs['prob_flat']:.2%}"],
                    "Prob. Turun": [f"{outputs['prob_down']:.2%}"],
                    "Target Entry": [f"{outputs['entry']:.2f}"],
                    "Target SL": [f"{outputs['stop_loss']:.2f}"],
                    "Target TP1": [f"{outputs['tp1']:.2f}"],
                    "Target TP2": [f"{outputs['tp2']:.2f}"],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with tabs[3]:
        st.markdown("#### Classification Report & Confusion Matrix")
        if outputs["cls_report"] is not None:
            st.text(outputs["cls_report"])
        if outputs["cm"] is not None:
            st.write("Confusion Matrix:")
            st.write(outputs["cm"])


if __name__ == "__main__":
    main()
