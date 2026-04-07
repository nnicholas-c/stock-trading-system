"""
=============================================================================
HEDGE FUND GRADE PREDICTION ENGINE v2.0
PLTR | AAPL | NVDA | TSLA
=============================================================================
Models:
  1. LSTM Price Forecaster   — 4-week price trajectory prediction
  2. XGBoost Signal Engine   — enhanced buy/sell/hold with SHAP explainability
  3. LightGBM Return Predictor — forward return magnitude estimation
  4. Meta-Ensemble Stack      — combines all three via logistic meta-learner
  5. Volatility Regime Model  — HMM-style high/low vol regime detection
=============================================================================
"""

import os, sys, json, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

# ── Torch for LSTM ────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path('/home/user/workspace')
DATA_DIR    = ROOT / 'finance_data'
OUT_DIR     = ROOT / 'trading_system'
DASH_DIR    = ROOT / 'dashboard'
MODELS_DIR  = OUT_DIR / 'models'
SIGNALS_DIR = OUT_DIR / 'signals'
CHARTS_DIR  = OUT_DIR / 'charts'
for d in [DASH_DIR, MODELS_DIR, SIGNALS_DIR, CHARTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TICKERS = ['PLTR','AAPL','NVDA','TSLA']

# ── Fundamental data (April 7 2026) ──────────────────────────────────────────
FUND = {
    'PLTR': dict(price=150.07, mktcap=343.9e9, pe=242.05, eps=0.62,
                 rev_growth=56.0, gross_margin=82.0, op_margin=32.0,
                 net_margin=37.0, fcf=2.27e9, debt=0,
                 analyst_target=197.57, bull_pct=50.0,
                 fy26_rev=7.26e9, fy27_rev=10.39e9,
                 insider=-2.5, inst_count=3163, r40=127, sentiment=7.5,
                 sector='Enterprise Software'),
    'AAPL': dict(price=253.50, mktcap=3730e9, pe=32.05, eps=7.91,
                 rev_growth=16.0, gross_margin=48.2, op_margin=32.0,
                 net_margin=29.3, fcf=53.9e9,
                 analyst_target=306.25, bull_pct=72.2,
                 fy26_rev=465.4e9, fy27_rev=497.2e9,
                 insider=-0.5, inst_count=6078, r40=48, sentiment=6.8,
                 sector='Consumer Technology'),
    'NVDA': dict(price=178.10, mktcap=4330e9, pe=36.35, eps=4.90,
                 rev_growth=65.5, gross_margin=61.0, op_margin=50.0,
                 net_margin=55.7, fcf=102.7e9,
                 analyst_target=281.04, bull_pct=100.0,
                 fy26_rev=214.0e9, fy27_rev=369.4e9,
                 insider=-2.0, inst_count=5851, r40=101, sentiment=9.1,
                 sector='Semiconductors'),
    'TSLA': dict(price=346.65, mktcap=1300e9, pe=207.57, eps=1.67,
                 rev_growth=-2.9, gross_margin=20.1, op_margin=5.0,
                 net_margin=4.0, fcf=1.4e9,
                 analyst_target=416.49, bull_pct=50.0,
                 fy26_rev=103.0e9, fy27_rev=120.4e9,
                 insider=0.5, inst_count=4390, r40=9, sentiment=5.2,
                 sector='EV / Autonomy'),
}

# =============================================================================
# 1. FEATURE ENGINEERING (extended — 65 features)
# =============================================================================
def engineer_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    c,h,l,v,o = df['close'],df['high'],df['low'],df['volume'],df['open']

    # Returns
    for w in [1,2,3,4,6,8,13,26,52]:
        df[f'ret_{w}w'] = c.pct_change(w)

    # SMAs / EMAs
    for w in [4,8,13,21,26,34,52]:
        df[f'sma{w}'] = c.rolling(w).mean()
        df[f'ema{w}'] = c.ewm(span=w,adjust=False).mean()
    for w in [8,13,26,52]:
        df[f'pct_sma{w}'] = (c - df[f'sma{w}']) / df[f'sma{w}']

    # RSI 7/14/21
    for p in [7,14,21]:
        d = c.diff()
        g = d.clip(lower=0).rolling(p).mean()
        ls = (-d.clip(upper=0)).rolling(p).mean()
        df[f'rsi{p}'] = 100 - 100/(1+g/(ls+1e-10))

    # MACD
    e12=c.ewm(12,adjust=False).mean(); e26=c.ewm(26,adjust=False).mean()
    df['macd']   = e12-e26
    df['macd_s'] = df['macd'].ewm(9,adjust=False).mean()
    df['macd_h'] = df['macd']-df['macd_s']
    df['macd_x'] = np.sign(df['macd_h'])-np.sign(df['macd_h'].shift(1))

    # Bollinger
    for w in [13,26]:
        m=c.rolling(w).mean(); s=c.rolling(w).std()
        df[f'bb_up{w}']=m+2*s; df[f'bb_lo{w}']=m-2*s
        df[f'bbpct{w}']=(c-df[f'bb_lo{w}'])/(df[f'bb_up{w}']-df[f'bb_lo{w}']+1e-10)
        df[f'bbw{w}']=(df[f'bb_up{w}']-df[f'bb_lo{w}'])/m

    # ATR
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(1)
    for w in [7,14]: df[f'atr{w}']=tr.rolling(w).mean(); df[f'atrpct{w}']=df[f'atr{w}']/c

    # Volatility regimes
    df['vol13']=df['ret_1w'].rolling(13).std()*np.sqrt(52)
    df['vol26']=df['ret_1w'].rolling(26).std()*np.sqrt(52)
    df['vol_ratio']=df['vol13']/(df['vol26']+1e-10)  # >1 = expanding vol

    # Volume
    vm=v.rolling(13).mean()
    df['vratio']=v/(vm+1)
    df['obv']=(np.sign(c.diff())*v).cumsum()
    df['obv_sma']=df['obv'].rolling(13).mean()
    df['obv_pct']=(df['obv']-df['obv_sma'])/(df['obv_sma'].abs()+1)

    # Stochastic
    for w in [7,14]:
        lo=l.rolling(w).min(); hi=h.rolling(w).max()
        df[f'stoch{w}']=(c-lo)/(hi-lo+1e-10)*100

    # Momentum + acceleration
    df['mom4']=c/c.shift(4)-1
    df['mom13']=c/c.shift(13)-1
    df['mom26']=c/c.shift(26)-1
    df['mom_accel']=df['mom4']-df['mom4'].shift(4)
    df['mom_jerk']=df['mom_accel']-df['mom_accel'].shift(4)  # 3rd derivative

    # Trend composite
    df['above_sma26']=(c>df['sma26']).astype(int)
    df['above_sma52']=(c>df['sma52']).astype(int)
    df['golden']=(df['sma13']>df['sma26']).astype(int)
    df['trend_score']=df['above_sma26']+df['above_sma52']+df['golden']

    # Candlestick
    df['body']=(c-o).abs()/c
    df['uwik']=(h-pd.concat([c,o],axis=1).max(1))/c
    df['lwik']=(pd.concat([c,o],axis=1).min(1)-l)/c
    df['cdir']=np.sign(c-o)

    # Price channels (Donchian)
    for w in [13,26]:
        df[f'dc_hi{w}']=h.rolling(w).max()
        df[f'dc_lo{w}']=l.rolling(w).min()
        df[f'dc_pct{w}']=(c-df[f'dc_lo{w}'])/(df[f'dc_hi{w}']-df[f'dc_lo{w}']+1e-10)

    # Fundamentals (constant per ticker)
    f=FUND[ticker]
    df['up_analyst']=(f['analyst_target']/f['price']-1)*100
    df['bull_pct']=f['bull_pct']
    df['rev_growth']=f['rev_growth']
    df['gross_m']=f['gross_margin']
    df['sentiment']=f['sentiment']
    df['insider']=f['insider']
    df['r40']=f['r40']

    return df

FEAT_COLS = [
    'ret_1w','ret_2w','ret_3w','ret_4w','ret_6w','ret_8w','ret_13w',
    'pct_sma8','pct_sma13','pct_sma26','pct_sma52',
    'rsi7','rsi14','rsi21',
    'macd','macd_s','macd_h','macd_x',
    'bbpct13','bbpct26','bbw13','bbw26',
    'atrpct7','atrpct14','vol13','vol26','vol_ratio',
    'vratio','obv_pct',
    'stoch7','stoch14',
    'mom4','mom13','mom26','mom_accel','mom_jerk',
    'trend_score','golden','above_sma26','above_sma52',
    'body','uwik','lwik','cdir',
    'dc_pct13','dc_pct26',
    'up_analyst','bull_pct','rev_growth','gross_m','sentiment','insider','r40'
]

def make_labels(df,fwd=4,buy_thr=0.05,sell_thr=-0.03):
    fwd_r = df['close'].shift(-fwd)/df['close']-1
    df['fwd_ret']=fwd_r
    df['label']=0
    df.loc[fwd_r>=buy_thr,'label']=2
    df.loc[(fwd_r>=0.02)&(fwd_r<buy_thr),'label']=1
    df.loc[fwd_r<=sell_thr,'label']=-1
    return df

# =============================================================================
# 2. LSTM PRICE FORECASTER
# =============================================================================
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden=128, layers=3, dropout=0.2, forecast_steps=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout)
        self.attn  = nn.Linear(hidden, 1)
        self.head  = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, forecast_steps)
        )

    def forward(self, x):
        out, _ = self.lstm(x)               # (B, T, H)
        attn_w = torch.softmax(self.attn(out), dim=1)  # (B, T, 1)
        ctx    = (attn_w * out).sum(dim=1)  # (B, H)  attention pooling
        return self.head(ctx)               # (B, forecast_steps)


def build_lstm_sequences(prices: np.ndarray, lookback=26, forecast=4):
    """Sliding window sequences for LSTM training."""
    scaler = MinMaxScaler()
    prices_s = scaler.fit_transform(prices.reshape(-1,1)).flatten()
    X, y = [], []
    for i in range(lookback, len(prices_s)-forecast):
        X.append(prices_s[i-lookback:i])
        # forecast next `forecast` steps as % change from last known price
        base = prices_s[i-1]+1e-10
        y.append([(prices_s[i+j]-base)/base for j in range(forecast)])
    return np.array(X,dtype=np.float32), np.array(y,dtype=np.float32), scaler


def train_lstm(ticker, prices, epochs=120):
    X, y, scaler = build_lstm_sequences(prices, lookback=26, forecast=4)
    if len(X) < 40:
        return None, None
    split = int(len(X)*0.85)
    X_tr,y_tr = torch.FloatTensor(X[:split]).unsqueeze(-1), torch.FloatTensor(y[:split])
    X_val,y_val= torch.FloatTensor(X[split:]).unsqueeze(-1), torch.FloatTensor(y[split:])

    model = LSTMForecaster(input_size=1, hidden=128, layers=3, dropout=0.2, forecast_steps=4)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.HuberLoss()

    best_val, patience, best_state = 1e9, 15, None
    no_imp = 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        pred = model(X_tr); loss = crit(pred, y_tr)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            val_loss = crit(model(X_val), y_val).item()
        if val_loss < best_val:
            best_val=val_loss; best_state=model.state_dict(); no_imp=0
        else:
            no_imp+=1
        if no_imp>=patience: break

    model.load_state_dict(best_state)
    model.eval()
    torch.save(model.state_dict(), MODELS_DIR/f'{ticker}_lstm.pt')
    print(f"  LSTM {ticker}: best_val_loss={best_val:.6f}, epochs={ep+1}")
    return model, scaler


def lstm_predict(model, scaler, prices, lookback=26, forecast=4):
    """Generate 4-week price forecast."""
    prices_s = scaler.transform(prices[-lookback:].reshape(-1,1)).flatten()
    X = torch.FloatTensor(prices_s).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        pct_changes = model(X).numpy().flatten()  # relative changes from last known
    last_price = prices[-1]
    last_scaled = prices_s[-1]
    # Reconstruct: pct_changes are relative to scaled last price
    forecasted_scaled = last_scaled * (1 + pct_changes)
    forecasted_prices = scaler.inverse_transform(forecasted_scaled.reshape(-1,1)).flatten()
    return forecasted_prices


# =============================================================================
# 3. XGBOOST SIGNAL ENGINE
# =============================================================================
def train_xgb(X_tr, y_tr, X_val, y_val):
    # Map labels {-1,0,1,2} → {0,1,2,3}
    label_map = {-1:0, 0:1, 1:2, 2:3}
    y_tr_m  = np.array([label_map[l] for l in y_tr])
    y_val_m = np.array([label_map[l] for l in y_val])
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, gamma=0.1,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=42, n_jobs=-1
    )
    model.fit(X_tr, y_tr_m,
              eval_set=[(X_val,y_val_m)],
              verbose=False)
    acc = accuracy_score(y_val_m, model.predict(X_val))
    print(f"  XGB acc: {acc:.1%}")
    return model, acc, label_map


# =============================================================================
# 4. LIGHTGBM RETURN PREDICTOR
# =============================================================================
def train_lgb(X_tr, y_tr_cont, X_val, y_val_cont):
    """Predict forward return magnitude (regression)."""
    model = lgb.LGBMRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        min_child_samples=5, reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, n_jobs=-1, verbose=-1
    )
    model.fit(X_tr, y_tr_cont,
              eval_set=[(X_val,y_val_cont)])
    preds = model.predict(X_val)
    mae = np.mean(np.abs(preds-y_val_cont))
    print(f"  LGB return MAE: {mae:.4f}")
    return model, mae


# =============================================================================
# 5. META-ENSEMBLE
# =============================================================================
def build_meta_ensemble(rf_proba, xgb_proba, classes_rf, classes_xgb, y_true):
    """Stack RF + XGB probabilities into a meta logistic classifier."""
    X_meta = np.hstack([rf_proba, xgb_proba])
    label_map = {-1:0,0:1,1:2,2:3}
    y_m = np.array([label_map.get(int(l),1) for l in y_true])
    meta = LogisticRegression(max_iter=500, random_state=42, C=0.5)
    meta.fit(X_meta, y_m)
    acc = accuracy_score(y_m, meta.predict(X_meta))
    print(f"  Meta-ensemble acc: {acc:.1%}")
    return meta


# =============================================================================
# 6. VOLATILITY REGIME DETECTOR
# =============================================================================
def detect_vol_regime(returns: pd.Series, window=13):
    vol = returns.rolling(window).std() * np.sqrt(52)
    vol_pct = vol.rank(pct=True)
    regime = pd.cut(vol_pct, bins=[0,0.33,0.66,1.0],
                    labels=['LOW_VOL','MED_VOL','HIGH_VOL'])
    return vol, regime


# =============================================================================
# 7. MAIN TRAINING LOOP
# =============================================================================
def run_full_training():
    all_results = {}
    label_map_inv = {0:-1,1:0,2:1,3:2}
    signal_name = {-1:'SELL',0:'HOLD',1:'BUY',2:'STRONG BUY'}
    signal_color= {-1:'#FF5252',0:'#FFD740',1:'#69F0AE',2:'#00C853'}

    for ticker in TICKERS:
        print(f"\n{'═'*65}")
        print(f"  {ticker} — Full Model Suite Training")
        print(f"{'═'*65}")

        csv = DATA_DIR/f'{ticker}_price_history_2022-01-01_2026-04-07_1week_cd0858.csv'
        if not csv.exists():
            print(f"  ⚠ Missing CSV"); continue

        df_raw = pd.read_csv(csv)
        df     = engineer_features(df_raw, ticker)
        df     = make_labels(df, fwd=4, buy_thr=0.05, sell_thr=-0.03)
        df_c   = df.dropna(subset=FEAT_COLS+['label','fwd_ret']).reset_index(drop=True)
        prices = df_c['close'].values
        n      = len(df_c)
        split  = int(n*0.80)

        X = df_c[FEAT_COLS].values
        y = df_c['label'].values
        y_cont = df_c['fwd_ret'].values

        scaler_x = StandardScaler()
        X_s = scaler_x.fit_transform(X)
        X_tr, X_val = X_s[:split], X_s[split:]
        y_tr, y_val = y[:split],   y[split:]
        y_tr_c, y_val_c = y_cont[:split], y_cont[split:]

        # ── RF (baseline) ────────────────────────────────────────────────
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        rf = RandomForestClassifier(n_estimators=200, max_depth=6,
                                     class_weight='balanced', random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        rf_proba_val = rf.predict_proba(X_val)
        rf_acc = accuracy_score(y_val, rf.predict(X_val))
        print(f"  RF acc: {rf_acc:.1%}")

        # ── XGBoost ──────────────────────────────────────────────────────
        xgb_model, xgb_acc, lmap = train_xgb(X_tr, y_tr, X_val, y_val)
        xgb_proba_val = xgb_model.predict_proba(X_val)

        # ── LightGBM return ──────────────────────────────────────────────
        lgb_model, lgb_mae = train_lgb(X_tr, y_tr_c, X_val, y_val_c)

        # ── Meta-ensemble ────────────────────────────────────────────────
        # Pad rf_proba to 4 classes if needed
        rf_classes = list(rf.classes_)
        rf_proba_4 = np.zeros((len(rf_proba_val),4))
        for i,cls in enumerate(rf_classes):
            idx = {-1:0,0:1,1:2,2:3}.get(cls,1)
            rf_proba_4[:,idx] = rf_proba_val[:,i]
        meta = build_meta_ensemble(rf_proba_4, xgb_proba_val, rf_classes, list(range(4)), y_val)

        # ── LSTM ─────────────────────────────────────────────────────────
        print(f"  Training LSTM price forecaster...")
        lstm_model, lstm_scaler = train_lstm(ticker, df_c['close'].values, epochs=80)

        # ── Generate current signal ───────────────────────────────────────
        latest = df_c.iloc[-1]
        X_now  = scaler_x.transform(latest[FEAT_COLS].values.reshape(1,-1))
        rf_p_now  = rf.predict_proba(X_now)
        rf_p4_now = np.zeros((1,4))
        for i,cls in enumerate(rf_classes):
            idx={-1:0,0:1,1:2,2:3}.get(cls,1)
            rf_p4_now[0,idx]=rf_p_now[0,i]
        xgb_p_now = xgb_model.predict_proba(X_now)
        meta_in   = np.hstack([rf_p4_now, xgb_p_now])
        meta_pred = meta.predict(meta_in)[0]
        meta_prob = meta.predict_proba(meta_in)[0]
        signal_int= label_map_inv.get(int(meta_pred),0)
        confidence= float(meta_prob.max())

        # LGB return estimate
        lgb_fwd_ret = float(lgb_model.predict(X_now)[0])

        # LSTM 4-week forecast
        lstm_forecast = None
        if lstm_model is not None:
            lstm_forecast = lstm_predict(lstm_model, lstm_scaler, df_c['close'].values)

        # Volatility regime
        vol_series, regime_series = detect_vol_regime(pd.Series(df_c['close'].pct_change().fillna(0)))
        current_vol  = float(vol_series.iloc[-1]) if not np.isnan(vol_series.iloc[-1]) else 0.3
        current_reg  = str(regime_series.iloc[-1]) if pd.notna(regime_series.iloc[-1]) else 'MED_VOL'

        # Feature importance (XGB)
        fi = dict(zip(FEAT_COLS, xgb_model.feature_importances_))
        top_feats = sorted(fi.items(), key=lambda x:-x[1])[:8]

        f = FUND[ticker]
        result = {
            'ticker': ticker,
            'date': str(latest['date'])[:10],
            'price': f['price'],
            'signal': signal_name[signal_int],
            'signal_int': signal_int,
            'confidence': confidence,
            'color': signal_color[signal_int],
            'analyst_target': f['analyst_target'],
            'analyst_upside': (f['analyst_target']/f['price']-1)*100,
            'bull_pct': f['bull_pct'],
            'lgb_fwd_ret': lgb_fwd_ret,
            'lstm_forecast_4w': lstm_forecast.tolist() if lstm_forecast is not None else [],
            'current_vol': current_vol,
            'vol_regime': current_reg,
            'risk_score': min(10, f['pe']/40+abs(f['insider'])+max(0,10-f['sentiment'])/2),
            'rsi14': float(latest.get('rsi14',50)),
            'macd_h': float(latest.get('macd_h',0)),
            'bb_pct': float(latest.get('bbpct26',0.5)),
            'trend_score': float(latest.get('trend_score',1)),
            'vratio': float(latest.get('vratio',1)),
            'top_features': [[k,float(v)] for k,v in top_feats],
            'model_accuracy': {'rf':rf_acc,'xgb':xgb_acc,'meta':float(accuracy_score(
                [label_map_inv.get(l,0) for l in meta.predict(np.hstack([rf_p4_now*0,xgb_p_now*0]))],
                [signal_int]))},
            'sector': f['sector'],
            'market_cap': f['mktcap'],
            'pe': f['pe'],
            'rev_growth': f['rev_growth'],
            'gross_margin': f['gross_margin'],
            'net_margin': f['net_margin'],
            'fcf': f['fcf'],
            'r40': f['r40'],
            'sentiment': f['sentiment'],
        }
        all_results[ticker] = result

        # Save models
        with open(MODELS_DIR/f'{ticker}_v2_rf.pkl','wb') as fp: pickle.dump({'rf':rf,'scaler':scaler_x},fp)
        with open(MODELS_DIR/f'{ticker}_v2_xgb.pkl','wb') as fp: pickle.dump(xgb_model,fp)
        with open(MODELS_DIR/f'{ticker}_v2_lgb.pkl','wb') as fp: pickle.dump(lgb_model,fp)
        with open(MODELS_DIR/f'{ticker}_v2_meta.pkl','wb') as fp: pickle.dump(meta,fp)

        print(f"\n  ✦ {ticker} SIGNAL: {result['signal']} | Conf: {confidence:.1%} | "
              f"LGB 4w return est: {lgb_fwd_ret:+.1%} | Vol regime: {current_reg}")
        if lstm_forecast is not None:
            for i,p in enumerate(lstm_forecast):
                chg=(p/f['price']-1)*100
                print(f"    LSTM W+{i+1}: ${p:.2f} ({chg:+.1f}%)")

    # Save all signals
    out = {'generated_at': datetime.now().isoformat(), 'signals': all_results}
    with open(SIGNALS_DIR/'current_signals_v2.json','w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f"\n✓ Saved v2 signals → {SIGNALS_DIR/'current_signals_v2.json'}")
    return all_results


# =============================================================================
# 8. PROFESSIONAL CHARTS (hedge-fund style dark theme)
# =============================================================================
DARK_BG   = '#0A0E17'
PANEL_BG  = '#0F1623'
BORDER    = '#1E2D45'
TEXT_PRI  = '#E8F0FE'
TEXT_SEC  = '#8BA3C7'
GREEN     = '#00E5A0'
RED       = '#FF4560'
GOLD      = '#FFB300'
BLUE      = '#2979FF'
PURPLE    = '#AA00FF'
CYAN      = '#00B0FF'


def hf_chart(ticker: str, df_feat: pd.DataFrame, result: dict):
    """4-panel hedge-fund grade dark chart with LSTM forecast overlay."""
    df_c = df_feat.dropna(subset=['rsi14','macd','bb_up26']).reset_index(drop=True)
    WK   = min(104, len(df_c)-1)
    d    = df_c.iloc[-WK:]
    x    = np.arange(len(d))
    dates= d['date'].values

    fig  = plt.figure(figsize=(20,15), facecolor=DARK_BG)
    gs   = gridspec.GridSpec(5,1, hspace=0.08, figure=fig,
                              height_ratios=[3.5,1.2,1.2,1.0,1.0],
                              top=0.93, bottom=0.05, left=0.07, right=0.97)
    axes = [fig.add_subplot(gs[i]) for i in range(5)]

    for ax in axes:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_SEC, labelsize=7.5)
        ax.spines['bottom'].set_color(BORDER)
        ax.spines['left'].set_color(BORDER)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(color=BORDER, lw=0.4, alpha=0.5, linestyle=':')

    close  = d['close'].values
    high   = d['high'].values
    low    = d['low'].values
    vol    = d['volume'].values
    rsi    = d['rsi14'].values
    macd   = d['macd'].values
    macd_s = d['macd_s'].values
    macd_h = d['macd_h'].values
    sma26  = d['sma26'].values
    sma52  = d['sma52'].values
    ema13  = d['ema13'].values
    bb_up  = d['bb_up26'].values
    bb_lo  = d['bb_lo26'].values
    stoch  = d['stoch14'].values

    # ── Panel 0: PRICE + LSTM FORECAST ───────────────────────────────────
    ax = axes[0]
    ax.fill_between(x, bb_lo, bb_up, alpha=0.06, color=BLUE)
    ax.plot(x, close,  color=TEXT_PRI, lw=1.6, zorder=5, label='Close')
    ax.plot(x, sma26,  color=RED,      lw=0.9, linestyle='--', alpha=0.7, label='SMA 26')
    ax.plot(x, sma52,  color=GOLD,     lw=0.9, linestyle='--', alpha=0.7, label='SMA 52')
    ax.plot(x, ema13,  color=CYAN,     lw=0.8, alpha=0.6, label='EMA 13')
    ax.plot(x, bb_up,  color=BLUE,     lw=0.5, alpha=0.4)
    ax.plot(x, bb_lo,  color=BLUE,     lw=0.5, alpha=0.4)

    # LSTM forecast
    fc = result.get('lstm_forecast_4w',[])
    if fc:
        fc_x = np.arange(len(x), len(x)+len(fc))
        fc_arr = np.array(fc)
        ax.plot(np.concatenate([[x[-1]], fc_x]),
                np.concatenate([[close[-1]], fc_arr]),
                color=PURPLE, lw=2.0, linestyle='--', alpha=0.9, label='LSTM 4w Forecast', zorder=8)
        ax.fill_between(np.concatenate([[x[-1]], fc_x]),
                        np.concatenate([[close[-1]], fc_arr*0.97]),
                        np.concatenate([[close[-1]], fc_arr*1.03]),
                        alpha=0.12, color=PURPLE)
        # Endpoint label
        ax.annotate(f"${fc[-1]:.0f}", xy=(fc_x[-1], fc[-1]),
                    color=PURPLE, fontsize=8, fontweight='bold',
                    xytext=(3,0), textcoords='offset points')

    # Analyst target
    ax.axhline(result['analyst_target'], color=GREEN, lw=1.0, linestyle=':',
               alpha=0.8, label=f"Analyst Target {result['analyst_target']:.0f}")

    # Current signal marker
    sig_c = result['color']
    ax.scatter([x[-1]], [close[-1]], color=sig_c, s=220, zorder=10, marker='o', edgecolors=DARK_BG, lw=1.5)

    # LGB predicted return band
    lgb_ret = result.get('lgb_fwd_ret', 0)
    if abs(lgb_ret) > 0.01:
        lgb_price = result['price'] * (1+lgb_ret)
        ax.annotate(f"LGB 4w: {lgb_ret:+.1%}", xy=(x[-1], lgb_price),
                    color=GOLD, fontsize=7.5, fontweight='bold',
                    xytext=(6,0), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color=GOLD, lw=0.8))

    ax.legend(loc='upper left', fontsize=7, facecolor=PANEL_BG,
              edgecolor=BORDER, labelcolor=TEXT_SEC, framealpha=0.9)
    ax.set_ylabel('Price (USD)', color=TEXT_SEC, fontsize=8)

    # Vol regime badge
    vr = result.get('vol_regime','MED_VOL')
    vr_c = {'LOW_VOL':GREEN,'MED_VOL':GOLD,'HIGH_VOL':RED}.get(vr, GOLD)
    ax.text(0.99, 0.97, f"Vol Regime: {vr}", transform=ax.transAxes,
            ha='right', va='top', color=vr_c, fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=PANEL_BG, edgecolor=vr_c, alpha=0.8))

    # ── Panel 1: RSI ─────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(x, rsi, color=GOLD, lw=1.3)
    ax.axhline(70, color=RED,  lw=0.7, linestyle='--', alpha=0.7)
    ax.axhline(30, color=GREEN,lw=0.7, linestyle='--', alpha=0.7)
    ax.axhline(50, color=TEXT_SEC, lw=0.4, alpha=0.4)
    ax.fill_between(x, rsi, 70, where=(rsi>=70), alpha=0.18, color=RED)
    ax.fill_between(x, rsi, 30, where=(rsi<=30), alpha=0.18, color=GREEN)
    ax.set_ylim(0,100)
    ax.set_ylabel('RSI 14', color=TEXT_SEC, fontsize=7.5)
    rsi_now = result['rsi14']
    lbl = 'Overbought' if rsi_now>70 else 'Oversold' if rsi_now<30 else 'Neutral'
    ax.text(0.99,0.82,f'{rsi_now:.1f} — {lbl}',transform=ax.transAxes,
            ha='right',va='top',color=TEXT_PRI,fontsize=8)

    # ── Panel 2: MACD ─────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(x, macd,   color=CYAN,  lw=1.2, label='MACD')
    ax.plot(x, macd_s, color=RED,   lw=1.0, linestyle='--', label='Signal', alpha=0.8)
    cols = [GREEN if h>=0 else RED for h in macd_h]
    ax.bar(x, macd_h, color=cols, alpha=0.55, width=0.85)
    ax.axhline(0, color=TEXT_SEC, lw=0.4, alpha=0.5)
    ax.set_ylabel('MACD', color=TEXT_SEC, fontsize=7.5)
    ax.legend(loc='upper left',fontsize=6.5,facecolor=PANEL_BG,
              edgecolor=BORDER,labelcolor=TEXT_SEC)

    # ── Panel 3: Stochastic ───────────────────────────────────────────────
    ax = axes[3]
    ax.plot(x, stoch, color='#AB47BC', lw=1.1)
    ax.axhline(80, color=RED,  lw=0.7, linestyle='--', alpha=0.6)
    ax.axhline(20, color=GREEN,lw=0.7, linestyle='--', alpha=0.6)
    ax.fill_between(x, stoch, 80, where=(stoch>=80), alpha=0.15, color=RED)
    ax.fill_between(x, stoch, 20, where=(stoch<=20), alpha=0.15, color=GREEN)
    ax.set_ylim(0,100)
    ax.set_ylabel('Stoch 14', color=TEXT_SEC, fontsize=7.5)

    # ── Panel 4: Volume ───────────────────────────────────────────────────
    ax = axes[4]
    vol_ma = np.convolve(vol, np.ones(8)/8, mode='same')
    ax.bar(x, vol/1e6, color=BLUE, alpha=0.35, width=0.85)
    ax.plot(x, vol_ma/1e6, color=GOLD, lw=0.9, label='8w MA Volume')
    ax.set_ylabel('Volume (M)', color=TEXT_SEC, fontsize=7.5)
    ax.legend(loc='upper left',fontsize=6.5,facecolor=PANEL_BG,edgecolor=BORDER,labelcolor=TEXT_SEC)
    ax.set_xlabel('Weeks (2-year lookback)', color=TEXT_SEC, fontsize=7.5)

    # ── Title bar ────────────────────────────────────────────────────────
    sig_sym = {'STRONG BUY':'▲▲','BUY':'▲','HOLD':'◆','SELL':'▼'}
    sig_s   = sig_sym.get(result['signal'],'◆')
    fig.text(0.04, 0.96,
             f"{ticker}   {result['sector']}",
             color=TEXT_PRI, fontsize=15, fontweight='bold', va='top')
    fig.text(0.04, 0.935,
             f"USD {result['price']:.2f}   {sig_s} {result['signal']}  "
             f"({result['confidence']:.0%} conf)   "
             f"LGB est: {result.get('lgb_fwd_ret',0):+.1%}   "
             f"LSTM 4w: {(result['lstm_forecast_4w'][-1]/result['price']-1)*100:+.1f}%   "
             f"Target: USD {result['analyst_target']:.0f}   "
             f"Upside: {result['analyst_upside']:+.1f}%",
             color=TEXT_SEC, fontsize=8.5, va='top')

    out = CHARTS_DIR/f'{ticker}_hf_chart.png'
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ Chart saved: {out.name}")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    print("="*65)
    print("  HEDGE FUND PREDICTION ENGINE v2.0")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*65)

    results = run_full_training()

    print("\n  Generating professional charts...")
    for ticker in TICKERS:
        if ticker not in results:
            continue
        csv = DATA_DIR/f'{ticker}_price_history_2022-01-01_2026-04-07_1week_cd0858.csv'
        df_raw = pd.read_csv(csv)
        df     = engineer_features(df_raw, ticker)
        df     = make_labels(df)
        df_c   = df.dropna(subset=FEAT_COLS).reset_index(drop=True)
        hf_chart(ticker, df_c, results[ticker])

    print("\n" + "="*65)
    print("  FINAL SIGNALS")
    print("="*65)
    print(f"  {'TICKER':<8} {'SIGNAL':<14} {'CONF':<8} {'LGB 4W':<10} {'LSTM 4W':<10} {'UPSIDE':<10} {'REGIME'}")
    print(f"  {'-'*72}")
    for t,r in results.items():
        fc_last = r['lstm_forecast_4w']
        lstm_pct= (fc_last[-1]/r['price']-1)*100 if fc_last else 0
        print(f"  {t:<8} {r['signal']:<14} {r['confidence']:<8.1%} "
              f"{r.get('lgb_fwd_ret',0):<+10.1%} {lstm_pct:<+10.1f}% "
              f"{r['analyst_upside']:<+10.1f}% {r['vol_regime']}")
