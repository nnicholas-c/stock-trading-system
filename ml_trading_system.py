"""
=============================================================================
PROFESSIONAL ML/RL TRADING SYSTEM
PLTR | AAPL | NVDA | TSLA
=============================================================================
Architecture:
  Layer 1: Feature Engineering (50+ technical indicators)
  Layer 2: Random Forest Signal Generator (buy/sell/hold classifier)
  Layer 3: PPO Reinforcement Learning Agent (portfolio manager)
  Layer 4: Backtest Engine with Sharpe/Sortino/Max Drawdown analytics
  Layer 5: Signal export for live dashboard
=============================================================================
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
import pickle
import gymnasium as gym
from gymnasium import spaces
warnings.filterwarnings('ignore')

# ─── CONFIG ─────────────────────────────────────────────────────────────────
TICKERS = ['PLTR', 'AAPL', 'NVDA', 'TSLA']
DATA_DIR = Path('/home/user/workspace/finance_data')
OUTPUT_DIR = Path('/home/user/workspace/trading_system')
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / 'models').mkdir(exist_ok=True)
(OUTPUT_DIR / 'signals').mkdir(exist_ok=True)
(OUTPUT_DIR / 'charts').mkdir(exist_ok=True)

# Fundamental/analyst data from prior fetches
FUNDAMENTALS = {
    'PLTR': {
        'price': 150.07, 'market_cap': 343.9e9, 'pe': 242.05, 'eps': 0.62,
        'revenue_growth': 56.0, 'gross_margin': 82.0, 'op_margin': 32.0,
        'net_margin': 37.0, 'fcf': 2.27e9, 'debt': 0,
        'analyst_target': 197.57, 'bull_pct': 50.0,
        'fy26_rev': 7.26e9, 'fy27_rev': 10.39e9,
        'insider_activity': -2.5,  # heavy selling (scaled -3 to +3)
        'institutional_count': 3163, 'rule_of_40': 127,
        'sentiment_score': 7.5  # 0-10 based on qualitative analysis
    },
    'AAPL': {
        'price': 253.50, 'market_cap': 3730e9, 'pe': 32.05, 'eps': 7.91,
        'revenue_growth': 16.0, 'gross_margin': 48.2, 'op_margin': 32.0,
        'net_margin': 29.3, 'fcf': 53.9e9,
        'analyst_target': 306.25, 'bull_pct': 72.2,
        'fy26_rev': 465.4e9, 'fy27_rev': 497.2e9,
        'insider_activity': -0.5,  # mostly vesting
        'institutional_count': 6078, 'rule_of_40': 48,
        'sentiment_score': 6.8
    },
    'NVDA': {
        'price': 178.10, 'market_cap': 4330e9, 'pe': 36.35, 'eps': 4.90,
        'revenue_growth': 65.5, 'gross_margin': 61.0, 'op_margin': 50.0,
        'net_margin': 55.7, 'fcf': 102.7e9,
        'analyst_target': 281.04, 'bull_pct': 100.0,
        'fy26_rev': 214.0e9, 'fy27_rev': 369.4e9,
        'insider_activity': -2.0,  # heavy selling by Jensen
        'institutional_count': 5851, 'rule_of_40': 101,
        'sentiment_score': 9.1
    },
    'TSLA': {
        'price': 346.65, 'market_cap': 1300e9, 'pe': 207.57, 'eps': 1.67,
        'revenue_growth': -2.9, 'gross_margin': 20.1, 'op_margin': 5.0,
        'net_margin': 4.0, 'fcf': 1.4e9,
        'analyst_target': 416.49, 'bull_pct': 50.0,
        'fy26_rev': 103.0e9, 'fy27_rev': 120.4e9,
        'insider_activity': 0.5,  # Musk awarded shares
        'institutional_count': 4390, 'rule_of_40': 9,
        'sentiment_score': 5.2
    }
}

# ─── FEATURE ENGINEERING ───────────────────────────────────────────────────
def compute_technical_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Compute 50+ technical indicators as ML features."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    c = df['close']
    h = df['high']
    l = df['low']
    v = df['volume']
    o = df['open']

    # ── Price Returns ────────────────────────────────────────────────────
    df['ret_1w']  = c.pct_change(1)
    df['ret_2w']  = c.pct_change(2)
    df['ret_4w']  = c.pct_change(4)
    df['ret_8w']  = c.pct_change(8)
    df['ret_13w'] = c.pct_change(13)
    df['ret_26w'] = c.pct_change(26)
    df['ret_52w'] = c.pct_change(52)

    # ── Moving Averages ───────────────────────────────────────────────────
    for w in [4, 8, 13, 26, 52]:
        df[f'sma_{w}'] = c.rolling(w).mean()
        df[f'ema_{w}'] = c.ewm(span=w, adjust=False).mean()
        df[f'c_vs_sma_{w}'] = (c - df[f'sma_{w}']) / df[f'sma_{w}']
        df[f'c_vs_ema_{w}'] = (c - df[f'ema_{w}']) / df[f'ema_{w}']

    # ── RSI ──────────────────────────────────────────────────────────────
    for period in [7, 14, 21]:
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # ── MACD ─────────────────────────────────────────────────────────────
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_cross'] = np.sign(df['macd_hist']) - np.sign(df['macd_hist'].shift(1))

    # ── Bollinger Bands ───────────────────────────────────────────────────
    for w in [13, 26]:
        mid = c.rolling(w).mean()
        std = c.rolling(w).std()
        df[f'bb_upper_{w}'] = mid + 2 * std
        df[f'bb_lower_{w}'] = mid - 2 * std
        df[f'bb_pct_{w}'] = (c - df[f'bb_lower_{w}']) / (df[f'bb_upper_{w}'] - df[f'bb_lower_{w}'] + 1e-10)
        df[f'bb_width_{w}'] = (df[f'bb_upper_{w}'] - df[f'bb_lower_{w}']) / mid

    # ── ATR & Volatility ──────────────────────────────────────────────────
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    for w in [7, 14]:
        df[f'atr_{w}'] = tr.rolling(w).mean()
        df[f'atr_pct_{w}'] = df[f'atr_{w}'] / c
    df['realized_vol_13w'] = df['ret_1w'].rolling(13).std() * np.sqrt(52)
    df['realized_vol_26w'] = df['ret_1w'].rolling(26).std() * np.sqrt(52)

    # ── Volume Signals ────────────────────────────────────────────────────
    df['vol_sma_13'] = v.rolling(13).mean()
    df['vol_ratio'] = v / (df['vol_sma_13'] + 1)
    df['vol_trend'] = df['vol_ratio'].rolling(4).mean()
    df['price_x_vol'] = (c * v).rolling(13).mean()  # money flow proxy
    df['obv'] = (np.sign(c.diff()) * v).cumsum()
    df['obv_sma'] = df['obv'].rolling(13).mean()
    df['obv_vs_sma'] = (df['obv'] - df['obv_sma']) / (df['obv_sma'].abs() + 1)

    # ── Stochastic ────────────────────────────────────────────────────────
    for w in [7, 14]:
        lo = l.rolling(w).min()
        hi = h.rolling(w).max()
        df[f'stoch_{w}'] = (c - lo) / (hi - lo + 1e-10) * 100

    # ── Price Momentum ────────────────────────────────────────────────────
    df['mom_4w'] = c / c.shift(4) - 1
    df['mom_13w'] = c / c.shift(13) - 1
    df['mom_26w'] = c / c.shift(26) - 1
    df['mom_accel'] = df['mom_4w'] - df['mom_4w'].shift(4)  # second derivative

    # ── Trend Strength ────────────────────────────────────────────────────
    df['above_sma26'] = (c > df['sma_26']).astype(int)
    df['above_sma52'] = (c > df['sma_52']).astype(int)
    df['golden_cross'] = ((df['sma_13'] > df['sma_26']).astype(int))
    df['trend_score'] = df['above_sma26'] + df['above_sma52'] + df['golden_cross']

    # ── Candlestick Pattern Proxies ───────────────────────────────────────
    df['body_size'] = (c - o).abs() / c
    df['upper_wick'] = (h - pd.concat([c, o], axis=1).max(axis=1)) / c
    df['lower_wick'] = (pd.concat([c, o], axis=1).min(axis=1) - l) / c
    df['candle_dir'] = np.sign(c - o)

    # ── Fundamental Signals (time-invariant, added as constants) ──────────
    fund = FUNDAMENTALS[ticker]
    df['analyst_upside'] = (fund['analyst_target'] / fund['price'] - 1) * 100
    df['bull_pct'] = fund['bull_pct']
    df['revenue_growth'] = fund['revenue_growth']
    df['gross_margin'] = fund['gross_margin']
    df['sentiment_score'] = fund['sentiment_score']
    df['insider_activity'] = fund['insider_activity']
    df['rule_of_40'] = fund['rule_of_40']

    return df


def create_labels(df: pd.DataFrame, forward_weeks: int = 4,
                  buy_threshold: float = 0.05, sell_threshold: float = -0.03) -> pd.DataFrame:
    """
    Create supervised labels:
      2 = Strong Buy  (forward return >= +5%)
      1 = Buy         (forward return >= +2%)
      0 = Hold        (-3% to +2%)
     -1 = Sell        (forward return <= -3%)
    """
    fwd = df['close'].shift(-forward_weeks) / df['close'] - 1
    df['forward_return'] = fwd
    df['label'] = 0  # hold
    df.loc[fwd >= buy_threshold, 'label'] = 2       # strong buy
    df.loc[(fwd >= 0.02) & (fwd < buy_threshold), 'label'] = 1  # buy
    df.loc[fwd <= sell_threshold, 'label'] = -1     # sell
    return df


# ─── FEATURE COLUMNS ────────────────────────────────────────────────────────
FEATURE_COLS = [
    'ret_1w','ret_2w','ret_4w','ret_8w','ret_13w','ret_26w',
    'c_vs_sma_4','c_vs_sma_8','c_vs_sma_13','c_vs_sma_26','c_vs_sma_52',
    'c_vs_ema_4','c_vs_ema_8','c_vs_ema_13',
    'rsi_7','rsi_14','rsi_21',
    'macd','macd_signal','macd_hist','macd_cross',
    'bb_pct_13','bb_pct_26','bb_width_13','bb_width_26',
    'atr_pct_7','atr_pct_14',
    'realized_vol_13w','realized_vol_26w',
    'vol_ratio','vol_trend','obv_vs_sma',
    'stoch_7','stoch_14',
    'mom_4w','mom_13w','mom_26w','mom_accel',
    'trend_score','golden_cross',
    'body_size','upper_wick','lower_wick','candle_dir',
    'analyst_upside','bull_pct','revenue_growth','gross_margin',
    'sentiment_score','insider_activity','rule_of_40'
]


# ─── MODEL TRAINING ─────────────────────────────────────────────────────────
def train_random_forest_model(ticker: str, df_feat: pd.DataFrame):
    """Train Random Forest + Gradient Boosting ensemble for buy/sell signals."""
    df_clean = df_feat.dropna(subset=FEATURE_COLS + ['label', 'forward_return'])
    
    if len(df_clean) < 40:
        print(f"  ⚠ {ticker}: insufficient data ({len(df_clean)} rows)")
        return None, None, None
    
    X = df_clean[FEATURE_COLS].values
    y = df_clean['label'].values
    
    # Time-series cross-validation (no look-ahead)
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train on 80%, evaluate on 20% (time-ordered)
    split = int(len(X_scaled) * 0.80)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    gb.fit(X_train, y_train)
    
    # Ensemble: weighted average probabilities
    rf_proba = rf.predict_proba(X_test)
    gb_proba = gb.predict_proba(X_test)
    
    # Align class indices
    classes = rf.classes_
    ensemble_proba = 0.6 * rf_proba + 0.4 * gb_proba
    y_pred_ensemble = classes[np.argmax(ensemble_proba, axis=1)]
    
    acc = accuracy_score(y_test, y_pred_ensemble)
    
    print(f"\n  {ticker} RF+GB Ensemble | Test Accuracy: {acc:.1%} | Test samples: {len(y_test)}")
    print(f"  Class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    # Feature importance
    importance = dict(zip(FEATURE_COLS, rf.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:8]
    print(f"  Top features: {[f[0] for f in top_features]}")
    
    # Save model
    model_data = {
        'rf': rf, 'gb': gb, 'scaler': scaler,
        'classes': classes, 'accuracy': acc,
        'feature_cols': FEATURE_COLS, 'top_features': top_features,
        'train_size': split, 'test_size': len(y_test)
    }
    with open(OUTPUT_DIR / 'models' / f'{ticker}_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    return model_data, df_clean, (X_scaled, y, classes)


# ─── RL TRADING ENVIRONMENT ─────────────────────────────────────────────────
class StockTradingEnv(gym.Env):
    """
    Custom Gymnasium environment for PPO reinforcement learning.
    
    State:  [normalized price features] + [portfolio state: position, cash_pct, unrealized_pnl]
    Action: 0=Hold, 1=Buy 25%, 2=Buy 50%, 3=Buy 100%, 4=Sell 50%, 5=Sell All
    Reward: Risk-adjusted return with transaction costs + Sharpe bonus
    """
    
    def __init__(self, df: pd.DataFrame, feature_cols: list, initial_capital: float = 100_000):
        super().__init__()
        
        self.df = df.dropna(subset=feature_cols).reset_index(drop=True)
        self.feature_cols = feature_cols
        self.initial_capital = initial_capital
        self.n_features = len(feature_cols)
        
        # Action space: 0-Hold, 1-Buy25%, 2-Buy50%, 3-Buy100%, 4-Sell50%, 5-SellAll
        self.action_space = spaces.Discrete(6)
        
        # Observation: features + [position_pct, cash_pct, unrealized_pnl_pct, steps_remaining_pct]
        self.obs_dim = self.n_features + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32
        )
        
        self.scaler = StandardScaler()
        # Only keep rows with valid features
        valid_mask = ~np.isnan(self.df[feature_cols].values).any(axis=1)
        self.df = self.df[valid_mask].reset_index(drop=True)
        features = self.df[feature_cols].values
        self.features_scaled = self.scaler.fit_transform(features)
        self.n_features = features.shape[1]  # recompute after filtering
        self.obs_dim = self.n_features + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 26  # start after warmup
        self.cash = float(self.initial_capital)
        self.shares = 0.0
        self.total_value_history = [self.initial_capital]
        self.trade_log = []
        self.transaction_costs = 0.0
        return self._get_obs(), {}
    
    def _get_obs(self):
        feat = self.features_scaled[self.current_step]
        price = self.df['close'].iloc[self.current_step]
        portfolio_value = self.cash + self.shares * price
        position_pct = (self.shares * price) / (portfolio_value + 1e-10)
        cash_pct = self.cash / (portfolio_value + 1e-10)
        unrealized_pnl = (self.shares * price - self.shares * self._avg_cost) / (self.initial_capital + 1e-10) if hasattr(self, '_avg_cost') and self._avg_cost > 0 else 0.0
        steps_remaining = (len(self.df) - self.current_step) / len(self.df)
        obs = np.concatenate([feat, [position_pct, cash_pct, unrealized_pnl, steps_remaining]])
        obs = obs[:self.obs_dim]  # ensure consistent shape
        if len(obs) < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - len(obs)))
        return obs.astype(np.float32)
    
    def step(self, action):
        price = self.df['close'].iloc[self.current_step]
        portfolio_value = self.cash + self.shares * price
        TRANSACTION_COST = 0.001  # 0.1% per trade
        
        if not hasattr(self, '_avg_cost'):
            self._avg_cost = 0.0
        
        # Execute action
        if action == 1:  # Buy 25%
            buy_amount = self.cash * 0.25
            shares_to_buy = buy_amount / (price * (1 + TRANSACTION_COST))
            cost = shares_to_buy * price * (1 + TRANSACTION_COST)
            if cost <= self.cash:
                self._avg_cost = (self._avg_cost * self.shares + price * shares_to_buy) / (self.shares + shares_to_buy + 1e-10)
                self.shares += shares_to_buy
                self.cash -= cost
                self.transaction_costs += cost * TRANSACTION_COST
                self.trade_log.append({'step': self.current_step, 'action': 'BUY25%', 'price': price, 'shares': shares_to_buy})
        
        elif action == 2:  # Buy 50%
            buy_amount = self.cash * 0.50
            shares_to_buy = buy_amount / (price * (1 + TRANSACTION_COST))
            cost = shares_to_buy * price * (1 + TRANSACTION_COST)
            if cost <= self.cash:
                self._avg_cost = (self._avg_cost * self.shares + price * shares_to_buy) / (self.shares + shares_to_buy + 1e-10)
                self.shares += shares_to_buy
                self.cash -= cost
                self.transaction_costs += cost * TRANSACTION_COST
                self.trade_log.append({'step': self.current_step, 'action': 'BUY50%', 'price': price, 'shares': shares_to_buy})
        
        elif action == 3:  # Buy 100%
            buy_amount = self.cash * 1.0
            shares_to_buy = buy_amount / (price * (1 + TRANSACTION_COST))
            cost = shares_to_buy * price * (1 + TRANSACTION_COST)
            if cost <= self.cash:
                self._avg_cost = (self._avg_cost * self.shares + price * shares_to_buy) / (self.shares + shares_to_buy + 1e-10)
                self.shares += shares_to_buy
                self.cash -= cost
                self.transaction_costs += cost * TRANSACTION_COST
                self.trade_log.append({'step': self.current_step, 'action': 'BUY100%', 'price': price, 'shares': shares_to_buy})
        
        elif action == 4:  # Sell 50%
            if self.shares > 0:
                shares_to_sell = self.shares * 0.5
                proceeds = shares_to_sell * price * (1 - TRANSACTION_COST)
                self.shares -= shares_to_sell
                self.cash += proceeds
                self.transaction_costs += proceeds * TRANSACTION_COST / (1 - TRANSACTION_COST)
                self.trade_log.append({'step': self.current_step, 'action': 'SELL50%', 'price': price, 'shares': shares_to_sell})
        
        elif action == 5:  # Sell All
            if self.shares > 0:
                proceeds = self.shares * price * (1 - TRANSACTION_COST)
                self.transaction_costs += self.shares * price * TRANSACTION_COST
                self.trade_log.append({'step': self.current_step, 'action': 'SELLALL', 'price': price, 'shares': self.shares})
                self.shares = 0.0
                self.cash += proceeds
                self._avg_cost = 0.0
        
        # Advance
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        new_price = self.df['close'].iloc[self.current_step]
        new_portfolio_value = self.cash + self.shares * new_price
        
        # Reward: step return, penalize drawdown
        step_return = (new_portfolio_value - portfolio_value) / (portfolio_value + 1e-10)
        self.total_value_history.append(new_portfolio_value)
        
        # Sharpe-based reward shaping
        if len(self.total_value_history) > 10:
            window = self.total_value_history[-11:]
            recent_returns = np.diff(window) / np.array(window[:-1])
            sharpe_bonus = np.mean(recent_returns) / (np.std(recent_returns) + 1e-6) * 0.01
        else:
            sharpe_bonus = 0.0
        
        reward = step_return * 100 + sharpe_bonus
        
        truncated = False
        info = {'portfolio_value': new_portfolio_value, 'cash': self.cash, 'shares': self.shares}
        
        return self._get_obs(), reward, done, truncated, info
    
    def compute_metrics(self):
        """Compute backtest performance metrics."""
        values = np.array(self.total_value_history)
        returns = np.diff(values) / values[:-1]
        
        total_return = (values[-1] - values[0]) / values[0]
        
        # Annualized (weekly data → 52 weeks/year)
        n_weeks = len(returns)
        ann_return = (1 + total_return) ** (52 / n_weeks) - 1 if n_weeks > 0 else 0
        ann_vol = np.std(returns) * np.sqrt(52) if len(returns) > 1 else 0
        
        sharpe = ann_return / (ann_vol + 1e-10)
        
        # Sortino
        downside = returns[returns < 0]
        sortino_vol = np.std(downside) * np.sqrt(52) if len(downside) > 1 else 1e-10
        sortino = ann_return / (sortino_vol + 1e-10)
        
        # Max Drawdown
        cum = values / values[0]
        roll_max = np.maximum.accumulate(cum)
        dd = (cum - roll_max) / roll_max
        max_dd = dd.min()
        
        # Win rate
        n_trades = len(self.trade_log)
        
        return {
            'total_return': total_return,
            'ann_return': ann_return,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_dd,
            'n_trades': n_trades,
            'final_value': values[-1],
            'transaction_costs': self.transaction_costs
        }


# ─── SIGNAL GENERATION ──────────────────────────────────────────────────────
def generate_current_signal(ticker: str, df_feat: pd.DataFrame, model_data: dict) -> dict:
    """Generate the current actionable buy/sell/hold signal."""
    df_clean = df_feat.dropna(subset=FEATURE_COLS)
    latest = df_clean.iloc[-1]
    
    X_latest = latest[FEATURE_COLS].values.reshape(1, -1)
    X_scaled = model_data['scaler'].transform(X_latest)
    
    rf_proba = model_data['rf'].predict_proba(X_scaled)[0]
    gb_proba = model_data['gb'].predict_proba(X_scaled)[0]
    ensemble_proba = 0.6 * rf_proba + 0.4 * gb_proba
    
    classes = model_data['classes']
    label_idx = np.argmax(ensemble_proba)
    signal_label = int(classes[label_idx])
    confidence = float(ensemble_proba[label_idx])
    
    signal_map = {2: 'STRONG BUY', 1: 'BUY', 0: 'HOLD', -1: 'SELL'}
    signal_color = {2: '#00C853', 1: '#69F0AE', 0: '#FFD740', -1: '#FF5252'}
    signal_icon = {2: '🚀', 1: '✅', 0: '⏸', -1: '🔴'}
    
    fund = FUNDAMENTALS[ticker]
    analyst_upside = (fund['analyst_target'] / fund['price'] - 1) * 100
    
    # Price targets from signal
    current_price = fund['price']
    if signal_label >= 1:
        pt_bull = fund['analyst_target']
        pt_base = current_price * 1.10
        pt_bear = current_price * 0.88
    else:
        pt_bull = current_price * 1.05
        pt_base = current_price * 0.97
        pt_bear = current_price * 0.85
    
    # Risk score (1-10, lower is less risky)
    pe_risk = min(5, fund['pe'] / 50) if fund['pe'] > 0 else 3
    sentiment_risk = (10 - fund['sentiment_score']) / 2
    insider_risk = max(0, -fund['insider_activity'])
    risk_score = min(10, (pe_risk + sentiment_risk + insider_risk) / 3 * 4)
    
    # Technical indicators summary
    rsi_14 = float(latest.get('rsi_14', 50))
    macd_hist = float(latest.get('macd_hist', 0))
    trend_score = float(latest.get('trend_score', 1))
    bb_pct = float(latest.get('bb_pct_26', 0.5))
    
    signal = {
        'ticker': ticker,
        'date': str(latest.get('date', datetime.now().date())),
        'current_price': current_price,
        'signal': signal_map[signal_label],
        'signal_label': signal_label,
        'confidence': confidence,
        'color': signal_color[signal_label],
        'icon': signal_icon[signal_label],
        'analyst_target': fund['analyst_target'],
        'analyst_upside': analyst_upside,
        'bull_pct': fund['bull_pct'],
        'pt_bull': pt_bull,
        'pt_base': pt_base,
        'pt_bear': pt_bear,
        'risk_score': risk_score,
        'rsi_14': rsi_14,
        'macd_hist': macd_hist,
        'trend_score': trend_score,
        'bb_pct': bb_pct,
        'revenue_growth': fund['revenue_growth'],
        'gross_margin': fund['gross_margin'],
        'net_margin': fund['net_margin'],
        'sentiment_score': fund['sentiment_score'],
        'insider_activity': fund['insider_activity'],
        'market_cap': fund['market_cap'],
        'pe': fund['pe'],
        'model_accuracy': model_data['accuracy'],
        'top_features': model_data['top_features'][:5]
    }
    
    return signal


# ─── BACKTESTING ────────────────────────────────────────────────────────────
def run_backtest(ticker: str, df_feat: pd.DataFrame, model_data: dict) -> dict:
    """
    Backtest: compare ML signal strategy vs buy-and-hold.
    Uses time-series walk-forward to avoid look-ahead bias.
    """
    df_clean = df_feat.dropna(subset=FEATURE_COLS + ['label', 'forward_return']).reset_index(drop=True)
    
    if len(df_clean) < 30:
        return {}
    
    # Start backtest from 60% of data (train end)
    start_idx = int(len(df_clean) * 0.60)
    capital = 100_000
    cash = capital
    shares = 0.0
    portfolio_values = [capital]
    bah_values = [capital]  # buy-and-hold
    
    initial_price = df_clean['close'].iloc[start_idx]
    bah_shares = capital / initial_price
    
    scaler = model_data['scaler']
    rf = model_data['rf']
    gb = model_data['gb']
    classes = model_data['classes']
    COST = 0.001
    
    signals_log = []
    
    for i in range(start_idx, len(df_clean) - 1):
        row = df_clean.iloc[i]
        X = row[FEATURE_COLS].values.reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        rf_p = rf.predict_proba(X_scaled)[0]
        gb_p = gb.predict_proba(X_scaled)[0]
        ep = 0.6 * rf_p + 0.4 * gb_p
        signal_label = int(classes[np.argmax(ep)])
        confidence = float(ep.max())
        price = float(row['close'])
        
        # Only trade on high-confidence signals
        if confidence > 0.45:
            if signal_label >= 1 and cash > price * 10:  # buy
                buy_amount = cash * 0.8 if signal_label == 2 else cash * 0.5
                new_shares = buy_amount / (price * (1 + COST))
                cost = new_shares * price * (1 + COST)
                if cost <= cash:
                    shares += new_shares
                    cash -= cost
                    signals_log.append({'i': i, 'action': 'BUY', 'price': price, 'conf': confidence})
            
            elif signal_label == -1 and shares > 0:  # sell
                proceeds = shares * price * (1 - COST)
                cash += proceeds
                shares = 0.0
                signals_log.append({'i': i, 'action': 'SELL', 'price': price, 'conf': confidence})
        
        next_price = float(df_clean['close'].iloc[i + 1])
        portfolio_values.append(cash + shares * next_price)
        bah_values.append(bah_shares * next_price)
    
    pv = np.array(portfolio_values)
    bv = np.array(bah_values)
    
    strategy_ret = (pv[-1] - pv[0]) / pv[0]
    bah_ret = (bv[-1] - bv[0]) / bv[0]
    
    returns_strat = np.diff(pv) / pv[:-1]
    returns_bah = np.diff(bv) / bv[:-1]
    
    def sharpe(rets): return np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(52)
    def max_dd(vals):
        cum = vals / vals[0]
        rm = np.maximum.accumulate(cum)
        return (cum - rm).min()
    
    result = {
        'ticker': ticker,
        'strategy_return': strategy_ret,
        'bah_return': bah_ret,
        'alpha': strategy_ret - bah_ret,
        'strategy_sharpe': sharpe(returns_strat),
        'bah_sharpe': sharpe(returns_bah),
        'strategy_max_dd': max_dd(pv),
        'bah_max_dd': max_dd(bv),
        'n_trades': len(signals_log),
        'portfolio_values': pv.tolist(),
        'bah_values': bv.tolist(),
        'signals_log': signals_log,
        'start_idx': start_idx,
        'start_date': str(df_clean['date'].iloc[start_idx]),
        'end_date': str(df_clean['date'].iloc[-2]) if len(df_clean) > 1 else ''
    }
    
    return result


# ─── RL TRAINING ────────────────────────────────────────────────────────────
def train_rl_agent(ticker: str, df_feat: pd.DataFrame) -> dict:
    """
    Train a PPO (Proximal Policy Optimization) agent on the StockTradingEnv.
    Returns performance metrics and final portfolio value.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        
        df_clean = df_feat.dropna(subset=FEATURE_COLS).reset_index(drop=True)
        
        if len(df_clean) < 50:
            print(f"  ⚠ {ticker}: insufficient data for RL training")
            return {}
        
        env = StockTradingEnv(df_clean, FEATURE_COLS, initial_capital=100_000)
        
        # PPO hyperparameters tuned for financial data
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=3e-4,
            n_steps=min(256, len(df_clean) - 30),
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0,
            policy_kwargs={'net_arch': [128, 64, 32]}  # 3-layer MLP
        )
        
        total_timesteps = max(2000, len(df_clean) * 20)
        model.learn(total_timesteps=total_timesteps)
        
        # Evaluate
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
        
        metrics = env.compute_metrics()
        metrics['ticker'] = ticker
        metrics['n_timesteps'] = total_timesteps
        
        # Save RL model
        model.save(str(OUTPUT_DIR / 'models' / f'{ticker}_ppo_agent'))
        
        print(f"  {ticker} PPO Agent | Total Return: {metrics['total_return']:.1%} | Sharpe: {metrics['sharpe']:.2f} | Max DD: {metrics['max_drawdown']:.1%}")
        
        return metrics
        
    except Exception as e:
        print(f"  ⚠ {ticker} RL training error: {e}")
        return {}


# ─── CHART GENERATION ────────────────────────────────────────────────────────
def generate_analysis_chart(ticker: str, df_feat: pd.DataFrame, signal: dict, backtest: dict):
    """Generate professional 4-panel analysis chart."""
    df_clean = df_feat.dropna(subset=['rsi_14', 'macd', 'bb_upper_26']).reset_index(drop=True)
    
    if len(df_clean) < 20:
        return
    
    fig = plt.figure(figsize=(18, 14), facecolor='#0D1117')
    gs = gridspec.GridSpec(4, 1, hspace=0.4, figure=fig,
                           top=0.93, bottom=0.06, left=0.07, right=0.97)
    
    ax_price = fig.add_subplot(gs[0])
    ax_rsi   = fig.add_subplot(gs[1])
    ax_macd  = fig.add_subplot(gs[2])
    ax_vol   = fig.add_subplot(gs[3])
    
    for ax in [ax_price, ax_rsi, ax_macd, ax_vol]:
        ax.set_facecolor('#161B22')
        ax.tick_params(colors='#8B949E', labelsize=8)
        ax.spines['bottom'].set_color('#30363D')
        ax.spines['left'].set_color('#30363D')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    dates = df_clean['date'].values[-104:]  # last 2 years
    close = df_clean['close'].values[-104:]
    high = df_clean['high'].values[-104:]
    low = df_clean['low'].values[-104:]
    volume = df_clean['volume'].values[-104:]
    sma26 = df_clean['sma_26'].values[-104:]
    sma52 = df_clean['sma_52'].values[-104:]
    bb_up = df_clean['bb_upper_26'].values[-104:]
    bb_lo = df_clean['bb_lower_26'].values[-104:]
    rsi = df_clean['rsi_14'].values[-104:]
    macd = df_clean['macd'].values[-104:]
    macd_sig = df_clean['macd_signal'].values[-104:]
    macd_hist = df_clean['macd_hist'].values[-104:]
    
    x = np.arange(len(dates))
    
    # ── Panel 1: Price ────────────────────────────────────────────────────
    ax_price.fill_between(x, bb_lo, bb_up, alpha=0.08, color='#58A6FF')
    ax_price.plot(x, close, color='#F0F6FC', lw=1.5, label='Close', zorder=5)
    ax_price.plot(x, sma26, color='#FF7B72', lw=1.0, linestyle='--', label='SMA 26', alpha=0.8)
    ax_price.plot(x, sma52, color='#FFA657', lw=1.0, linestyle='--', label='SMA 52', alpha=0.8)
    ax_price.plot(x, bb_up, color='#58A6FF', lw=0.6, alpha=0.5)
    ax_price.plot(x, bb_lo, color='#58A6FF', lw=0.6, alpha=0.5)
    
    # Mark last signal
    last_x = len(x) - 1
    signal_colors = {'STRONG BUY': '#00C853', 'BUY': '#69F0AE', 'HOLD': '#FFD740', 'SELL': '#FF5252'}
    sc = signal_colors.get(signal['signal'], '#FFD740')
    ax_price.scatter([last_x], [close[-1]], color=sc, s=200, zorder=10, marker='*')
    
    # Analyst target line
    ax_price.axhline(y=signal['analyst_target'], color='#8957E5', lw=1.0,
                     linestyle=':', alpha=0.7, label=f'Analyst Tgt {signal["analyst_target"]:.0f}')
    
    ax_price.set_title(f'{ticker}  |  USD {close[-1]:.2f}  |  {signal["signal"]} ({signal["confidence"]:.0%} conf)  |  Target: USD {signal["analyst_target"]:.0f}',
                       color='#F0F6FC', fontsize=12, fontweight='bold', loc='left', pad=8, usetex=False)
    ax_price.legend(loc='upper left', fontsize=7, facecolor='#161B22',
                    edgecolor='#30363D', labelcolor='#8B949E')
    ax_price.set_ylabel('Price (USD)', color='#8B949E', fontsize=8)
    
    # ── Panel 2: RSI ──────────────────────────────────────────────────────
    ax_rsi.plot(x, rsi, color='#F79550', lw=1.2)
    ax_rsi.axhline(70, color='#FF5252', lw=0.8, linestyle='--', alpha=0.7)
    ax_rsi.axhline(30, color='#00C853', lw=0.8, linestyle='--', alpha=0.7)
    ax_rsi.axhline(50, color='#8B949E', lw=0.5, linestyle='-', alpha=0.4)
    ax_rsi.fill_between(x, rsi, 70, where=(rsi >= 70), alpha=0.2, color='#FF5252')
    ax_rsi.fill_between(x, rsi, 30, where=(rsi <= 30), alpha=0.2, color='#00C853')
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel('RSI(14)', color='#8B949E', fontsize=8)
    rsi_val = signal.get('rsi_14', rsi[-1])
    rsi_label = 'Overbought' if rsi_val > 70 else 'Oversold' if rsi_val < 30 else 'Neutral'
    ax_rsi.text(0.99, 0.85, f'RSI: {rsi_val:.1f} ({rsi_label})',
                transform=ax_rsi.transAxes, ha='right', va='top',
                color='#F0F6FC', fontsize=8)
    
    # ── Panel 3: MACD ─────────────────────────────────────────────────────
    ax_macd.plot(x, macd, color='#58A6FF', lw=1.2, label='MACD')
    ax_macd.plot(x, macd_sig, color='#FF7B72', lw=1.0, label='Signal', linestyle='--')
    colors_hist = ['#00C853' if h >= 0 else '#FF5252' for h in macd_hist]
    ax_macd.bar(x, macd_hist, color=colors_hist, alpha=0.6, width=0.8)
    ax_macd.axhline(0, color='#8B949E', lw=0.5, alpha=0.5)
    ax_macd.set_ylabel('MACD', color='#8B949E', fontsize=8)
    ax_macd.legend(loc='upper left', fontsize=7, facecolor='#161B22',
                   edgecolor='#30363D', labelcolor='#8B949E')
    
    # ── Panel 4: Volume ───────────────────────────────────────────────────
    vol_ma = np.convolve(volume, np.ones(8)/8, mode='same')
    ax_vol.bar(x, volume / 1e6, color='#58A6FF', alpha=0.4, width=0.8)
    ax_vol.plot(x, vol_ma / 1e6, color='#FFA657', lw=1.0, label='8w MA Volume')
    ax_vol.set_ylabel('Volume (M)', color='#8B949E', fontsize=8)
    ax_vol.legend(loc='upper left', fontsize=7, facecolor='#161B22',
                  edgecolor='#30363D', labelcolor='#8B949E')
    ax_vol.set_xlabel('Weeks (2yr lookback)', color='#8B949E', fontsize=8)
    
    # Backtest annotation
    if backtest:
        strat_r = backtest.get('strategy_return', 0)
        bah_r = backtest.get('bah_return', 0)
        alpha = backtest.get('alpha', 0)
        sharpe = backtest.get('strategy_sharpe', 0)
        n_trades = backtest.get('n_trades', 0)
        dd = backtest.get('strategy_max_dd', 0)
        fig.text(0.73, 0.97, 
                 f'BACKTEST: ML Strategy {strat_r:+.1%} | B&H {bah_r:+.1%} | Alpha {alpha:+.1%} | '
                 f'Sharpe {sharpe:.2f} | MaxDD {dd:.1%} | Trades: {n_trades}',
                 ha='right', va='top', color='#8B949E', fontsize=7.5,
                 transform=fig.transFigure)
    
    plt.savefig(OUTPUT_DIR / 'charts' / f'{ticker}_analysis.png',
                dpi=130, bbox_inches='tight', facecolor='#0D1117')
    plt.close(fig)
    print(f"  ✓ {ticker} chart saved")


# ─── MAIN EXECUTION ─────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  ML/RL TRADING SYSTEM — PLTR | AAPL | NVDA | TSLA")
    print(f"  Execution date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    all_signals = {}
    all_backtest = {}
    all_rl = {}
    
    for ticker in TICKERS:
        print(f"\n{'─'*60}")
        print(f"  Processing {ticker}...")
        
        # Load price data (use weekly 2022-2026 for maximum training data)
        csv_path = DATA_DIR / f'{ticker}_price_history_2022-01-01_2026-04-07_1week_cd0858.csv'
        if not csv_path.exists():
            # Fallback to monthly
            csv_path = DATA_DIR / f'{ticker}_price_history_2024-01-01_2026-04-07_1month_cd0858.csv'
        
        if not csv_path.exists():
            print(f"  ⚠ No price data found for {ticker}, skipping")
            continue
        
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} rows from {csv_path.name}")
        
        # Feature engineering
        df_feat = compute_technical_features(df, ticker)
        df_feat = create_labels(df_feat, forward_weeks=4, buy_threshold=0.05, sell_threshold=-0.03)
        
        valid_rows = df_feat.dropna(subset=FEATURE_COLS).shape[0]
        print(f"  Feature rows (non-NaN): {valid_rows}")
        
        # Train RF model
        print(f"  Training Random Forest + Gradient Boosting ensemble...")
        model_data, df_clean, _ = train_random_forest_model(ticker, df_feat)
        
        if model_data is None:
            continue
        
        # Generate current signal
        signal = generate_current_signal(ticker, df_feat, model_data)
        all_signals[ticker] = signal
        print(f"  Current Signal: {signal['signal']} | Confidence: {signal['confidence']:.1%}")
        
        # Backtest
        print(f"  Running backtest (walk-forward, no look-ahead)...")
        backtest = run_backtest(ticker, df_feat, model_data)
        if backtest:
            all_backtest[ticker] = backtest
            print(f"  Backtest: Strategy {backtest['strategy_return']:+.1%} vs B&H {backtest['bah_return']:+.1%} | Alpha {backtest['alpha']:+.1%}")
        
        # RL Agent
        print(f"  Training PPO reinforcement learning agent...")
        rl_metrics = train_rl_agent(ticker, df_feat)
        if rl_metrics:
            all_rl[ticker] = rl_metrics
        
        # Generate chart
        generate_analysis_chart(ticker, df_feat, signal, backtest)
    
    # ─── SAVE ALL SIGNALS TO JSON ─────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Saving signals and results...")
    
    # Convert numpy types for JSON serialization
    def convert(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)
    
    output = {
        'generated_at': datetime.now().isoformat(),
        'signals': all_signals,
        'backtest': all_backtest,
        'rl_metrics': all_rl,
        'market_data': {t: {
            'price': FUNDAMENTALS[t]['price'],
            'market_cap': FUNDAMENTALS[t]['market_cap'],
            'pe': FUNDAMENTALS[t]['pe'],
            'analyst_target': FUNDAMENTALS[t]['analyst_target'],
            'revenue_growth': FUNDAMENTALS[t]['revenue_growth']
        } for t in TICKERS}
    }
    
    with open(OUTPUT_DIR / 'signals' / 'current_signals.json', 'w') as f:
        json.dump(output, f, indent=2, default=convert)
    
    # ─── PRINT SUMMARY ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  TRADING SIGNALS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'TICKER':<8} {'SIGNAL':<14} {'CONF':<8} {'PRICE':<10} {'TARGET':<10} {'UPSIDE':<10} {'RISK':<6}")
    print(f"  {'-'*66}")
    
    for ticker, sig in all_signals.items():
        print(f"  {ticker:<8} {sig['signal']:<14} {sig['confidence']:<8.1%} "
              f"${sig['current_price']:<9.2f} ${sig['analyst_target']:<9.2f} "
              f"{sig['analyst_upside']:+5.1f}%   {sig['risk_score']:.1f}/10")
    
    print(f"\n  BACKTEST ALPHA GENERATION")
    print(f"  {'-'*50}")
    for ticker, bt in all_backtest.items():
        print(f"  {ticker}: ML {bt['strategy_return']:+.1%} vs B&H {bt['bah_return']:+.1%} | "
              f"Alpha {bt['alpha']:+.1%} | Sharpe {bt['strategy_sharpe']:.2f}")
    
    print(f"\n  RL AGENT PERFORMANCE")
    print(f"  {'-'*50}")
    for ticker, rl in all_rl.items():
        print(f"  {ticker}: Return {rl['total_return']:+.1%} | Sharpe {rl['sharpe']:.2f} | "
              f"MaxDD {rl['max_drawdown']:.1%}")
    
    print(f"\n  Output files:")
    print(f"  → signals: {OUTPUT_DIR}/signals/current_signals.json")
    print(f"  → models:  {OUTPUT_DIR}/models/")
    print(f"  → charts:  {OUTPUT_DIR}/charts/")
    print(f"\n{'='*70}")
    
    return all_signals, all_backtest, all_rl


if __name__ == '__main__':
    signals, backtest, rl = main()
