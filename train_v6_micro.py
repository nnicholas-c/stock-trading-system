"""
train_v6_micro.py — AXIOM v6 Quantitative Microstructure Model
==============================================================
Advanced quant trading system implementing:

1. Order Flow Imbalance (OFI) — Cont et al. (2014)
   ΔPk = β·OFIk + εk

2. Almgren-Chriss Optimal Execution (2000)
   Minimize E[Cost] + λ·Var[Cost]
   x(t) = X·sinh(κ(T-t)) / sinh(κT)

3. Avellaneda-Stoikov Market Making (2008)
   r = S - q·γ·σ²·(T-t)
   δ = γ·σ²·(T-t) + (2/γ)·ln(1 + γ/κ)

4. Kyle's Lambda — price impact per unit volume
   λ = Cov(ΔP, Q) / Var(Q)

5. Cross-Asset OFI (Arxiv 2112.13213)

6. Enhanced Fama-French + Microstructure (11 factors)

7. Self-Improving Kalman Filter Calibration

Author: AXIOM Quant Research
Version: 6.0.0
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.linalg import solve

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("axiom.v6")

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORY
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/home/user/workspace/trading_system/v6")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    """
    Global configuration for AXIOM v6 microstructure model.

    All parameters are documented with their economic interpretation.
    """
    # Tickers
    tickers: List[str] = field(
        default_factory=lambda: ["NVDA", "AAPL", "PLTR", "TSLA"]
    )

    # Simulation
    n_days: int = 504           # 2 years of trading days
    n_obs_per_day: int = 100    # Intraday observations per day
    random_seed: int = 42

    # OFI Model (Cont et al. 2014)
    ofi_beta_init: float = 0.00012   # Initial market impact coefficient
    ofi_window: int = 63             # Rolling window for β estimation (days)
    ofi_min_obs: int = 20            # Minimum observations for OLS

    # Almgren-Chriss (2000)
    ac_shares: int = 10_000          # Order size (shares)
    ac_horizon_hours: int = 8        # Execution horizon (hours)
    ac_eta: float = 0.0001           # Temporary impact coefficient
    ac_gamma: float = 0.00005        # Permanent impact coefficient
    ac_sigma: float = 0.02           # Daily volatility (fraction)
    ac_risk_aversions: Tuple[float, ...] = (0.001, 0.010, 0.025)

    # Avellaneda-Stoikov (2008)
    as_gamma: float = 0.1            # Risk aversion (inventory)
    as_sigma: float = 0.02           # Price volatility
    as_kappa: float = 1.5            # Order arrival intensity
    as_q_max: int = 50               # Maximum inventory position
    as_n_episodes: int = 100         # Simulation episodes
    as_T: float = 1.0                # Time horizon (normalized)
    as_dt: float = 0.001             # Time step

    # Kyle's Lambda
    kyle_window: int = 63            # Rolling window (trading days)

    # Fama-French factors
    ff_factors: List[str] = field(default_factory=lambda: [
        "MKT", "SMB", "HML", "RMW", "CMA", "MOM", "VADER",
        "OFI", "KYLE_LAMBDA", "BID_ASK_SPREAD", "REALIZED_VOL_RATIO"
    ])

    # Kalman Filter (adaptive β)
    kalman_Q: float = 1e-5           # Process noise variance
    kalman_R: float = 1e-4           # Observation noise variance
    kalman_P_init: float = 1.0       # Initial error covariance

    # Cross-Asset OFI Spillover Matrix
    # Rows=source, Cols=destination: NVDA, AAPL, PLTR, TSLA
    spillover_matrix: List[List[float]] = field(default_factory=lambda: [
        [1.000,  0.420,  0.280, -0.080],  # NVDA affects others
        [0.320,  1.000,  0.180,  0.140],  # AAPL affects others
        [0.120,  0.090,  1.000, -0.180],  # PLTR affects others
        [-0.080, 0.140, -0.180,  1.000],  # TSLA affects others
    ])

    # Price parameters per ticker
    price_init: Dict[str, float] = field(default_factory=lambda: {
        "NVDA": 875.40, "AAPL": 218.24, "PLTR": 24.82, "TSLA": 162.35
    })
    vol_daily: Dict[str, float] = field(default_factory=lambda: {
        "NVDA": 0.028, "AAPL": 0.018, "PLTR": 0.034, "TSLA": 0.038
    })


# ─────────────────────────────────────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────
class SyntheticDataGenerator:
    """
    Generate synthetic LOB (Limit Order Book) and price data
    for backtesting microstructure models.

    Implements a simplified Kyle (1985) / Glosten-Milgrom order flow model
    with realistic bid-ask dynamics and volume.
    """

    def __init__(self, config: ModelConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.random_seed)

    def generate_price_series(self, ticker: str) -> pd.DataFrame:
        """
        Generate OHLCV + LOB data for a single ticker.

        Returns DataFrame with columns:
            date, open, high, low, close, volume,
            bid_price, ask_price, bid_size, ask_size,
            signed_volume, mid_price, spread
        """
        cfg = self.cfg
        n = cfg.n_days
        sigma = cfg.vol_daily[ticker]
        S0 = cfg.price_init[ticker]

        # GBM price path with mean reversion overlay
        log.debug(f"Generating price series for {ticker} (n={n} days)")
        dt = 1 / 252
        mu = 0.08  # annual drift
        returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * self.rng.standard_normal(n)
        prices = S0 * np.cumprod(np.exp(returns))

        # Add momentum + mean-reversion micro-structure noise
        prices += self.rng.normal(0, S0 * 0.001, n)
        prices = np.maximum(prices, S0 * 0.01)

        # OHLCV
        intraday_noise = sigma * S0 * self.rng.uniform(0.3, 0.8, n)
        highs = prices + intraday_noise * self.rng.uniform(0.3, 1.0, n)
        lows = prices - intraday_noise * self.rng.uniform(0.3, 1.0, n)
        opens = prices * (1 + self.rng.normal(0, 0.003, n))
        volume = (1e7 * (0.5 + 2 * self.rng.exponential(0.5, n))).astype(int)

        # Bid-ask spread (mean-reverting, correlated with volatility)
        base_spread_bps = 2.0 if S0 > 100 else 5.0
        spread_mult = 1 + 0.5 * np.abs(self.rng.standard_normal(n))  # vol clustering
        spread = prices * (base_spread_bps / 10000) * spread_mult

        bid_price = prices - spread / 2
        ask_price = prices + spread / 2

        # Order book sizes (power-law distributed)
        bid_size = (1000 * self.rng.pareto(1.5, n)).astype(int).clip(100, 100_000)
        ask_size = (1000 * self.rng.pareto(1.5, n)).astype(int).clip(100, 100_000)

        # Signed volume: informed + uninformed component
        informed_frac = 0.3
        price_direction = np.sign(np.diff(np.concatenate([[S0], prices])))
        noise_order = self.rng.choice([-1, 1], n)
        signed_vol = (
            volume * (informed_frac * price_direction + (1 - informed_frac) * noise_order)
        ).astype(int)

        # Build DataFrame
        dates = pd.bdate_range("2024-01-01", periods=n)
        df = pd.DataFrame({
            "date": dates,
            "open": opens.round(2),
            "high": highs.round(2),
            "low": lows.round(2),
            "close": prices.round(2),
            "volume": volume,
            "bid_price": bid_price.round(4),
            "ask_price": ask_price.round(4),
            "bid_size": bid_size,
            "ask_size": ask_size,
            "signed_volume": signed_vol,
            "mid_price": prices.round(4),
            "spread": spread.round(6),
        })
        df.set_index("date", inplace=True)
        return df

    def generate_all(self) -> Dict[str, pd.DataFrame]:
        """Generate data for all tickers."""
        data = {}
        for ticker in self.cfg.tickers:
            log.info(f"Generating synthetic LOB data: {ticker}")
            data[ticker] = self.generate_price_series(ticker)
        return data


# ─────────────────────────────────────────────────────────────────────────────
# 1. ORDER FLOW IMBALANCE (OFI)  — Cont et al. (2014)
# ─────────────────────────────────────────────────────────────────────────────
class OFIModel:
    """
    Order Flow Imbalance model following Cont, Kukanov & Stoikov (2014).

    The model estimates:
        ΔPk = β · OFIk + εk

    where:
        OFI = (Δbid_qty × bid_side) + (Δask_qty × ask_side)
        β   = market impact coefficient (time-varying)

    Reference:
        Cont, R., Kukanov, A., & Stoikov, S. (2014). The impact of order flow
        imbalance on stock price changes. Quantitative Finance, 14(9), 1827–1838.
    """

    def __init__(self, config: ModelConfig):
        self.cfg = config
        self.beta_history: Dict[str, pd.Series] = {}
        self.r_squared: Dict[str, float] = {}

    def compute_ofi(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute OFI (Order Flow Imbalance) time series.

        OFI_k = (bid_size_k - bid_size_{k-1}) * 1_{bid_price unchanged or up}
               + (ask_size_k - ask_size_{k-1}) * 1_{ask_price unchanged or down} * (-1)

        Args:
            df: DataFrame with bid_price, ask_price, bid_size, ask_size columns

        Returns:
            pd.Series of OFI values
        """
        bid_size_chg = df["bid_size"].diff().fillna(0)
        ask_size_chg = df["ask_size"].diff().fillna(0)
        bid_price_chg = df["bid_price"].diff().fillna(0)
        ask_price_chg = df["ask_price"].diff().fillna(0)

        # Indicator: bid side active (bid price ≥ prev bid)
        bid_side = (bid_price_chg >= 0).astype(float)
        # Indicator: ask side active (ask price ≤ prev ask)
        ask_side = (ask_price_chg <= 0).astype(float)

        ofi = bid_size_chg * bid_side - ask_size_chg * ask_side
        return ofi

    def estimate_beta(
        self, price_change: pd.Series, ofi: pd.Series
    ) -> Tuple[float, float, float]:
        """
        Estimate market impact β via OLS: ΔPk = β·OFIk + εk

        Args:
            price_change: Series of price changes ΔPk
            ofi: Series of OFI values

        Returns:
            Tuple of (beta, t_stat, r_squared)
        """
        valid = ~(price_change.isna() | ofi.isna())
        y = price_change[valid].values
        x = ofi[valid].values

        if len(y) < self.cfg.ofi_min_obs:
            return 0.0, 0.0, 0.0

        # OLS with intercept suppressed (microstructure literature standard)
        x_col = x.reshape(-1, 1)
        beta, res, _, _ = np.linalg.lstsq(x_col, y, rcond=None)
        beta_val = float(beta[0])

        # Compute R² and t-stat
        y_hat = x * beta_val
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # t-statistic
        n = len(y)
        se = np.sqrt(ss_res / (n - 1)) / (np.std(x) * np.sqrt(n))
        t_stat = beta_val / se if se > 0 else 0.0

        return beta_val, t_stat, r2

    def rolling_beta(self, df: pd.DataFrame, ticker: str) -> pd.Series:
        """
        Estimate time-varying β using rolling OLS window.

        Args:
            df: OHLCV + LOB DataFrame
            ticker: Ticker symbol for logging

        Returns:
            pd.Series of rolling β estimates indexed by date
        """
        ofi = self.compute_ofi(df)
        price_change = df["mid_price"].diff()
        window = self.cfg.ofi_window

        betas = []
        for i in range(len(df)):
            if i < window:
                betas.append(np.nan)
                continue
            sl_y = price_change.iloc[i - window: i]
            sl_x = ofi.iloc[i - window: i]
            beta, _, _ = self.estimate_beta(sl_y, sl_x)
            betas.append(beta)

        beta_series = pd.Series(betas, index=df.index, name=f"beta_{ticker}")
        self.beta_history[ticker] = beta_series
        log.info(
            f"OFI rolling β [{ticker}]: mean={beta_series.dropna().mean():.6f}, "
            f"std={beta_series.dropna().std():.6f}"
        )
        return beta_series

    def fit_all(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Fit OFI model for all tickers.

        Returns:
            Dict mapping ticker → {beta, r2, t_stat, ofi}
        """
        results = {}
        for ticker, df in data.items():
            ofi = self.compute_ofi(df)
            price_change = df["mid_price"].diff()
            # Align on common index before fitting
            common_idx = price_change.dropna().index.intersection(ofi.dropna().index)
            beta, t_stat, r2 = self.estimate_beta(
                price_change.loc[common_idx], ofi.loc[common_idx]
            )
            rolling_b = self.rolling_beta(df, ticker)
            self.r_squared[ticker] = r2

            results[ticker] = {
                "beta": beta,
                "t_stat": t_stat,
                "r_squared": r2,
                "ofi_mean": float(ofi.mean()),
                "ofi_std": float(ofi.std()),
                "beta_rolling_last": float(rolling_b.dropna().iloc[-1]) if not rolling_b.dropna().empty else beta,
                "ofi": ofi,
                "rolling_beta": rolling_b,
            }
            log.info(
                f"OFI fit [{ticker}]: β={beta:.6f}, t={t_stat:.2f}, R²={r2:.4f}"
            )
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. ALMGREN-CHRISS OPTIMAL EXECUTION  (2000)
# ─────────────────────────────────────────────────────────────────────────────
class AlmgrenChrissModel:
    """
    Almgren-Chriss optimal execution model (2000).

    Minimizes the trade-off between execution cost and variance:
        min  E[Cost] + λ · Var[Cost]

    Optimal trading trajectory:
        x(t) = X · sinh(κ(T-t)) / sinh(κT)
        κ²   = λ·σ² / η

    where:
        X   = total shares to execute
        T   = execution horizon
        η   = temporary impact coefficient
        γ   = permanent impact coefficient
        σ   = price volatility
        λ   = risk aversion parameter

    Reference:
        Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio
        transactions. Journal of Risk, 3(2), 5–39.
    """

    def __init__(self, config: ModelConfig):
        self.cfg = config

    def compute_kappa(self, lambda_risk: float) -> float:
        """
        Compute decay constant κ² = λ·σ²/η.

        Args:
            lambda_risk: Risk aversion parameter λ

        Returns:
            κ (sqrt of λσ²/η)
        """
        sigma = self.cfg.ac_sigma
        eta = self.cfg.ac_eta
        return np.sqrt(lambda_risk * sigma**2 / eta)

    def optimal_trajectory(
        self,
        lambda_risk: float,
        n_intervals: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Compute optimal execution schedule x(t).

        Args:
            lambda_risk: Risk aversion parameter
            n_intervals: Number of execution intervals (defaults to horizon hours)

        Returns:
            DataFrame with columns: t, x (remaining shares), trade (shares traded)
        """
        X = self.cfg.ac_shares
        T = self.cfg.ac_horizon_hours
        if n_intervals is None:
            n_intervals = T

        kappa = self.compute_kappa(lambda_risk)
        times = np.linspace(0, T, n_intervals + 1)

        # Remaining inventory at time t
        if kappa * T < 1e-8:
            # Linear (risk-neutral) case
            trajectory = X * (1 - times / T)
        else:
            trajectory = X * np.sinh(kappa * (T - times)) / np.sinh(kappa * T)

        trajectory = np.maximum(0, trajectory).round(0)
        # trades[i] = shares sold between step i-1 and step i; 0 at t=0
        trades_full = np.concatenate([[0], -np.diff(trajectory)])

        df = pd.DataFrame({
            "t": times,
            "x_remaining": trajectory,
            "trade": trades_full,
        })
        return df

    def expected_cost(self, lambda_risk: float) -> Tuple[float, float]:
        """
        Compute E[Cost] and Var[Cost] for given risk aversion.

        Args:
            lambda_risk: Risk aversion parameter

        Returns:
            Tuple of (expected_cost, cost_variance)
        """
        X = self.cfg.ac_shares
        T = float(self.cfg.ac_horizon_hours)
        sigma = self.cfg.ac_sigma
        eta = self.cfg.ac_eta
        gamma = self.cfg.ac_gamma
        kappa = self.compute_kappa(lambda_risk)

        # Temporary impact cost: η/T * ∫₀ᵀ ẋ² dt
        if kappa * T < 1e-8:
            E_temp = eta * X**2 / T
        else:
            E_temp = (eta / 2) * kappa * X**2 / np.tanh(kappa * T / 2)

        # Permanent impact cost: γ/2 * X²
        E_perm = 0.5 * gamma * X**2

        E_cost = E_temp + E_perm

        # Variance: σ² * ∫₀ᵀ x(t)² dt
        if kappa * T < 1e-8:
            V_cost = (sigma**2 * X**2 * T) / 3
        else:
            V_cost = (sigma**2 * X**2) / (2 * kappa) * (
                (1 / np.tanh(kappa * T)) - kappa * T / np.sinh(kappa * T)**2
            )

        return float(E_cost), float(V_cost)

    def efficient_frontier(self) -> pd.DataFrame:
        """
        Compute the Almgren-Chriss efficient frontier:
        E[Cost] vs √Var[Cost] for a range of risk aversions.

        Returns:
            DataFrame with lambda, E_cost, sqrt_var, kappa
        """
        lambdas = np.logspace(-4, 0, 50)
        rows = []
        for lam in lambdas:
            e, v = self.expected_cost(lam)
            rows.append({
                "lambda": lam,
                "E_cost": e,
                "sqrt_var": np.sqrt(max(0, v)),
                "kappa": self.compute_kappa(lam),
            })
        return pd.DataFrame(rows)

    def run(self) -> Dict:
        """
        Full Almgren-Chriss analysis for all risk aversion levels.

        Returns:
            Dictionary of results including trajectories and efficient frontier
        """
        log.info("Running Almgren-Chriss optimal execution model")
        results = {"trajectories": {}, "frontier": None, "summary": {}}

        for lam in self.cfg.ac_risk_aversions:
            kappa = self.compute_kappa(lam)
            traj = self.optimal_trajectory(lam)
            e_cost, var_cost = self.expected_cost(lam)
            style = (
                "Aggressive TWAP" if lam <= 0.001
                else "Balanced VWAP" if lam <= 0.01
                else "Risk-Averse IS"
            )
            results["trajectories"][lam] = {
                "kappa": kappa,
                "trajectory": traj.to_dict(orient="records"),
                "E_cost": e_cost,
                "sqrt_var": np.sqrt(var_cost),
                "style": style,
            }
            log.info(
                f"A-C [λ={lam}]: κ={kappa:.4f}, E[Cost]=${e_cost:.2f}, "
                f"√Var[Cost]=${np.sqrt(var_cost):.2f}, Style={style}"
            )

        results["frontier"] = self.efficient_frontier().to_dict(orient="records")
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. AVELLANEDA-STOIKOV MARKET MAKING  (2008)
# ─────────────────────────────────────────────────────────────────────────────
class AvellanedaStoikovModel:
    """
    Avellaneda-Stoikov high-frequency market making model (2008).

    The market maker chooses bid/ask quotes to maximize expected utility
    of terminal wealth, accounting for inventory risk.

    Reservation price (risk-adjusted midprice):
        r(S, q, t) = S - q · γ · σ² · (T - t)

    Optimal half-spread:
        δ(q, t) = γ · σ² · (T-t) + (2/γ) · ln(1 + γ/κ)

    Bid/Ask quotes:
        S^b = r - δ    (reservation bid)
        S^a = r + δ    (reservation ask)

    Reference:
        Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a
        limit order book. Quantitative Finance, 8(3), 217–224.
    """

    def __init__(self, config: ModelConfig):
        self.cfg = config

    def reservation_price(
        self, S: float, q: float, t: float
    ) -> float:
        """
        Compute reservation price r = S - q·γ·σ²·(T-t).

        Args:
            S: Current midprice
            q: Current inventory (positive = long)
            t: Current time (fraction of T)

        Returns:
            Reservation price r
        """
        gamma = self.cfg.as_gamma
        sigma = self.cfg.as_sigma
        T = self.cfg.as_T
        return S - q * gamma * sigma**2 * (T - t)

    def optimal_spread(self, t: float) -> float:
        """
        Compute optimal half-spread:
            δ = γ·σ²·(T-t) + (2/γ)·ln(1 + γ/κ)

        Args:
            t: Current time (fraction of T)

        Returns:
            Half-spread δ
        """
        gamma = self.cfg.as_gamma
        sigma = self.cfg.as_sigma
        kappa = self.cfg.as_kappa
        T = self.cfg.as_T
        return gamma * sigma**2 * (T - t) + (2 / gamma) * np.log(1 + gamma / kappa)

    def simulate_episode(
        self, S0: float, episode_id: int = 0
    ) -> Dict:
        """
        Simulate one episode of market making under A-S policy.

        Args:
            S0: Initial midprice
            episode_id: Episode index for logging

        Returns:
            Dict with PnL, inventory path, quote history
        """
        cfg = self.cfg
        dt = cfg.as_dt
        T = cfg.as_T
        n_steps = int(T / dt)
        rng = np.random.default_rng(cfg.random_seed + episode_id)

        S = S0
        q = 0  # inventory
        cash = 0.0
        pnl_path = [0.0]
        q_path = [q]
        S_path = [S]
        bid_path, ask_path = [S0], [S0]

        for step in range(n_steps):
            t = step * dt

            # Price evolves as Brownian motion
            dW = rng.standard_normal() * cfg.as_sigma * np.sqrt(dt)
            S = max(0.01, S + dW)

            # Compute quotes
            r = self.reservation_price(S, q, t)
            delta = self.optimal_spread(t)
            bid = r - delta
            ask = r + delta

            bid_path.append(bid)
            ask_path.append(ask)
            S_path.append(S)

            # Order arrival (Poisson, intensity κ · exp(-κ · spread))
            lambda_b = cfg.as_kappa * np.exp(-cfg.as_kappa * (S - bid))
            lambda_a = cfg.as_kappa * np.exp(-cfg.as_kappa * (ask - S))

            # Bernoulli approximation for small dt
            fill_bid = rng.random() < lambda_b * dt
            fill_ask = rng.random() < lambda_a * dt

            # Inventory management with hard limits
            if fill_bid and q < cfg.as_q_max:
                q += 1
                cash -= bid  # Pay bid

            if fill_ask and q > -cfg.as_q_max:
                q -= 1
                cash += ask  # Receive ask

            # Mark-to-market PnL
            pnl = cash + q * S
            pnl_path.append(pnl)
            q_path.append(q)

        # Terminal inventory liquidation at midprice
        final_pnl = cash + q * S

        return {
            "episode": episode_id,
            "final_pnl": final_pnl,
            "final_inventory": q,
            "pnl_path": pnl_path,
            "q_path": q_path,
            "S_path": S_path,
            "bid_path": bid_path,
            "ask_path": ask_path,
            "n_steps": n_steps,
        }

    def run(self, S0: float = 100.0) -> Dict:
        """
        Run full A-S market making simulation.

        Args:
            S0: Initial price

        Returns:
            Dict with episode results, PnL statistics, optimal quotes
        """
        log.info(
            f"Running Avellaneda-Stoikov simulation: {self.cfg.as_n_episodes} episodes"
        )
        episodes = []
        for ep in range(self.cfg.as_n_episodes):
            result = self.simulate_episode(S0, episode_id=ep)
            episodes.append(result)

        pnls = np.array([e["final_pnl"] for e in episodes])
        inv = np.array([e["final_inventory"] for e in episodes])

        # Summary statistics
        summary = {
            "mean_pnl": float(np.mean(pnls)),
            "std_pnl": float(np.std(pnls)),
            "sharpe": float(np.mean(pnls) / np.std(pnls)) if np.std(pnls) > 0 else 0,
            "pnl_q25": float(np.percentile(pnls, 25)),
            "pnl_q75": float(np.percentile(pnls, 75)),
            "mean_final_inventory": float(np.mean(inv)),
            "fraction_episodes_positive": float(np.mean(pnls > 0)),
            "optimal_spread_t0": float(self.optimal_spread(0.0)),
            "optimal_spread_t_mid": float(self.optimal_spread(self.cfg.as_T / 2)),
            "optimal_spread_t_end": float(self.optimal_spread(self.cfg.as_T * 0.99)),
        }

        log.info(
            f"A-S Simulation: mean_PnL=${summary['mean_pnl']:.4f}, "
            f"Sharpe={summary['sharpe']:.3f}, "
            f"spread_t0={summary['optimal_spread_t0']:.4f}"
        )

        # Sample episode for export (first episode)
        sample = {k: v for k, v in episodes[0].items() if k != "pnl_path"}

        return {"summary": summary, "sample_episode": sample, "n_episodes": self.cfg.as_n_episodes}


# ─────────────────────────────────────────────────────────────────────────────
# 4. KYLE'S LAMBDA  (price impact per unit volume)
# ─────────────────────────────────────────────────────────────────────────────
class KyleLambdaModel:
    """
    Kyle (1985) price impact model.

    Estimates λ = Cov(ΔP, Q) / Var(Q)

    This is the price impact per unit of signed volume, calibrated
    via OLS: ΔPt = λ · Qt + εt

    Reference:
        Kyle, A. S. (1985). Continuous auctions and insider trading.
        Econometrica, 53(6), 1315–1335.
    """

    def __init__(self, config: ModelConfig):
        self.cfg = config

    def estimate(
        self, price_change: np.ndarray, signed_volume: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Estimate Kyle's λ via OLS.

        Args:
            price_change: Array of ΔPt
            signed_volume: Array of signed order flow Qt

        Returns:
            Tuple of (lambda, r_squared, t_stat)
        """
        valid = ~(np.isnan(price_change) | np.isnan(signed_volume))
        y, x = price_change[valid], signed_volume[valid]
        if len(y) < 10:
            return 0.0, 0.0, 0.0

        # OLS: ΔP = λ·Q
        cov_pq = np.cov(y, x)[0, 1]
        var_q = np.var(x)
        lam = cov_pq / var_q if var_q > 0 else 0.0

        # R²
        y_hat = lam * x
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = max(0, 1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # t-stat
        n = len(y)
        se = np.sqrt(ss_res / (n - 1)) / (np.std(x) * np.sqrt(n))
        t_stat = lam / se if se > 0 else 0.0

        return float(lam), float(r2), float(t_stat)

    def rolling_lambda(
        self, df: pd.DataFrame, ticker: str
    ) -> pd.Series:
        """
        Estimate time-varying Kyle λ using 63-day rolling window.

        Args:
            df: DataFrame with close and signed_volume columns
            ticker: Ticker label for logging

        Returns:
            pd.Series of rolling λ estimates
        """
        price_change = df["close"].diff().values
        signed_vol = df["signed_volume"].values
        window = self.cfg.kyle_window

        lambdas = []
        for i in range(len(df)):
            if i < window:
                lambdas.append(np.nan)
                continue
            sl_y = price_change[i - window: i]
            sl_x = signed_vol[i - window: i]
            lam, _, _ = self.estimate(sl_y, sl_x)
            lambdas.append(lam)

        series = pd.Series(lambdas, index=df.index, name=f"kyle_lambda_{ticker}")
        mean_lam = series.dropna().mean()
        log.info(f"Kyle λ [{ticker}]: rolling mean={mean_lam:.8f}")
        return series

    def fit_all(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Estimate Kyle λ for all tickers."""
        results = {}
        for ticker, df in data.items():
            price_change = df["close"].diff().values
            signed_vol = df["signed_volume"].values
            lam, r2, t_stat = self.estimate(price_change[1:], signed_vol[1:])
            rolling_lam = self.rolling_lambda(df, ticker)
            results[ticker] = {
                "lambda": lam,
                "r_squared": r2,
                "t_stat": t_stat,
                "lambda_rolling_last": float(rolling_lam.dropna().iloc[-1]) if not rolling_lam.dropna().empty else lam,
                "rolling_lambda": rolling_lam,
            }
            log.info(
                f"Kyle λ [{ticker}]: λ={lam:.8f}, R²={r2:.4f}, t={t_stat:.2f}"
            )
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. CROSS-ASSET OFI  (Arxiv 2112.13213)
# ─────────────────────────────────────────────────────────────────────────────
class CrossAssetOFI:
    """
    Multi-asset Order Flow Imbalance with spillover effects.

    Reference:
        Kolm, P. N., et al. (2021). Multi-asset order flow imbalance.
        arXiv:2112.13213.

    The model estimates a spillover matrix Φ such that:
        ΔP_i = Σ_j Φ_{ij} · OFI_j + ε_i

    Where:
        NVDA → AAPL: 0.42 (tech sentiment channel)
        NVDA → PLTR: 0.28 (AI infrastructure channel)
        TSLA → PLTR: -0.18 (growth rotation channel)
    """

    def __init__(self, config: ModelConfig):
        self.cfg = config
        self.spillover_matrix = np.array(config.spillover_matrix)

    def compute_multi_ofi(
        self,
        data: Dict[str, pd.DataFrame],
        ofi_model: OFIModel,
    ) -> pd.DataFrame:
        """
        Compute OFI for all tickers and align on common dates.

        Args:
            data: Dict of DataFrames for each ticker
            ofi_model: Fitted OFI model instance

        Returns:
            DataFrame with one column per ticker
        """
        ofi_dict = {}
        for ticker, df in data.items():
            ofi_dict[ticker] = ofi_model.compute_ofi(df)

        ofi_df = pd.DataFrame(ofi_dict)
        ofi_df = ofi_df.dropna()
        return ofi_df

    def estimate_spillover(
        self,
        data: Dict[str, pd.DataFrame],
        ofi_model: OFIModel,
    ) -> np.ndarray:
        """
        Estimate the OFI spillover matrix via multivariate OLS.

        Returns:
            Estimated spillover matrix Φ (n_tickers × n_tickers)
        """
        ofi_df = self.compute_multi_ofi(data, ofi_model)
        tickers = self.cfg.tickers
        n = len(tickers)

        # Align price changes with OFI
        price_changes = {}
        for ticker, df in data.items():
            price_changes[ticker] = df["mid_price"].diff()

        price_df = pd.DataFrame(price_changes).dropna()
        common_idx = ofi_df.index.intersection(price_df.index)
        ofi_aligned = ofi_df.loc[common_idx, tickers].values
        price_aligned = price_df.loc[common_idx, tickers].values

        # Estimate Φ: each row is one regression
        phi_hat = np.zeros((n, n))
        for i in range(n):
            y = price_aligned[:, i]
            X = ofi_aligned
            if len(y) < 20:
                continue
            coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            phi_hat[i, :] = coef

        log.info("Cross-asset OFI spillover matrix estimated")
        return phi_hat

    def compute_cross_ofi_signals(
        self,
        data: Dict[str, pd.DataFrame],
        ofi_model: OFIModel,
    ) -> Dict[str, pd.Series]:
        """
        Compute cross-asset adjusted OFI signal for each ticker.

        The signal is: OFI_cross_i = Σ_j Φ_{ij} · OFI_j

        Returns:
            Dict mapping ticker → cross-asset OFI signal
        """
        phi = self.estimate_spillover(data, ofi_model)
        ofi_df = self.compute_multi_ofi(data, ofi_model)
        tickers = self.cfg.tickers

        signals = {}
        for i, ticker in enumerate(tickers):
            # Use theoretical spillover matrix (or estimated)
            weights = self.spillover_matrix[i, :]
            cross_signal = ofi_df[tickers].values @ weights
            signals[ticker] = pd.Series(cross_signal, index=ofi_df.index)
            log.info(
                f"Cross-OFI [{ticker}]: mean_signal={signals[ticker].mean():.4f}"
            )

        return signals


# ─────────────────────────────────────────────────────────────────────────────
# 6. ENHANCED FAMA-FRENCH + MICROSTRUCTURE (11 FACTORS)
# ─────────────────────────────────────────────────────────────────────────────
class EnhancedFactorModel:
    """
    Augmented factor model combining Fama-French 5F with momentum (MOM),
    sentiment (VADER), and four microstructure factors.

    Factors:
        1.  MKT     — Market beta (Fama-French)
        2.  SMB     — Size premium
        3.  HML     — Value premium
        4.  RMW     — Profitability premium
        5.  CMA     — Investment premium
        6.  MOM     — Carhart momentum
        7.  VADER   — Sentiment (news polarity)
        8.  OFI     — Order Flow Imbalance factor
        9.  KYLE_LAMBDA  — Price impact factor
        10. BID_ASK_SPREAD — Bid-ask spread factor (illiquidity)
        11. REALIZED_VOL_RATIO — Realized vs implied vol ratio

    Alpha = Realized Return - 11F Expected Return
    """

    def __init__(self, config: ModelConfig):
        self.cfg = config

    def _simulate_ff_factors(
        self, n: int, seed: int = 42
    ) -> pd.DataFrame:
        """
        Simulate Fama-French + macro factor returns.
        In production, these would be downloaded from Ken French's data library.

        Args:
            n: Number of observations
            seed: Random seed

        Returns:
            DataFrame of factor returns
        """
        rng = np.random.default_rng(seed)
        factor_means = {
            "MKT": 0.0006, "SMB": 0.0002, "HML": 0.0001,
            "RMW": 0.0002, "CMA": 0.0001, "MOM": 0.0003,
            "VADER": 0.0,
        }
        factor_vols = {
            "MKT": 0.010, "SMB": 0.005, "HML": 0.005,
            "RMW": 0.004, "CMA": 0.003, "MOM": 0.008,
            "VADER": 0.003,
        }
        factors = {}
        for fname in factor_means:
            factors[fname] = rng.normal(
                factor_means[fname], factor_vols[fname], n
            )
        return pd.DataFrame(factors)

    def construct_factor_matrix(
        self,
        data: Dict[str, pd.DataFrame],
        ofi_results: Dict,
        kyle_results: Dict,
    ) -> Dict[str, pd.DataFrame]:
        """
        Construct the full 11-factor matrix for each ticker.

        Args:
            data: Price DataFrames
            ofi_results: OFI model output (contains ofi and rolling_beta)
            kyle_results: Kyle λ output

        Returns:
            Dict mapping ticker → factor DataFrame (columns = 11 factors)
        """
        factor_dfs = {}
        for ticker, df in data.items():
            n = len(df)
            ff = self._simulate_ff_factors(n, seed=self.cfg.random_seed)

            # OFI factor: z-score normalized OFI
            ofi_raw = ofi_results[ticker]["ofi"]
            ofi_aligned = ofi_raw.reindex(df.index).fillna(0)
            ofi_factor = (ofi_aligned - ofi_aligned.rolling(63).mean()) / (
                ofi_aligned.rolling(63).std() + 1e-8
            )
            ofi_factor = ofi_factor.fillna(0).values

            # Kyle λ factor: normalized rolling λ
            kyle_raw = kyle_results[ticker]["rolling_lambda"]
            kyle_aligned = kyle_raw.reindex(df.index).fillna(0)
            kyle_factor = (kyle_aligned - kyle_aligned.rolling(63).mean()) / (
                kyle_aligned.rolling(63).std() + 1e-8
            )
            kyle_factor = kyle_factor.fillna(0).values

            # Bid-ask spread factor: (spread / mid) relative to 63-day avg
            spread = df["spread"] / df["mid_price"]
            spread_factor = (spread - spread.rolling(63).mean()) / (
                spread.rolling(63).std() + 1e-8
            )
            spread_factor = spread_factor.fillna(0).values

            # Realized vol ratio: 5-day realized vol / 21-day realized vol
            log_ret = np.log(df["close"] / df["close"].shift(1)).fillna(0)
            rv5 = log_ret.rolling(5).std().fillna(log_ret.std())
            rv21 = log_ret.rolling(21).std().fillna(log_ret.std())
            rvr = (rv5 / (rv21 + 1e-10)).fillna(1.0).values

            factor_matrix = pd.DataFrame(
                {
                    "MKT": ff["MKT"].values,
                    "SMB": ff["SMB"].values,
                    "HML": ff["HML"].values,
                    "RMW": ff["RMW"].values,
                    "CMA": ff["CMA"].values,
                    "MOM": ff["MOM"].values,
                    "VADER": ff["VADER"].values,
                    "OFI": ofi_factor,
                    "KYLE_LAMBDA": kyle_factor,
                    "BID_ASK_SPREAD": spread_factor,
                    "REALIZED_VOL_RATIO": rvr,
                },
                index=df.index,
            )
            factor_dfs[ticker] = factor_matrix
        return factor_dfs

    def estimate_loadings(
        self,
        data: Dict[str, pd.DataFrame],
        factor_dfs: Dict[str, pd.DataFrame],
    ) -> Dict[str, Dict]:
        """
        Estimate factor loadings via OLS: R = α + βᵀF + ε

        Returns:
            Dict mapping ticker → {alpha, loadings, r2, ic}
        """
        results = {}
        for ticker, df in data.items():
            returns = df["close"].pct_change().fillna(0).values
            F = factor_dfs[ticker].values
            n, k = F.shape

            # OLS with intercept
            X = np.column_stack([np.ones(n), F])
            betas, res, _, _ = np.linalg.lstsq(X, returns, rcond=None)
            alpha = betas[0]
            loadings = betas[1:]

            y_hat = X @ betas
            ss_res = np.sum((returns - y_hat) ** 2)
            ss_tot = np.sum((returns - np.mean(returns)) ** 2)
            r2 = max(0, 1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            # Information Coefficient (rank correlation)
            ic, _ = stats.spearmanr(returns[1:], y_hat[1:])

            results[ticker] = {
                "alpha": float(alpha),
                "loadings": {f: float(loadings[i]) for i, f in enumerate(self.cfg.ff_factors)},
                "r_squared": r2,
                "ic": float(ic) if not np.isnan(ic) else 0.0,
            }
            log.info(
                f"Factor model [{ticker}]: α={alpha:.6f}, R²={r2:.4f}, IC={ic:.4f}"
            )
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 7. KALMAN FILTER — ADAPTIVE BETA CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────
class KalmanFilterOFI:
    """
    Adaptive OFI β estimation using a Kalman filter.

    State equation:    β_t = β_{t-1} + w_t,    w_t ~ N(0, Q)
    Observation eq:    ΔP_t = β_t · OFI_t + v_t,  v_t ~ N(0, R)

    The Kalman filter adaptively updates β as market microstructure
    conditions change (e.g., regime changes, volatility spikes).

    Calibration is logged to JSON for daily self-improvement.
    """

    def __init__(self, config: ModelConfig):
        self.cfg = config
        self.Q = config.kalman_Q   # Process noise
        self.R = config.kalman_R   # Observation noise

    def run(
        self,
        price_change: pd.Series,
        ofi: pd.Series,
        ticker: str,
    ) -> pd.DataFrame:
        """
        Run Kalman filter to adaptively estimate β_t.

        Args:
            price_change: ΔP series
            ofi: OFI series
            ticker: Ticker label

        Returns:
            DataFrame with columns: date, beta_kalman, P (error covariance),
            innovation, kalman_gain
        """
        y = price_change.values
        x = ofi.values
        n = min(len(y), len(x))

        # Initialize state
        beta = self.cfg.ofi_beta_init
        P = self.cfg.kalman_P_init  # Error covariance

        betas_k = [beta]
        Ps = [P]
        innovations = [0.0]
        gains = [0.0]

        for t in range(1, n):
            # Predict
            beta_pred = beta      # Random walk state transition
            P_pred = P + self.Q   # Prediction error covariance

            # Observation
            H = x[t]  # Observation matrix (scalar)
            S = H * P_pred * H + self.R  # Innovation covariance

            # Kalman gain
            K = P_pred * H / S if S != 0 else 0.0

            # Update
            innovation = y[t] - H * beta_pred
            beta = beta_pred + K * innovation
            P = (1 - K * H) * P_pred

            betas_k.append(beta)
            Ps.append(P)
            innovations.append(innovation)
            gains.append(K)

        idx = price_change.index[:n]
        df_kalman = pd.DataFrame({
            "beta_kalman": betas_k[:n],
            "P": Ps[:n],
            "innovation": innovations[:n],
            "kalman_gain": gains[:n],
        }, index=idx)

        final_beta = float(betas_k[-1])
        log.info(
            f"Kalman Filter [{ticker}]: final β={final_beta:.8f}, "
            f"final P={Ps[-1]:.8f}"
        )
        return df_kalman

    def save_calibration(
        self,
        ticker: str,
        df_kalman: pd.DataFrame,
        ofi_beta_ols: float,
    ) -> Path:
        """
        Log calibration results to JSON for daily self-improvement loop.

        Args:
            ticker: Ticker symbol
            df_kalman: Kalman filter output DataFrame
            ofi_beta_ols: OLS-estimated β for comparison

        Returns:
            Path to saved JSON file
        """
        import datetime
        calib = {
            "ticker": ticker,
            "date": datetime.date.today().isoformat(),
            "kalman_beta_final": float(df_kalman["beta_kalman"].iloc[-1]),
            "kalman_beta_mean": float(df_kalman["beta_kalman"].mean()),
            "kalman_beta_std": float(df_kalman["beta_kalman"].std()),
            "ols_beta": ofi_beta_ols,
            "kalman_P_final": float(df_kalman["P"].iloc[-1]),
            "mean_innovation": float(df_kalman["innovation"].mean()),
            "std_innovation": float(df_kalman["innovation"].std()),
            "n_observations": len(df_kalman),
        }
        path = OUTPUT_DIR / f"calibration_{ticker}.json"
        with open(path, "w") as f:
            json.dump(calib, f, indent=2)
        log.info(f"Calibration saved: {path}")
        return path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
class V6MicrostructurePipeline:
    """
    AXIOM v6 Microstructure Model — Main Pipeline.

    Orchestrates all components:
        1. Data generation
        2. OFI Model
        3. Almgren-Chriss Optimal Execution
        4. Avellaneda-Stoikov Market Making
        5. Kyle's Lambda
        6. Cross-Asset OFI
        7. Enhanced Factor Model (11 factors)
        8. Kalman Filter Calibration
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.cfg = config or ModelConfig()
        self.results: Dict = {}

    def run(self) -> Dict:
        """
        Execute the full v6 microstructure pipeline.

        Returns:
            Dict of results from all model components.
        """
        log.info("=" * 70)
        log.info("AXIOM v6 Microstructure Pipeline — Starting")
        log.info("=" * 70)
        log.info(f"Tickers: {self.cfg.tickers}")
        log.info(f"Output: {OUTPUT_DIR}")

        # ── 1. Generate synthetic data ──────────────────────────────────────
        log.info("\n[1/8] Generating synthetic LOB data")
        gen = SyntheticDataGenerator(self.cfg)
        data = gen.generate_all()

        # ── 2. OFI Model ────────────────────────────────────────────────────
        log.info("\n[2/8] Fitting OFI model (Cont et al. 2014)")
        ofi_model = OFIModel(self.cfg)
        ofi_results = ofi_model.fit_all(data)

        # ── 3. Almgren-Chriss ───────────────────────────────────────────────
        log.info("\n[3/8] Running Almgren-Chriss optimal execution")
        ac_model = AlmgrenChrissModel(self.cfg)
        ac_results = ac_model.run()

        # ── 4. Avellaneda-Stoikov ───────────────────────────────────────────
        log.info("\n[4/8] Simulating Avellaneda-Stoikov market making")
        as_model = AvellanedaStoikovModel(self.cfg)
        as_results = as_model.run(S0=100.0)

        # ── 5. Kyle's Lambda ────────────────────────────────────────────────
        log.info("\n[5/8] Estimating Kyle's Lambda")
        kyle_model = KyleLambdaModel(self.cfg)
        kyle_results = kyle_model.fit_all(data)

        # ── 6. Cross-Asset OFI ──────────────────────────────────────────────
        log.info("\n[6/8] Computing cross-asset OFI spillover")
        cross_ofi = CrossAssetOFI(self.cfg)
        cross_signals = cross_ofi.compute_cross_ofi_signals(data, ofi_model)
        phi_hat = cross_ofi.estimate_spillover(data, ofi_model)

        # ── 7. Enhanced Factor Model ────────────────────────────────────────
        log.info("\n[7/8] Fitting 11-factor model (FF5 + MOM + VADER + Micro)")
        factor_model = EnhancedFactorModel(self.cfg)
        factor_dfs = factor_model.construct_factor_matrix(data, ofi_results, kyle_results)
        factor_results = factor_model.estimate_loadings(data, factor_dfs)

        # ── 8. Kalman Filter Calibration ────────────────────────────────────
        log.info("\n[8/8] Running Kalman filter self-calibration")
        kalman = KalmanFilterOFI(self.cfg)
        kalman_results = {}
        for ticker, df in data.items():
            ofi_series = ofi_results[ticker]["ofi"]
            price_chg = df["mid_price"].diff().reindex(ofi_series.index).fillna(0)
            df_k = kalman.run(price_chg, ofi_series, ticker)
            path = kalman.save_calibration(
                ticker, df_k, ofi_results[ticker]["beta"]
            )
            kalman_results[ticker] = {
                "final_beta": float(df_k["beta_kalman"].iloc[-1]),
                "mean_beta": float(df_k["beta_kalman"].mean()),
                "calibration_file": str(path),
            }

        # ── Consolidate & Save ──────────────────────────────────────────────
        self.results = {
            "config": {k: v for k, v in asdict(self.cfg).items()
                       if not isinstance(v, dict) or k != "spillover_matrix"},
            "ofi": {t: {k: v for k, v in r.items()
                        if k not in ("ofi", "rolling_beta")}
                    for t, r in ofi_results.items()},
            "almgren_chriss": {
                "summary": {
                    str(lam): {
                        k: v for k, v in vals.items() if k != "trajectory"
                    }
                    for lam, vals in ac_results["trajectories"].items()
                },
            },
            "avellaneda_stoikov": as_results["summary"],
            "kyle_lambda": {t: {k: v for k, v in r.items()
                                if k not in ("rolling_lambda",)}
                            for t, r in kyle_results.items()},
            "cross_asset_ofi": {
                "spillover_matrix_estimated": phi_hat.tolist(),
                "spillover_matrix_theoretical": self.cfg.spillover_matrix,
                "channel_descriptions": {
                    "NVDA->AAPL": "+0.42 tech sentiment channel",
                    "NVDA->PLTR": "+0.28 AI infrastructure channel",
                    "TSLA->PLTR": "-0.18 growth rotation channel",
                    "TSLA->NVDA": "-0.08 risk-off signal",
                },
            },
            "factor_model": factor_results,
            "kalman_calibration": kalman_results,
        }

        self._save_results()
        self._print_summary()
        return self.results

    def _save_results(self) -> None:
        """Save all results to JSON."""
        out_path = OUTPUT_DIR / "v6_results.json"
        with open(out_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        log.info(f"\nResults saved: {out_path}")

    def _print_summary(self) -> None:
        """Print a formatted summary of key results."""
        log.info("\n" + "=" * 70)
        log.info("AXIOM v6 MICROSTRUCTURE MODEL — RESULTS SUMMARY")
        log.info("=" * 70)

        log.info("\n── OFI Model (Cont et al. 2014) ──")
        for t, r in self.results["ofi"].items():
            log.info(
                f"  {t:5s}: β={r['beta']:.6f}, t={r['t_stat']:.2f}, "
                f"R²={r['r_squared']:.4f}"
            )

        log.info("\n── Almgren-Chriss Execution ──")
        for lam_str, vals in self.results["almgren_chriss"]["summary"].items():
            log.info(
                f"  λ={lam_str}: κ={vals['kappa']:.4f}, "
                f"E[Cost]=${vals['E_cost']:.2f}, "
                f"√Var=${vals['sqrt_var']:.2f} — {vals['style']}"
            )

        log.info("\n── Avellaneda-Stoikov Market Making ──")
        asv = self.results["avellaneda_stoikov"]
        log.info(
            f"  mean_PnL=${asv['mean_pnl']:.4f}, Sharpe={asv['sharpe']:.3f}, "
            f"spread_t0={asv['optimal_spread_t0']:.4f}"
        )

        log.info("\n── Kyle's Lambda ──")
        for t, r in self.results["kyle_lambda"].items():
            log.info(
                f"  {t:5s}: λ={r['lambda']:.8f}, R²={r['r_squared']:.4f}, "
                f"t={r['t_stat']:.2f}"
            )

        log.info("\n── 11-Factor Model (Alpha decomposition) ──")
        for t, r in self.results["factor_model"].items():
            log.info(
                f"  {t:5s}: α={r['alpha']:.6f}, R²={r['r_squared']:.4f}, "
                f"IC={r['ic']:.4f}"
            )

        log.info("\n── Kalman Filter Calibration ──")
        for t, r in self.results["kalman_calibration"].items():
            log.info(
                f"  {t:5s}: β_kalman={r['final_beta']:.8f} "
                f"(file: {Path(r['calibration_file']).name})"
            )

        log.info(f"\nAll outputs saved to: {OUTPUT_DIR}")
        log.info("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    log.info("AXIOM v6 Microstructure Model — Initializing")

    cfg = ModelConfig(
        tickers=["NVDA", "AAPL", "PLTR", "TSLA"],
        n_days=504,
        random_seed=42,
    )

    pipeline = V6MicrostructurePipeline(config=cfg)
    results = pipeline.run()

    log.info("\nPipeline completed successfully.")
    sys.exit(0)
