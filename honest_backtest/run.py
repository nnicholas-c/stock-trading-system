"""Run a leakage-aware, broad-universe signal backtest.

This module is intentionally plain. It uses one fixed model specification and
reports net-of-cost out-of-sample metrics instead of searching for a flattering
configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"

FALLBACK_SP500_SAMPLE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "BRK-B", "LLY", "AVGO",
    "JPM", "V", "XOM", "UNH", "MA", "COST", "HD", "PG", "NFLX", "JNJ",
    "ABBV", "BAC", "CRM", "ORCL", "WMT", "KO", "MRK", "CVX", "CSCO", "AMD",
    "ACN", "MCD", "LIN", "IBM", "GE", "ADBE", "QCOM", "TXN", "INTU", "AMAT",
    "DIS", "PEP", "TMO", "DHR", "CAT", "NOW", "VZ", "ISRG", "AMGN", "CMCSA",
    "NKE", "PM", "GS", "RTX", "HON", "LOW", "SPGI", "BKNG", "COP", "UNP",
    "PFE", "MS", "BA", "LRCX", "SBUX", "DE", "ADP", "ELV", "MDLZ", "GILD",
    "TJX", "ADI", "SYK", "BLK", "C", "VRTX", "REGN", "PANW", "MU", "UBER",
]

FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "vol_10d",
    "vol_20d",
    "ma_10_dist",
    "ma_20_dist",
    "ma_50_dist",
    "rsi_14",
    "volume_z_20",
    "intraday_range",
]


@dataclass(frozen=True)
class BacktestConfig:
    start: str = "2018-01-01"
    end: str | None = None
    max_tickers: int = 60
    min_tickers: int = 50
    min_train_days: int = 504
    validation_days: int = 63
    step_days: int = 63
    purge_days: int = 5
    embargo_days: int = 5
    long_threshold: float = 0.55
    short_threshold: float = 0.45
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 5.0
    random_seed: int = 42


def load_sp500_universe(limit: int) -> list[str]:
    """Load current S&P 500 symbols from Wikipedia, with a static fallback."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        if len(tickers) >= limit:
            return tickers[:limit]
    except Exception:
        pass
    return FALLBACK_SP500_SAMPLE[:limit]


def _extract_ticker_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        if ticker in raw.columns.get_level_values(0):
            frame = raw[ticker].copy()
        else:
            frame = raw.xs(ticker, axis=1, level=1).copy()
    else:
        frame = raw.copy()
    return frame.rename(columns={col: str(col).title() for col in frame.columns})


def download_prices(tickers: list[str], config: BacktestConfig) -> dict[str, pd.DataFrame]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required; install requirements.txt or run in the project .venv") from exc

    raw = yf.download(
        tickers,
        start=config.start,
        end=config.end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    prices: dict[str, pd.DataFrame] = {}
    required = {"Open", "High", "Low", "Close", "Volume"}
    for ticker in tickers:
        try:
            frame = _extract_ticker_frame(raw, ticker)
        except Exception:
            continue
        if not required.issubset(frame.columns):
            continue
        frame = frame[list(required)].copy()
        frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
        if len(frame) >= config.min_train_days + config.validation_days + 30:
            prices[ticker] = frame
    if len(prices) < config.min_tickers:
        raise RuntimeError(f"Only {len(prices)} usable tickers downloaded; need at least {config.min_tickers}.")
    return prices


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_panel(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for ticker, frame in prices.items():
        df = frame.sort_index().copy()
        close = df["Close"]
        ret_1d = close.pct_change()
        df["ret_1d"] = ret_1d
        df["ret_5d"] = close.pct_change(5)
        df["ret_10d"] = close.pct_change(10)
        df["ret_20d"] = close.pct_change(20)
        df["vol_10d"] = ret_1d.rolling(10, min_periods=10).std()
        df["vol_20d"] = ret_1d.rolling(20, min_periods=20).std()
        for window in (10, 20, 50):
            df[f"ma_{window}_dist"] = close / close.rolling(window, min_periods=window).mean() - 1.0
        df["rsi_14"] = compute_rsi(close)
        vol_mean = df["Volume"].rolling(20, min_periods=20).mean()
        vol_std = df["Volume"].rolling(20, min_periods=20).std()
        df["volume_z_20"] = (df["Volume"] - vol_mean) / vol_std.replace(0, np.nan)
        df["intraday_range"] = (df["High"] - df["Low"]) / df["Close"]
        df["target_return"] = df["Close"].shift(-1) / df["Open"].shift(-1) - 1.0
        df["target_up"] = (df["target_return"] > 0).astype(int)
        df["execution_date"] = df.index.to_series().shift(-1)
        df["ticker"] = ticker
        df["signal_date"] = df.index
        rows.append(df.reset_index(drop=True))
    panel = pd.concat(rows, ignore_index=True)
    panel = panel.dropna(subset=FEATURE_COLUMNS + ["target_return", "execution_date"])
    panel = panel.sort_values(["signal_date", "ticker"]).reset_index(drop=True)
    return panel


def walkforward_predictions(panel: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    dates = pd.Index(sorted(panel["signal_date"].unique()))
    records = []
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.25, class_weight="balanced", max_iter=1000, random_state=config.random_seed)),
        ]
    )
    split_start = config.min_train_days
    while split_start + config.validation_days <= len(dates):
        val_dates = dates[split_start : split_start + config.validation_days]
        train_end = split_start - config.purge_days - config.embargo_days
        if train_end <= 0:
            split_start += config.step_days
            continue
        train_dates = dates[:train_end]
        train = panel[panel["signal_date"].isin(train_dates)]
        val = panel[panel["signal_date"].isin(val_dates)]
        if train["target_up"].nunique() < 2 or val.empty:
            split_start += config.step_days
            continue
        model.fit(train[FEATURE_COLUMNS], train["target_up"])
        probs = model.predict_proba(val[FEATURE_COLUMNS])[:, 1]
        fold = val[["signal_date", "execution_date", "ticker", "target_return", "target_up"]].copy()
        fold["prob_up"] = probs
        fold["score"] = fold["prob_up"] - 0.5
        fold["fold_train_start"] = train_dates[0]
        fold["fold_train_end"] = train_dates[-1]
        fold["fold_val_start"] = val_dates[0]
        fold["fold_val_end"] = val_dates[-1]
        records.append(fold)
        split_start += config.step_days
    if not records:
        raise RuntimeError("No walk-forward predictions were produced.")
    return pd.concat(records, ignore_index=True)


def apply_strategy(predictions: pd.DataFrame, config: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds = predictions.copy()
    preds["position"] = 0
    preds.loc[preds["prob_up"] >= config.long_threshold, "position"] = 1
    preds.loc[preds["prob_up"] <= config.short_threshold, "position"] = -1
    round_trip_cost = 2.0 * (config.transaction_cost_bps + config.slippage_bps) / 10_000.0
    preds["turnover"] = 2.0 * preds["position"].abs()
    preds["strategy_return_gross"] = preds["position"] * preds["target_return"]
    preds["strategy_return_net"] = preds["strategy_return_gross"] - preds["position"].abs() * round_trip_cost

    rng = np.random.default_rng(config.random_seed)
    active_rate = float(preds["position"].abs().mean())
    active = preds[preds["position"] != 0]
    long_share = float((active["position"] > 0).mean()) if not active.empty else 0.5
    probabilities = [active_rate * (1.0 - long_share), max(0.0, 1.0 - active_rate), active_rate * long_share]
    probabilities = np.array(probabilities) / np.sum(probabilities)
    random_pos = rng.choice(np.array([-1, 0, 1]), size=len(preds), p=probabilities)
    preds["random_position"] = random_pos
    preds["random_return_net"] = random_pos * preds["target_return"] - np.abs(random_pos) * round_trip_cost

    daily = preds.groupby("signal_date").agg(
        strategy_return_net=("strategy_return_net", "mean"),
        strategy_return_gross=("strategy_return_gross", "mean"),
        buy_hold_return=("target_return", "mean"),
        random_return_net=("random_return_net", "mean"),
        avg_turnover=("turnover", "mean"),
        active_fraction=("position", lambda x: float(np.mean(np.abs(x) > 0))),
        tickers=("ticker", "nunique"),
    )
    return preds, daily.reset_index()


def annualized_sharpe(daily_returns: pd.Series) -> float:
    sigma = daily_returns.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return 0.0
    return float(daily_returns.mean() / sigma * math.sqrt(252))


def max_drawdown(daily_returns: pd.Series) -> float:
    equity = (1.0 + daily_returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def deflated_sharpe_stats(daily_returns: pd.Series, sharpe: float, n_trials: int) -> dict[str, float]:
    n = len(daily_returns)
    if n < 3:
        return {"deflated_sharpe_threshold": 0.0, "deflated_sharpe": 0.0, "dsr_probability": 0.0}
    skew = float(stats.skew(daily_returns, bias=False, nan_policy="omit"))
    kurt = float(stats.kurtosis(daily_returns, fisher=False, bias=False, nan_policy="omit"))
    threshold = float(norm.ppf(1.0 - 1.0 / max(n_trials, 2)) * math.sqrt(252 / max(n - 1, 1)))
    denom = math.sqrt(max(1e-12, 1.0 - skew * sharpe + ((kurt - 1.0) / 4.0) * sharpe * sharpe))
    dsr_probability = float(norm.cdf((sharpe - threshold) * math.sqrt(max(n - 1, 1)) / denom))
    return {
        "deflated_sharpe_threshold": threshold,
        "deflated_sharpe": float(sharpe - threshold),
        "dsr_probability": dsr_probability,
    }


def estimate_configuration_trials(repo_root: Path) -> int:
    patterns = re.compile(
        r"(RandomForestClassifier\(|GradientBoostingClassifier\(|XGBClassifier\(|XGBRegressor\(|"
        r"LGBMClassifier\(|lgb\.train\(|LogisticRegression\(|PPO\(|A2C\(|MLPClassifier\(|"
        r"HistGradientBoostingClassifier\(|nn\.LSTM\(|LSTM\()"
    )
    files = list(repo_root.glob("experiments/*.py")) + list(repo_root.glob("train_*.py")) + [
        repo_root / "ml_trading_system.py",
        repo_root / "experiments" / "self_improve.py",
        repo_root / "research" / "pipeline.py",
    ]
    count = 0
    for path in files:
        if not path.exists():
            continue
        try:
            count += len(patterns.findall(path.read_text(errors="ignore")))
        except OSError:
            continue
    return max(count, 1)


def summarize(preds: pd.DataFrame, daily: pd.DataFrame, n_trials: int) -> pd.DataFrame:
    strategy = pd.Series(daily["strategy_return_net"].values)
    buy_hold = pd.Series(daily["buy_hold_return"].values)
    random = pd.Series(daily["random_return_net"].values)
    ic, _ = stats.spearmanr(preds["score"], preds["target_return"], nan_policy="omit")
    ic = float(ic) if not np.isnan(ic) else 0.0
    ic_n = int(preds[["score", "target_return"]].dropna().shape[0])
    ic_t = float(ic * math.sqrt(max(ic_n - 2, 1) / max(1e-12, 1 - ic * ic))) if ic_n > 2 else 0.0
    active = preds[preds["position"] != 0]
    hit_rate = float((active["position"] * active["target_return"] > 0).mean()) if not active.empty else 0.0
    sharpe = annualized_sharpe(strategy)
    dsr = deflated_sharpe_stats(strategy, sharpe, n_trials)

    rows = [
        {
            "strategy": "single_fixed_logistic_signal_net",
            "n_daily_obs": len(strategy),
            "n_prediction_rows": len(preds),
            "n_tickers_median": float(daily["tickers"].median()),
            "total_return": float((1 + strategy).prod() - 1),
            "annualized_sharpe": sharpe,
            "max_drawdown": max_drawdown(strategy),
            "information_coefficient": ic,
            "ic_t_stat": ic_t,
            "hit_rate_active_signals": hit_rate,
            "avg_daily_turnover": float(daily["avg_turnover"].mean()),
            "avg_active_fraction": float(daily["active_fraction"].mean()),
            "configuration_trials_lower_bound": n_trials,
            **dsr,
        },
        {
            "strategy": "equal_weight_buy_hold_baseline",
            "n_daily_obs": len(buy_hold),
            "n_prediction_rows": len(preds),
            "n_tickers_median": float(daily["tickers"].median()),
            "total_return": float((1 + buy_hold).prod() - 1),
            "annualized_sharpe": annualized_sharpe(buy_hold),
            "max_drawdown": max_drawdown(buy_hold),
            "information_coefficient": np.nan,
            "ic_t_stat": np.nan,
            "hit_rate_active_signals": np.nan,
            "avg_daily_turnover": 0.0,
            "avg_active_fraction": 1.0,
            "configuration_trials_lower_bound": n_trials,
            "deflated_sharpe_threshold": np.nan,
            "deflated_sharpe": np.nan,
            "dsr_probability": np.nan,
        },
        {
            "strategy": "zero_skill_random_baseline_net",
            "n_daily_obs": len(random),
            "n_prediction_rows": len(preds),
            "n_tickers_median": float(daily["tickers"].median()),
            "total_return": float((1 + random).prod() - 1),
            "annualized_sharpe": annualized_sharpe(random),
            "max_drawdown": max_drawdown(random),
            "information_coefficient": np.nan,
            "ic_t_stat": np.nan,
            "hit_rate_active_signals": np.nan,
            "avg_daily_turnover": float(daily["active_fraction"].mean() * 2),
            "avg_active_fraction": float(daily["active_fraction"].mean()),
            "configuration_trials_lower_bound": n_trials,
            "deflated_sharpe_threshold": np.nan,
            "deflated_sharpe": np.nan,
            "dsr_probability": np.nan,
        },
    ]
    return pd.DataFrame(rows)


def plot_results(daily: pd.DataFrame, preds: pd.DataFrame, output_dir: Path) -> None:
    equity = pd.DataFrame(
        {
            "strategy_net": (1 + daily["strategy_return_net"]).cumprod(),
            "buy_hold": (1 + daily["buy_hold_return"]).cumprod(),
            "random_net": (1 + daily["random_return_net"]).cumprod(),
        },
        index=pd.to_datetime(daily["signal_date"]),
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    equity.plot(ax=ax)
    ax.set_title("Honest Walk-Forward Equity Curves")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "equity_curves.png", dpi=160)
    plt.close(fig)

    sample = preds.sample(min(len(preds), 5000), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sample["score"], sample["target_return"], s=8, alpha=0.25)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Signal Score vs Next-Day Open-to-Close Return")
    ax.set_xlabel("Predicted probability minus 0.5")
    ax.set_ylabel("Realized return")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "score_vs_return.png", dpi=160)
    plt.close(fig)


def write_outputs(config: BacktestConfig, tickers: list[str], preds: pd.DataFrame, daily: pd.DataFrame, summary: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(RESULTS_DIR / "universe.csv", index=False)
    preds.to_csv(RESULTS_DIR / "fold_predictions.csv", index=False)
    daily.to_csv(RESULTS_DIR / "daily_returns.csv", index=False)
    summary.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)
    (RESULTS_DIR / "run_config.json").write_text(
        json.dumps({**asdict(config), "feature_columns": FEATURE_COLUMNS}, indent=2, default=str)
    )
    plot_results(daily, preds, RESULTS_DIR)


def run(config: BacktestConfig) -> pd.DataFrame:
    tickers = load_sp500_universe(config.max_tickers)
    prices = download_prices(tickers, config)
    usable_tickers = sorted(prices)
    panel = build_panel(prices)
    preds = walkforward_predictions(panel, config)
    preds, daily = apply_strategy(preds, config)
    n_trials = estimate_configuration_trials(REPO_ROOT)
    summary = summarize(preds, daily, n_trials)
    write_outputs(config, usable_tickers, preds, daily, summary)
    return summary


def parse_args(argv: Iterable[str] | None = None) -> BacktestConfig:
    parser = argparse.ArgumentParser(description="Run the honest AXIOM walk-forward backtest.")
    parser.add_argument("--start", default=BacktestConfig.start)
    parser.add_argument("--end", default=BacktestConfig.end)
    parser.add_argument("--max-tickers", type=int, default=BacktestConfig.max_tickers)
    parser.add_argument("--min-tickers", type=int, default=BacktestConfig.min_tickers)
    parser.add_argument("--transaction-cost-bps", type=float, default=BacktestConfig.transaction_cost_bps)
    parser.add_argument("--slippage-bps", type=float, default=BacktestConfig.slippage_bps)
    parser.add_argument("--long-threshold", type=float, default=BacktestConfig.long_threshold)
    parser.add_argument("--short-threshold", type=float, default=BacktestConfig.short_threshold)
    parser.add_argument("--random-seed", type=int, default=BacktestConfig.random_seed)
    args = parser.parse_args(argv)
    return BacktestConfig(
        start=args.start,
        end=args.end,
        max_tickers=args.max_tickers,
        min_tickers=args.min_tickers,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
        random_seed=args.random_seed,
    )


def main(argv: Iterable[str] | None = None) -> None:
    summary = run(parse_args(argv))
    print(summary.to_string(index=False))
    print(f"\nWrote results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
