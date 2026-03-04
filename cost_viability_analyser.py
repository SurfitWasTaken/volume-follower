from __future__ import annotations
"""
Cost viability pre-flight analyser.

Runs BEFORE any backtest to determine which timeframe/instrument
combinations are viable given OANDA's spread cost model.  Non-viable
combinations are excluded unless force_run is set.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG

logger = logging.getLogger(__name__)


def compute_cost_viability(
    instruments: list[str],
    timeframes: list[str],
    data_loader,
    output_dir: str | None = None,
) -> pd.DataFrame:
    """
    For each instrument × timeframe, compute:
      - typical_atr (median ATR from a 200-candle sample)
      - round_trip_cost_atr
      - break_even_win_rate
      - viable (bool)

    Saves the table to output/preflight/cost_viability.md.

    Parameters
    ----------
    instruments : list[str]
    timeframes : list[str]
    data_loader : DataLoader instance (used to fetch small samples)
    output_dir : str or None

    Returns
    -------
    pd.DataFrame with columns:
        instrument, timeframe, typical_atr, round_trip_cost_atr,
        break_even_win_rate, viable
    """
    rr_ratio = CONFIG.get("min_move_atr", 1.5)
    widen = CONFIG.get("spread_widening_factor", 1.5)
    max_be_wr = CONFIG.get("cost_viability_max_be_wr", 0.60)

    rows = []
    for inst in instruments:
        inst_cfg = CONFIG["instruments"].get(inst, {})
        spread_pips = inst_cfg.get("spread_pips", 1.0)
        pip_value = inst_cfg.get("pip", 0.0001)
        base_spread = spread_pips * pip_value

        for tf in timeframes:
            # Try to load cached data for a quick ATR estimate
            try:
                csv_path = Path(CONFIG.get("data_dir", "data")) / f"{inst}_{tf}.csv"
                if csv_path.exists():
                    df = data_loader.load_csv(csv_path)
                else:
                    # Fetch a small sample
                    df = data_loader.fetch_from_oanda(
                        instrument=inst, granularity=tf,
                        start=CONFIG.get("data_start", "2024-01-01"),
                        end=CONFIG.get("data_end", "2024-04-01"),
                    )

                if len(df) < 50:
                    logger.warning(
                        "Insufficient data for ATR calc: %s %s (%d candles)",
                        inst, tf, len(df),
                    )
                    rows.append({
                        "instrument": inst, "timeframe": tf,
                        "typical_atr": np.nan, "round_trip_cost_atr": np.nan,
                        "break_even_win_rate": np.nan, "viable": False,
                        "reason": "insufficient_data",
                    })
                    continue

                # Compute ATR on a sample
                high = df["high"]
                low = df["low"]
                prev_close = df["close"].shift(1)
                tr = pd.concat([
                    high - low,
                    (high - prev_close).abs(),
                    (low - prev_close).abs(),
                ], axis=1).max(axis=1)
                typical_atr = float(tr.tail(200).median())

                if typical_atr <= 0:
                    typical_atr = float(tr.median())

                # Round-trip cost in ATR units (session-open widened)
                round_trip_cost = (2 * base_spread * widen) / typical_atr

                # Break-even win rate for a 1:rr_ratio risk-reward
                # At RR = rr_ratio, profit per win = rr_ratio * ATR, loss per loss = 1 * ATR
                # BE: WR * rr_ratio - (1 - WR) * 1 = cost
                # WR * (rr_ratio + 1) = 1 + cost
                # WR = (1 + cost) / (rr_ratio + 1)
                break_even_wr = (1 + round_trip_cost) / (rr_ratio + 1)

                viable = break_even_wr < max_be_wr

                rows.append({
                    "instrument": inst,
                    "timeframe": tf,
                    "typical_atr": typical_atr,
                    "round_trip_cost_atr": round_trip_cost,
                    "break_even_win_rate": break_even_wr,
                    "viable": viable,
                    "reason": "" if viable else f"BE_WR={break_even_wr:.1%} >= {max_be_wr:.0%}",
                })

                logger.info(
                    "Cost viability %s %s: ATR=%.6f, cost=%.4f ATR, BE_WR=%.1f%% → %s",
                    inst, tf, typical_atr, round_trip_cost,
                    break_even_wr * 100, "VIABLE" if viable else "NOT VIABLE",
                )

            except Exception as e:
                logger.warning("Cost viability failed for %s %s: %s", inst, tf, e)
                rows.append({
                    "instrument": inst, "timeframe": tf,
                    "typical_atr": np.nan, "round_trip_cost_atr": np.nan,
                    "break_even_win_rate": np.nan, "viable": False,
                    "reason": f"error: {e}",
                })

    result_df = pd.DataFrame(rows)

    # Write markdown report
    out = Path(output_dir or CONFIG["output_dir"]) / "preflight"
    out.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Cost Viability Pre-Flight Analysis\n",
        f"Spread model: OANDA practice account, widening factor = {widen}×\n",
        f"Profit target: {rr_ratio} ATR, Stop: {CONFIG.get('stop_distance_atr', 1.0)} ATR\n",
        f"Viability threshold: break-even win rate < {max_be_wr:.0%}\n",
        "",
        "| Instrument | Timeframe | Typical ATR | Cost (ATR) | Break-Even WR | Viable? | Reason |",
        "|------------|-----------|-------------|------------|---------------|---------|--------|",
    ]

    for _, row in result_df.iterrows():
        atr_str = f"{row['typical_atr']:.6f}" if not pd.isna(row["typical_atr"]) else "N/A"
        cost_str = f"{row['round_trip_cost_atr']:.4f}" if not pd.isna(row["round_trip_cost_atr"]) else "N/A"
        be_str = f"{row['break_even_win_rate']:.1%}" if not pd.isna(row["break_even_win_rate"]) else "N/A"
        viable_str = "✅" if row["viable"] else "❌"
        lines.append(
            f"| {row['instrument']} | {row['timeframe']} | {atr_str} | "
            f"{cost_str} | {be_str} | {viable_str} | {row.get('reason', '')} |"
        )

    md_path = out / "cost_viability.md"
    md_path.write_text("\n".join(lines))
    logger.info("Cost viability report → %s", md_path)

    return result_df
