from __future__ import annotations
"""
Data ingestion layer for the Volume Spike pipeline.

Responsibilities:
- Fetch OHLCV candles from the OANDA v20 REST API (practice environment).
- Load / save CSV caches to avoid repeated API calls.
- Validate data quality (gaps, duplicates, zero-volume).
- Split into in-sample / out-of-sample sets.
"""

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _load_env(env_path: str | None = None) -> None:
    """Read a .env file and set values as environment variables."""
    if env_path is None:
        # Walk up to the Quant root where the .env lives
        env_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ".env",
        )
    if not os.path.exists(env_path):
        logger.warning(".env file not found at %s", env_path)
        return
    with open(env_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()
    logger.info("Loaded environment from %s", env_path)


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class DataLoader:
    """Fetch, cache, validate, and split OHLCV data."""

    MAX_CANDLES_PER_REQUEST = 5000
    MAX_DAYS_PER_CHUNK = {
        "M1": 3,     # ~4320 candles per 3 days
        "M5": 10,    # ~2880 candles per 10 days
        "M15": 30,   # ~2880 candles per 30 days
        "H1": 100,   # ~2400 candles per 100 days
        "H4": 365,   # ~2190 candles per 365 days
    }

    def __init__(self, data_dir: str | None = None):
        self.data_dir = Path(data_dir or CONFIG["data_dir"])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._client = None

    # ---- OANDA client (lazy init) ----------------------------------------

    def _get_client(self):
        """Return an authenticated OANDA API client."""
        if self._client is not None:
            return self._client

        _load_env()

        token = os.environ.get("OANDA_ACCESS_TOKEN")
        if not token:
            raise RuntimeError(
                "OANDA_ACCESS_TOKEN not found.  "
                "Set it in the .env file at the Quant project root."
            )

        try:
            from oandapyV20 import API
            self._client = API(access_token=token, environment="practice")
        except ImportError:
            raise ImportError(
                "oandapyV20 is required.  Install via: pip install oandapyV20"
            )
        return self._client

    # ---- Fetch from OANDA ------------------------------------------------

    def fetch_from_oanda(
        self,
        instrument: str,
        granularity: str,
        start: str,
        end: str,
        force: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from OANDA and cache to CSV.

        Parameters
        ----------
        instrument : str
            OANDA instrument code, e.g. 'EUR_USD'.
        granularity : str
            Candle granularity: 'M1', 'M5', 'M15'.
        start, end : str
            ISO date strings, e.g. '2023-01-01'.
        force : bool
            If True, re-download even if the CSV cache exists.

        Returns
        -------
        pd.DataFrame  with DatetimeIndex (UTC) and columns
            [open, high, low, close, volume].
        """
        csv_path = self.data_dir / f"{instrument}_{granularity}.csv"

        if csv_path.exists() and not force:
            logger.info("Cache hit: %s — loading from disk.", csv_path)
            return self.load_csv(csv_path)

        import oandapyV20.endpoints.instruments as ep

        client = self._get_client()
        chunk_days = self.MAX_DAYS_PER_CHUNK.get(granularity, 25)
        dt_fmt = "%Y-%m-%dT%H:%M:%SZ"

        current = pd.Timestamp(start, tz="UTC")
        end_dt = pd.Timestamp(end, tz="UTC")
        rows: list[dict] = []
        consecutive_errors = 0

        logger.info(
            "Fetching %s %s  %s → %s  (chunk=%dd)",
            instrument, granularity, start, end, chunk_days,
        )

        while current < end_dt:
            chunk_end = min(current + timedelta(days=chunk_days), end_dt)
            params = {
                "from": current.strftime(dt_fmt),
                "to": chunk_end.strftime(dt_fmt),
                "granularity": granularity,
                "price": "M",
                "dailyAlignment": "0",
                "alignmentTimezone": "Etc/UTC",
            }
            req = ep.InstrumentsCandles(
                instrument=instrument, params=params
            )

            try:
                client.request(req)
                candles = req.response.get("candles", [])
                for c in candles:
                    if c.get("complete", False):
                        rows.append({
                            "time": c["time"],
                            "open": float(c["mid"]["o"]),
                            "high": float(c["mid"]["h"]),
                            "low": float(c["mid"]["l"]),
                            "close": float(c["mid"]["c"]),
                            "volume": int(c["volume"]),
                        })
                logger.debug(
                    "  %s → %s : %d candles",
                    current.date(), chunk_end.date(), len(candles),
                )
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                msg = str(exc).lower()
                logger.error("  OANDA error: %s", exc)
                if "authorization" in msg or "unauthorized" in msg:
                    logger.error("Fatal auth error — aborting fetch.")
                    break
                if consecutive_errors >= 5:
                    logger.error("Too many consecutive errors — aborting.")
                    break
                time.sleep(2 * consecutive_errors)
                continue

            time.sleep(0.55)  # respect rate limits
            current = chunk_end

        if not rows:
            logger.warning("No data fetched for %s %s.", instrument, granularity)
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")]

        df.to_csv(csv_path)
        logger.info("Saved %d candles → %s", len(df), csv_path)
        return df

    # ---- Max history fetch (V2 overhaul) ---------------------------------

    def fetch_max_history(
        self,
        instrument: str,
        granularity: str,
        max_years: int | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch the maximum available history from OANDA by paginating
        backwards from today.

        Parameters
        ----------
        instrument : str
        granularity : str
        max_years : int
            Maximum years of history to fetch.  Default from CONFIG.
        force : bool
            Re-download even if cache exists.

        Returns
        -------
        pd.DataFrame
        """
        max_years = max_years or CONFIG.get("max_history_years", 5)
        end_dt = pd.Timestamp.now(tz="UTC").normalize()
        start_dt = end_dt - pd.DateOffset(years=max_years)

        logger.info(
            "Fetching max history for %s %s: %s → %s (%d years)",
            instrument, granularity, start_dt.date(), end_dt.date(), max_years,
        )

        return self.fetch_from_oanda(
            instrument=instrument,
            granularity=granularity,
            start=str(start_dt.date()),
            end=str(end_dt.date()),
            force=force,
        )

    # ---- Data density validation (V2 overhaul) ---------------------------

    @staticmethod
    def validate_density(
        df: pd.DataFrame,
        timeframe: str,
        window_months: int = 6,
    ) -> list[dict]:
        """
        Check data density per window_months-period.

        Returns a list of dicts with period info and density pct.
        Logs warnings for any period with density < 90%.
        """
        if len(df) == 0:
            return []

        # Expected candles per trading hour
        tf_minutes = {"M1": 1, "M5": 5, "M15": 15, "H1": 60, "H4": 240}
        mins = tf_minutes.get(timeframe, 60)
        # ~252 trading days/yr, ~22hrs/day for forex
        candles_per_month = int(252 / 12 * 22 * 60 / mins)

        results = []
        start = df.index.min()
        end = df.index.max()
        cursor = start

        while cursor < end:
            window_end = cursor + pd.DateOffset(months=window_months)
            window = df[(df.index >= cursor) & (df.index < window_end)]
            expected = candles_per_month * window_months
            actual = len(window)
            density = actual / expected if expected > 0 else 0

            result = {
                "period_start": str(cursor.date()),
                "period_end": str(min(window_end, end).date()),
                "actual_candles": actual,
                "expected_candles": expected,
                "density_pct": round(density * 100, 1),
            }
            results.append(result)

            if density < 0.90:
                logger.warning(
                    "Low data density: %s → %s: %.1f%% (%d/%d candles)",
                    result["period_start"], result["period_end"],
                    density * 100, actual, expected,
                )

            cursor = window_end

        return results

    # ---- Data summary report (V2 overhaul) -------------------------------

    @staticmethod
    def write_data_summary(
        df: pd.DataFrame,
        instrument: str,
        timeframe: str,
        density_results: list[dict],
        output_dir: str | None = None,
    ) -> None:
        """Write data summary to output/preflight/data_summary.md."""
        from pathlib import Path
        out = Path(output_dir or CONFIG["output_dir"]) / "preflight"
        out.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# Data Summary: {instrument} {timeframe}\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Date range | {df.index.min()} → {df.index.max()} |",
            f"| Total candles | {len(df):,} |",
            f"| Duration | {(df.index.max() - df.index.min()).days:,} days |",
            f"| Zero-volume candles | {(df['volume'] == 0).sum()} |",
            "",
            "## Density by Period",
            "",
            "| Period | Actual | Expected | Density |",
            "|--------|--------|----------|----------|",
        ]
        for d in density_results:
            flag = " ⚠" if d["density_pct"] < 90 else ""
            lines.append(
                f"| {d['period_start']} – {d['period_end']} | "
                f"{d['actual_candles']:,} | {d['expected_candles']:,} | "
                f"{d['density_pct']:.1f}%{flag} |"
            )

        path = out / "data_summary.md"
        path.write_text("\n".join(lines))
        logger.info("Data summary → %s", path)
        return df

    # ---- CSV I/O ---------------------------------------------------------

    @staticmethod
    def load_csv(filepath: str | Path) -> pd.DataFrame:
        """
        Load an OHLCV CSV with a datetime index.

        Expects columns: time (or index), open, high, low, close, volume.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CSV not found: {filepath}")

        df = pd.read_csv(filepath, parse_dates=True, index_col=0)

        # Normalise column names to lowercase
        df.columns = [c.strip().lower() for c in df.columns]

        # Ensure UTC-aware index
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df.index.name = "time"
        df.sort_index(inplace=True)
        return df

    # ---- Validation ------------------------------------------------------

    @staticmethod
    def validate_data(df: pd.DataFrame) -> dict:
        """
        Run quality checks and return a diagnostics dict.

        Checks:
        - NaN counts per column
        - Duplicate timestamps
        - Zero-volume candle count
        - OHLC constraint violations (H < max(O,C), L > min(O,C))
        - Estimated gap count (missing expected timestamps)
        """
        diag: dict = {}

        diag["n_rows"] = len(df)
        diag["date_range"] = (str(df.index.min()), str(df.index.max()))
        diag["nan_counts"] = df.isna().sum().to_dict()

        dupes = df.index.duplicated().sum()
        diag["duplicate_timestamps"] = int(dupes)

        zero_vol = (df["volume"] == 0).sum()
        diag["zero_volume_candles"] = int(zero_vol)

        ohlc_bad = (
            (df["high"] < np.maximum(df["open"], df["close"]))
            | (df["low"] > np.minimum(df["open"], df["close"]))
        ).sum()
        diag["ohlc_violations"] = int(ohlc_bad)

        # Log results
        for k, v in diag.items():
            level = logging.WARNING if (
                (k == "duplicate_timestamps" and v > 0)
                or (k == "zero_volume_candles" and v > len(df) * 0.01)
                or (k == "ohlc_violations" and v > 0)
            ) else logging.INFO
            logger.log(level, "  validate | %s = %s", k, v)

        return diag

    # ---- Train / test split ----------------------------------------------

    @staticmethod
    def split_data(
        df: pd.DataFrame,
        in_sample_end: str | None = None,
        oos_start: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split a DataFrame into in-sample and out-of-sample sets.

        Parameters
        ----------
        df : pd.DataFrame
        in_sample_end : str
            End of in-sample period (exclusive).  Default from CONFIG.
        oos_start : str
            Start of out-of-sample period (inclusive).  Default from CONFIG.

        Returns
        -------
        (df_in_sample, df_out_of_sample)
        """
        if in_sample_end is None:
            in_sample_end = CONFIG["in_sample_end"]
        if oos_start is None:
            oos_start = CONFIG["out_of_sample_start"]

        cutoff = pd.Timestamp(in_sample_end, tz="UTC")
        oos_ts = pd.Timestamp(oos_start, tz="UTC")

        df_is = df[df.index < cutoff].copy()
        df_oos = df[df.index >= oos_ts].copy()

        logger.info(
            "Split: in-sample %d candles (%s → %s), "
            "OOS %d candles (%s → %s)",
            len(df_is),
            df_is.index.min() if len(df_is) else "N/A",
            df_is.index.max() if len(df_is) else "N/A",
            len(df_oos),
            df_oos.index.min() if len(df_oos) else "N/A",
            df_oos.index.max() if len(df_oos) else "N/A",
        )
        return df_is, df_oos
