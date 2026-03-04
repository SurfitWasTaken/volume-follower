from __future__ import annotations
"""
Performance reporting and visualisation module.

Generates all required output artefacts:
1. Signal summary table
2. Equity curves with drawdown shading
3. Distribution plots (signal vs baseline)
4. Walk-forward Sharpe box plot
5. Permutation test histogram
6. MFE / MAE scatter
7. Session heatmap
8. Structured conclusion (templated, not free-form)
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from config import CONFIG

logger = logging.getLogger(__name__)

# Seaborn style
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.05)


class PerformanceReporter:
    """Generate tables, plots, and a structured conclusion."""

    def __init__(self, output_dir: str | None = None):
        self.output_dir = Path(output_dir or CONFIG["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ─── 1. Summary table ────────────────────────────────────────────────

    @staticmethod
    def summary_table(results: list[dict]) -> pd.DataFrame:
        """
        Combine per-variant results into the master summary DataFrame.

        Expected keys per result dict:
            variant, filters_applied, instrument, timeframe, K,
            n_signals, win_rate, base_rate, win_rate_excess,
            p_value_raw, p_value_corrected,
            avg_return_gross, avg_return_net,
            sharpe_gross, sharpe_net,
            avg_mfe_atr, avg_mae_atr, mfe_mae_ratio,
            win_rate_ci_lo, win_rate_ci_hi,
            sharpe_ci_lo, sharpe_ci_hi
        """
        df = pd.DataFrame(results)
        cols_order = [
            "instrument", "timeframe", "variant", "filters_applied", "K",
            "n_signals", "win_rate", "base_rate", "win_rate_excess",
            "p_value_raw", "p_value_corrected", "cohens_h", "sample_warning",
            "avg_return_gross", "avg_return_net",
            "sharpe_gross", "sharpe_net",
            "avg_mfe_atr", "avg_mae_atr", "mfe_mae_ratio",
            "win_rate_ci_lo", "win_rate_ci_hi",
            "sharpe_ci_lo", "sharpe_ci_hi",
        ]
        for c in cols_order:
            if c not in df.columns:
                df[c] = np.nan
        return df[cols_order].sort_values(
            ["instrument", "timeframe", "variant", "K"]
        ).reset_index(drop=True)

    # ─── 2. Equity curves ────────────────────────────────────────────────

    def plot_equity_curves(
        self,
        equity_dict: dict[str, pd.Series],
        title: str = "Cumulative PnL (ATR units, net of costs)",
        filename: str = "equity_curves.png",
    ) -> Path:
        """
        Plot cumulative PnL for multiple variants on one chart.
        Mark drawdown periods with shading.

        Parameters
        ----------
        equity_dict : {label: pd.Series of cumulative returns}
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        for label, eq in equity_dict.items():
            ax.plot(eq.index, eq.values, label=label, linewidth=1.2)
            # Drawdown shading
            peak = eq.cummax()
            dd = eq - peak
            ax.fill_between(eq.index, eq.values, peak.values,
                            where=dd < 0, alpha=0.15)

        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Signal #")
        ax.set_ylabel("Cumulative Return (ATR)")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper left")
        plt.tight_layout()

        path = self.output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved equity curve → %s", path)
        return path

    # ─── 3. Distribution plots ───────────────────────────────────────────

    def plot_distribution(
        self,
        signal_returns: np.ndarray,
        all_returns: np.ndarray,
        K: int,
        title: str | None = None,
        filename: str = "return_distribution.png",
    ) -> Path:
        """Overlay histograms of signal returns vs all-candle returns."""
        fig, ax = plt.subplots(figsize=(10, 5))

        bins = np.linspace(
            min(np.nanmin(signal_returns), np.nanmin(all_returns)),
            max(np.nanmax(signal_returns), np.nanmax(all_returns)),
            60,
        )
        ax.hist(all_returns, bins=bins, alpha=0.4, density=True,
                label=f"All candles (n={len(all_returns)})", color="grey")
        ax.hist(signal_returns, bins=bins, alpha=0.6, density=True,
                label=f"Signal candles (n={len(signal_returns)})", color="steelblue")

        ax.axvline(np.nanmean(signal_returns), color="blue", linestyle="--",
                   label=f"Signal mean: {np.nanmean(signal_returns):.4f}")
        ax.axvline(np.nanmean(all_returns), color="grey", linestyle="--",
                   label=f"Baseline mean: {np.nanmean(all_returns):.4f}")

        ax.set_xlabel(f"Forward {K}-candle return (ATR)")
        ax.set_ylabel("Density")
        ax.set_title(title or f"Return Distribution: Signal vs Baseline (K={K})")
        ax.legend(fontsize=8)
        plt.tight_layout()

        path = self.output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved distribution → %s", path)
        return path

    # ─── 4. Walk-forward Sharpe box plot ──────────────────────────────────

    def plot_walk_forward_sharpe(
        self,
        wf_results: list[dict],
        filename: str = "walk_forward_sharpe.png",
    ) -> Path:
        """Box plot of rolling out-of-window Sharpe ratios."""
        sharpes = [r["sharpe"] for r in wf_results if not np.isnan(r.get("sharpe", np.nan))]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(sharpes, vert=True, widths=0.4, patch_artist=True,
                   boxprops=dict(facecolor="lightblue"))
        ax.axhline(0, color="red", linestyle="--", linewidth=0.8, label="Sharpe = 0")
        ax.set_ylabel("Sharpe Ratio (annualised)")
        ax.set_title("Walk-Forward Sharpe Distribution")
        ax.set_xticklabels(["Out-of-sample folds"])
        ax.legend()
        plt.tight_layout()

        path = self.output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved walk-forward → %s", path)
        return path

    # ─── 5. Permutation test histogram ───────────────────────────────────

    def plot_permutation_test(
        self,
        perm_result: dict,
        filename: str = "permutation_test.png",
    ) -> Path:
        """Histogram of permuted Sharpes + vertical line for true Sharpe."""
        fig, ax = plt.subplots(figsize=(10, 5))

        perm_sharpes = np.array(perm_result["perm_sharpes"])
        true_sharpe = perm_result["true_sharpe"]
        emp_p = perm_result["empirical_p_value"]

        ax.hist(perm_sharpes, bins=50, alpha=0.6, color="grey",
                edgecolor="white", label="Permuted Sharpes")
        ax.axvline(true_sharpe, color="red", linewidth=2, linestyle="--",
                   label=f"True Sharpe = {true_sharpe:.3f}")
        ax.set_xlabel("Sharpe Ratio")
        ax.set_ylabel("Count")
        ax.set_title(f"Permutation Test  (empirical p = {emp_p:.4f})")
        ax.legend()
        plt.tight_layout()

        path = self.output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved permutation → %s", path)
        return path

    # ─── 6. MFE / MAE scatter ────────────────────────────────────────────

    def plot_mfe_mae(
        self,
        outcomes: pd.DataFrame,
        K: int = 10,
        filename: str = "mfe_mae_scatter.png",
    ) -> Path:
        """Scatter of MFE vs MAE colour-coded by win/loss."""
        mfe_col = f"mfe_{K}"
        mae_col = f"mae_{K}"
        win_col = f"win_{K}"

        if mfe_col not in outcomes.columns:
            logger.warning("MFE/MAE columns not found for K=%d", K)
            return self.output_dir / filename

        fig, ax = plt.subplots(figsize=(8, 8))

        wins = outcomes[outcomes[win_col] == 1.0]
        losses = outcomes[outcomes[win_col] == 0.0]
        undecided = outcomes[outcomes[win_col] == 0.5]

        ax.scatter(wins[mae_col], wins[mfe_col], alpha=0.4, s=12,
                   c="green", label=f"Win (n={len(wins)})")
        ax.scatter(losses[mae_col], losses[mfe_col], alpha=0.4, s=12,
                   c="red", label=f"Loss (n={len(losses)})")
        if len(undecided) > 0:
            ax.scatter(undecided[mae_col], undecided[mfe_col], alpha=0.3,
                       s=12, c="grey", label=f"Undecided (n={len(undecided)})")

        # 45-degree line
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], "k--", linewidth=0.5, label="MFE = MAE")

        ax.set_xlabel("MAE (ATR)")
        ax.set_ylabel("MFE (ATR)")
        ax.set_title(f"MFE vs MAE  (K={K})")
        ax.legend(fontsize=8)
        plt.tight_layout()

        path = self.output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved MFE/MAE → %s", path)
        return path

    # ─── 7. Session heatmap ──────────────────────────────────────────────

    def plot_session_heatmap(
        self,
        outcomes: pd.DataFrame,
        K: int = 10,
        filename: str = "session_heatmap.png",
    ) -> Path:
        """Heatmap of win rate by hour-of-day (rows) × day-of-week (cols)."""
        win_col = f"win_{K}"
        if win_col not in outcomes.columns or len(outcomes) == 0:
            logger.warning("Cannot plot heatmap — no outcomes.")
            return self.output_dir / filename

        outcomes = outcomes.copy()
        outcomes["win_binary"] = (outcomes[win_col] == 1.0).astype(float)

        pivot = outcomes.pivot_table(
            values="win_binary", index="hour", columns="day_of_week",
            aggfunc=["mean", "count"],
        )

        # Extract win-rate matrix
        wr = pivot["mean"] if "mean" in pivot.columns.get_level_values(0) else pivot
        counts = pivot["count"] if "count" in pivot.columns.get_level_values(0) else None

        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        wr.columns = [day_labels[c] if c < len(day_labels) else str(c) for c in wr.columns]

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            wr, annot=True, fmt=".2f", cmap="RdYlGn", center=0.5,
            vmin=0.3, vmax=0.7, ax=ax, linewidths=0.5,
        )
        ax.set_xlabel("Day of Week")
        ax.set_ylabel("Hour (UTC)")
        ax.set_title(f"Signal Win Rate by Session Hour × Day  (K={K})")
        plt.tight_layout()

        path = self.output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved heatmap → %s", path)
        return path

    # ─── 8. Structured conclusion ────────────────────────────────────────

    def write_conclusion(
        self,
        summary_df: pd.DataFrame,
        perm_result: dict | None = None,
        wf_results: list[dict] | None = None,
        stationarity: dict | None = None,
        cc_validity: dict | None = None,
        filename: str = "conclusion.md",
    ) -> Path:
        """
        Synthesise all results into a final markdown report.
        """
        sig_level = CONFIG["significance_level"]
        lines = [
            "# Backtest Conclusion & Strategy Assessment\n",
            "## 1. Executive Summary\n",
        ]

        if len(summary_df) == 0:
            lines.append("No results found in summary table.")
        else:
            best = summary_df.loc[summary_df["sharpe_net"].idxmax()] if not summary_df["sharpe_net"].isna().all() else summary_df.iloc[0]
            lines.append(f"The best performing variant was **{best['instrument']} {best['timeframe']} {best['variant']}** with filters **{best['filters_applied']}** (K={best['K']}).")
            lines.append(f"- Sharpe (Net): {best['sharpe_net']:.3f}")
            lines.append(f"- Win Rate: {best['win_rate']:.1%} (vs baseline {best['base_rate']:.1%})")
            lines.append(f"- Total Signals: {best['n_signals']}")
            lines.append(f"- Edge significance (p-value): {best['p_value_corrected']:.4f}")

        # Cross-Currency Validity
        if cc_validity:
            lines.extend(["", "## 2. Cross-Currency Decomposition Validity", ""])
            lines.append(f"- **EUR-driven Continuation Rate**: {cc_validity.get('eur_driven_success_rate', np.nan):.1%} ({cc_validity.get('eur_driven_n', 0)} signals)")
            lines.append(f"- **USD-driven Mirroring Rate**: {cc_validity.get('usd_driven_mirror_rate', np.nan):.1%} ({cc_validity.get('usd_driven_n', 0)} signals)")
            lines.append("  > [!NOTE]")
            lines.append("  > High continuation on EUR-driven and high mirroring on USD-driven validates the decomposition hypothesis.")

        # Statistical Guardrails
        lines.extend(["", "## 3. Statistical Guardrails", ""])
        n_significant_raw = (summary_df["p_value_raw"] < sig_level).sum() if "p_value_raw" in summary_df.columns else 0
        n_significant_fdr = (summary_df["p_value_corrected"] < sig_level).sum() if "p_value_corrected" in summary_df.columns else 0
        n_total = len(summary_df)

        lines.extend([
            "## 3.1. Statistical Significance",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total tests conducted | {n_total} |",
            f"| Significant at α={sig_level} (raw) | {n_significant_raw} |",
            f"| Significant after FDR correction | {n_significant_fdr} |",
            f"| Edge exists (after correction)? | **{'YES' if n_significant_fdr > 0 else 'NO'}** |",
            "",
        ])

        if not best.empty:
            lines.extend([
                "## 4. Best Signal Configuration",
                "",
                f"| Parameter | Value |",
                f"|-----------|-------|",
                f"| Instrument | {best.get('instrument', 'N/A')} |",
                f"| Timeframe | {best.get('timeframe', 'N/A')} |",
                f"| Variant | {best.get('variant', 'N/A')} |",
                f"| Filters | {best.get('filters_applied', 'N/A')} |",
                f"| K (forward window) | {best.get('K', 'N/A')} |",
                f"| n signals | {best.get('n_signals', 'N/A')} |",
                f"| Win rate | {best.get('win_rate', np.nan):.4f} |",
                f"| Base rate | {best.get('base_rate', np.nan):.4f} |",
                f"| Win rate excess | {best.get('win_rate_excess', np.nan):.4f} |",
                f"| p-value (corrected) | {best.get('p_value_corrected', np.nan):.4f} |",
                f"| Sharpe (gross) | {best.get('sharpe_gross', np.nan):.3f} |",
                f"| Sharpe (net of costs) | {best.get('sharpe_net', np.nan):.3f} |",
                f"| MFE/MAE ratio | {best.get('mfe_mae_ratio', np.nan):.3f} |",
                "",
            ])

        # Costs
        lines.extend([
            "## 3. Transaction Cost Impact",
            "",
        ])
        if "sharpe_gross" in summary_df.columns and "sharpe_net" in summary_df.columns:
            mean_gross = summary_df["sharpe_gross"].mean()
            mean_net = summary_df["sharpe_net"].mean()
            lines.append(f"- Mean Sharpe (gross): {mean_gross:.3f}")
            lines.append(f"- Mean Sharpe (net):   {mean_net:.3f}")
            lines.append(f"- Edge survives costs? **{'YES' if mean_net > 0 else 'NO'}**")
        lines.append("")

        # Stability
        lines.extend(["## 4. Temporal Stability", ""])
        if stationarity:
            lines.append(f"- First-half Sharpe: {stationarity.get('first_half_sharpe', np.nan):.3f}")
            lines.append(f"- Second-half Sharpe: {stationarity.get('second_half_sharpe', np.nan):.3f}")
            lines.append(f"- Sharpe difference: {stationarity.get('sharpe_diff', np.nan):.3f}")
            stable = stationarity.get("sharpe_diff", 999) < 1.0
            lines.append(f"- Edge stable? **{'YES' if stable else 'NO — significant decay'}**")
        if wf_results:
            wf_sharpes = [r["sharpe"] for r in wf_results if not np.isnan(r.get("sharpe", np.nan))]
            pct_pos = sum(1 for s in wf_sharpes if s > 0) / max(len(wf_sharpes), 1)
            lines.append(f"- Walk-forward folds with positive Sharpe: {pct_pos:.0%} ({len(wf_sharpes)} folds)")
        lines.append("")

        # Permutation
        lines.extend(["## 5. Permutation Test", ""])
        if perm_result:
            lines.append(f"- True Sharpe: {perm_result['true_sharpe']:.3f}")
            lines.append(f"- Empirical p-value: {perm_result['empirical_p_value']:.4f}")
            lines.append(f"- Signal is {'**NOT noise**' if perm_result['empirical_p_value'] < sig_level else '**indistinguishable from noise**'}")
        lines.append("")

        # Caveats
        lines.extend([
            "## 6. Caveats",
            "",
            "- OANDA tick volume is a proxy for real volume (correlation ≈ 0.7–0.85).",
            "- CFD instruments (SPX500, NAS100, XAU) may have different microstructure than exchange-traded futures.",
            "- Hardcoded news calendar may miss some events.",
            "- Past edge does not guarantee future edge — microstructure strategies are sensitive to regime change.",
            "- Each timeframe is tested independently; results should NOT be pooled across timeframes.",
            "",
            "## 7. Recommended Next Steps",
            "",
            "- [ ] Validate on exchange data (real volume) if edge found",
            "- [ ] Live paper-trade for ≥3 months before capital deployment",
            "- [ ] Monitor for edge decay with monthly Sharpe tracking",
            "- [ ] Investigate economic mechanism of any discovered edge",
        ])

        text = "\n".join(lines)
        path = self.output_dir / filename
        path.write_text(text)
        logger.info("Saved conclusion → %s", path)
        return path

    @staticmethod
    def _df_to_markdown(df: pd.DataFrame, index: bool = True) -> str:
        """Manual markdown table generator to avoid 'tabulate' dependency."""
        if df.empty:
            return "No data available."
        
        cols = list(df.columns)
        if index:
            cols = [df.index.name or ""] + cols
            
        header = "| " + " | ".join(map(str, cols)) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        
        rows = []
        for idx, row in df.iterrows():
            vals = list(row.values)
            if index:
                vals = [idx] + vals
            rows.append("| " + " | ".join(map(str, vals)) + " |")
            
        return "\n".join([header, sep] + rows)

    # ─── 9. Cross-Currency Specific Plots ─────────────────────────────────

    def plot_cc_adaptive_equity(
        self,
        uniform_eq: pd.Series,
        adaptive_eq: pd.Series,
        title: str = "Equity Curve: Uniform vs Adaptive Sizing",
        filename: str = "cc_adaptive_equity.png",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(uniform_eq.index, uniform_eq.values, label="Uniform Sizing", linewidth=1.2, color="grey")
        ax.plot(adaptive_eq.index, adaptive_eq.values, label="Adaptive Sizing", linewidth=1.5, color="blue")
        
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Signal #")
        ax.set_ylabel("Cumulative Return (ATR)")
        ax.set_title(title)
        ax.legend(fontsize=10, loc="upper left")
        plt.tight_layout()

        path = self.output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_cc_eur_move_distribution(
        self,
        features_wins: pd.DataFrame,
        features_losses: pd.DataFrame,
        filename: str = "cc_eur_move_distribution.png"
    ) -> Path:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        mx = max(features_wins["eur_move_score"].max(), features_losses["eur_move_score"].max())
        mn = min(features_wins["eur_move_score"].min(), features_losses["eur_move_score"].min())
        
        bins = np.linspace(mn if not pd.isna(mn) else -0.01, mx if not pd.isna(mx) else 0.01, 50)
        
        ax.hist(features_losses["eur_move_score"], bins=bins, alpha=0.5, density=True, color="red", label="Losses")
        ax.hist(features_wins["eur_move_score"], bins=bins, alpha=0.5, density=True, color="green", label="Wins")
        
        ax.axvline(0, color="k", linestyle="--", linewidth=1)
        ax.set_xlabel("EUR Move Score")
        ax.set_ylabel("Density")
        ax.set_title("EUR Move Score Distribution: Wins vs Losses")
        ax.legend()
        plt.tight_layout()

        path = self.output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
        
    def write_cc_summary(
        self,
        classification_summary: pd.DataFrame,
        regime_summary: pd.DataFrame,
        catch_up_stats: dict,
        filename: str = "cc_summary.md"
    ) -> Path:
        lines = [
            "# Cross-Currency Confirmation Summary\n",
            "## 1. Classification Breakdown\n",
            self._df_to_markdown(classification_summary, index=False),
            "\n## 2. Correlation Regime Performance\n",
            self._df_to_markdown(regime_summary, index=False),
            "\n## 3. GBP/USD Catch-Up Analysis\n",
            f"- EUR-driven signals followed by pure continuation: {catch_up_stats.get('eur_driven_success_rate', np.nan):.1%}",
            f"- USD-driven signals correctly mirroring: {catch_up_stats.get('usd_driven_mirror_rate', np.nan):.1%}",
        ]
        
        text = "\n".join(lines)
        path = self.output_dir / filename
        path.write_text(text)
        logger.info("Saved CC summary → %s", path)
        return path
