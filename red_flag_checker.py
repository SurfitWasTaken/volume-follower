from __future__ import annotations
"""
Red Flag Checker — automatic result validity assessment.

Evaluates every result against a battery of red-flag conditions.
Any triggered flag appears at the TOP of the conclusion, before
any performance numbers, making it impossible to misinterpret
a statistically invalid result as evidence of an edge.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from config import CONFIG

logger = logging.getLogger(__name__)


@dataclass
class RedFlag:
    """A single triggered red flag."""
    code: str
    severity: str  # "CRITICAL", "WARNING"
    message: str


@dataclass
class RedFlagReport:
    """Collection of red flags for a single result."""
    instrument: str
    timeframe: str
    variant: str
    n_signals: int
    flags: list[RedFlag] = field(default_factory=list)

    @property
    def n_flags(self) -> int:
        return len(self.flags)

    @property
    def n_critical(self) -> int:
        return sum(1 for f in self.flags if f.severity == "CRITICAL")

    @property
    def is_reportable(self) -> bool:
        """A result is reportable only if it has 0 CRITICAL flags."""
        return self.n_critical == 0

    def to_markdown(self) -> str:
        if self.n_flags == 0:
            return "✅ **0 RED FLAGS** — Result passes all validity checks.\n"

        lines = [
            f"## ⚠ {self.n_flags} RED FLAG{'S' if self.n_flags > 1 else ''} TRIGGERED — "
            f"READ BEFORE INTERPRETING RESULTS\n",
        ]
        for f in self.flags:
            icon = "🔴" if f.severity == "CRITICAL" else "🟡"
            lines.append(f"- {icon} **[{f.code}]** {f.message}")
        lines.append("")
        return "\n".join(lines)


def check_red_flags(result: dict, base_rate: float | None = None) -> RedFlagReport:
    """
    Evaluate a single result dict against all red-flag conditions.

    Parameters
    ----------
    result : dict
        A row from the summary table (or equivalent dict).
    base_rate : float or None
        Session-matched base rate for win-rate comparison.

    Returns
    -------
    RedFlagReport
    """
    report = RedFlagReport(
        instrument=result.get("instrument", "?"),
        timeframe=result.get("timeframe", "?"),
        variant=result.get("variant", "?"),
        n_signals=result.get("n_signals", 0),
    )

    n = result.get("n_signals", 0)
    sharpe = result.get("sharpe_net", np.nan)
    win_rate = result.get("win_rate", np.nan)
    wr_ci_lo = result.get("win_rate_ci_lo", np.nan)
    cohens_h = result.get("cohens_h", np.nan)
    first_wr = result.get("first_half_win_rate", np.nan)
    second_wr = result.get("second_half_win_rate", np.nan)
    perm_rank = result.get("perm_empirical_rank", np.nan)
    br = base_rate if base_rate is not None else result.get("base_rate", 0.5)

    max_sharpe = CONFIG.get("max_reportable_sharpe", 5.0)
    min_h = CONFIG.get("min_cohens_h", 0.05)
    max_delta = CONFIG.get("max_stationarity_delta", 0.10)

    # ── Flag 1: Insufficient sample size ──
    if n < 100:
        report.flags.append(RedFlag(
            code="INSUFFICIENT_N",
            severity="CRITICAL",
            message=f"Only {n} signals (minimum 100 required). "
                    f"No statistical claims can be made.",
        ))

    # ── Flag 2: Sharpe too good on small sample ──
    if not np.isnan(sharpe) and sharpe > 3.0 and n < 200:
        report.flags.append(RedFlag(
            code="SHARPE_ON_SMALL_N",
            severity="CRITICAL",
            message=f"Sharpe {sharpe:.2f} on only {n} signals — almost certainly "
                    f"noise or look-ahead bias.",
        ))

    # ── Flag 3: Sharpe suspiciously high ──
    if not np.isnan(sharpe) and sharpe > max_sharpe:
        report.flags.append(RedFlag(
            code="EXTREME_SHARPE",
            severity="CRITICAL",
            message=f"Sharpe {sharpe:.2f} exceeds {max_sharpe:.1f} — verify code "
                    f"for look-ahead bias before proceeding.",
        ))

    # ── Flag 4: Win rate CI includes base rate ──
    if not np.isnan(wr_ci_lo) and not np.isnan(br) and wr_ci_lo < br:
        report.flags.append(RedFlag(
            code="WR_CI_BELOW_BASE",
            severity="WARNING",
            message=f"Win rate CI lower bound ({wr_ci_lo:.1%}) is below base rate "
                    f"({br:.1%}) — not statistically significant.",
        ))

    # ── Flag 5: Non-stationary edge ──
    if not np.isnan(first_wr) and not np.isnan(second_wr):
        delta = abs(first_wr - second_wr)
        if delta > max_delta:
            report.flags.append(RedFlag(
                code="UNSTABLE_EDGE",
                severity="WARNING",
                message=f"Win rate varies by {delta:.1%} between first half "
                        f"({first_wr:.1%}) and second half ({second_wr:.1%}) — "
                        f"likely regime-dependent, not systematic.",
            ))

    # ── Flag 6: Fails permutation test ──
    if not np.isnan(perm_rank) and perm_rank < 0.90:
        report.flags.append(RedFlag(
            code="PERM_TEST_FAIL",
            severity="WARNING",
            message=f"True Sharpe sits at {perm_rank:.0%} of permutation distribution "
                    f"(need top 10%).",
        ))

    # ── Flag 7: Trivially small effect size ──
    if not np.isnan(cohens_h) and abs(cohens_h) < min_h:
        report.flags.append(RedFlag(
            code="TRIVIAL_EFFECT",
            severity="WARNING",
            message=f"Cohen's h = {cohens_h:.4f} — effect size too small to be "
                    f"economically meaningful regardless of p-value.",
        ))

    # ── Flag 8: Large WR delta between halves ──
    if not np.isnan(first_wr) and not np.isnan(second_wr):
        delta = abs(first_wr - second_wr)
        if delta > 0.15:
            report.flags.append(RedFlag(
                code="EXTREME_WR_SHIFT",
                severity="CRITICAL",
                message=f"Win rate shifts by {delta:.0%} between time halves — "
                        f"suggests the signal works only in a specific regime.",
            ))

    if report.n_flags > 0:
        logger.warning(
            "Red flags for %s %s %s: %d flags (%d critical)",
            report.instrument, report.timeframe, report.variant,
            report.n_flags, report.n_critical,
        )

    return report


def check_all_results(
    results: list[dict],
    base_rates: dict[str, float] | None = None,
) -> list[RedFlagReport]:
    """
    Run red flag checks on all results.

    Parameters
    ----------
    results : list[dict]
        Full results list from the pipeline.
    base_rates : dict mapping some key to base rate, optional.

    Returns
    -------
    list[RedFlagReport]
    """
    reports = []
    for r in results:
        br = base_rates.get(
            f"{r.get('instrument')}_{r.get('timeframe')}_{r.get('K')}",
            r.get("base_rate", 0.5),
        ) if base_rates else r.get("base_rate", 0.5)

        reports.append(check_red_flags(r, base_rate=br))

    n_total = len(reports)
    n_critical = sum(1 for r in reports if r.n_critical > 0)
    logger.info(
        "Red flag check: %d/%d results have CRITICAL flags.",
        n_critical, n_total,
    )
    return reports
