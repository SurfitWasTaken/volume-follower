from __future__ import annotations
"""
Pre-commitment logging module — anti-p-hacking enforcement.

Locks the full test matrix into an immutable SHA-256 hash BEFORE any
data is loaded or results are observed.  At run end, re-hashes and
compares — if the matrix was modified mid-run, the contamination flag
fires and every output file is stamped with a warning.
"""

import hashlib
import json
import logging
from pathlib import Path
from itertools import product

from config import CONFIG

logger = logging.getLogger(__name__)


def _build_test_matrix(
    instruments: list[str],
    timeframes: list[str],
    variants: list[str],
    session_norm_options: list[bool],
    filter_ladders: list[list[str]],
    k_values: list[int],
) -> list[dict]:
    """Generate the full pre-committed test list."""
    matrix = []
    for inst in instruments:
        for tf in timeframes:
            for var in variants:
                for sn in session_norm_options:
                    if var == "D" and sn:
                        continue
                    for filters in filter_ladders:
                        for K in k_values:
                            matrix.append({
                                "instrument": inst,
                                "timeframe": tf,
                                "variant": var,
                                "session_normalised": sn,
                                "filters": filters,
                                "K": K,
                            })
    return matrix


def _hash_matrix(matrix: list[dict]) -> str:
    """SHA-256 hash of the serialised test matrix."""
    serialised = json.dumps(matrix, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()


def lock_pre_commitment(
    instruments: list[str],
    timeframes: list[str],
    variants: list[str],
    session_norm_options: list[bool],
    filter_ladders: list[list[str]],
    k_values: list[int],
    output_dir: str | None = None,
) -> str:
    """
    Write the full test matrix + SHA-256 hash to disk.

    Returns the hash string for later verification.
    """
    out = Path(output_dir or CONFIG["output_dir"]) / "preflight"
    out.mkdir(parents=True, exist_ok=True)

    matrix = _build_test_matrix(
        instruments, timeframes, variants,
        session_norm_options, filter_ladders, k_values,
    )
    matrix_hash = _hash_matrix(matrix)

    payload = {
        "hash": matrix_hash,
        "n_tests": len(matrix),
        "config_snapshot": {
            k: v for k, v in CONFIG.items()
            if not callable(v) and k != "cc_param_sweep"
        },
        "test_matrix": matrix,
    }

    path = out / "pre_commitment.json"
    path.write_text(json.dumps(payload, indent=2, default=str))

    logger.info(
        "Pre-commitment lock: %d tests, hash=%s → %s",
        len(matrix), matrix_hash[:16], path,
    )
    return matrix_hash


def verify_pre_commitment(
    instruments: list[str],
    timeframes: list[str],
    variants: list[str],
    session_norm_options: list[bool],
    filter_ladders: list[list[str]],
    k_values: list[int],
    expected_hash: str,
    output_dir: str | None = None,
) -> bool:
    """
    Re-hash the test matrix and compare to the locked hash.

    Returns True if clean, False if contaminated.
    Writes verification result to pipeline_log.json.
    """
    matrix = _build_test_matrix(
        instruments, timeframes, variants,
        session_norm_options, filter_ladders, k_values,
    )
    actual_hash = _hash_matrix(matrix)
    is_clean = actual_hash == expected_hash

    out = Path(output_dir or CONFIG["output_dir"]) / "preflight"
    out.mkdir(parents=True, exist_ok=True)

    status = "CLEAN" if is_clean else "CONTAMINATED"
    result = {
        "verification": status,
        "expected_hash": expected_hash,
        "actual_hash": actual_hash,
    }

    # Append to pipeline_log.json
    log_path = out / "pipeline_log.json"
    log_data = {}
    if log_path.exists():
        log_data = json.loads(log_path.read_text())
    log_data["pre_commitment_verification"] = result
    log_path.write_text(json.dumps(log_data, indent=2))

    if not is_clean:
        logger.error(
            "⚠ PRE-COMMITMENT CONTAMINATED: test matrix was modified mid-run! "
            "Expected %s, got %s", expected_hash[:16], actual_hash[:16],
        )
    else:
        logger.info("Pre-commitment verification: CLEAN (hash match).")

    return is_clean
