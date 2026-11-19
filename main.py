"""GTA VI PlayStation 5 Trophy Earn Rate Quantile Forecaster.

Forecasts the 5th, 25th, 50th, 75th, and 95th percentiles of earn rates for
all PlayStation 5 trophies of Grand Theft Auto VI on Exophase at target timestamps.

Uses adaptive Bayesian rate tracking, player influx calibration, and
growth-damped logit progression models.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import sys
from pathlib import Path

# Ensure local modules can be loaded under grader's -P sandbox
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS: list[float] = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES: list[float] = [-1.645, -0.674, 0.0, 0.674, 1.645]


def clamp_rate(val: float) -> float:
    """Clamps an earn rate percentage to the valid open interval (0.01, 99.99)."""
    return min(99.99, max(0.01, val))


def to_logit(rate_pct: float) -> float:
    """Transforms rate percentage to log-odds scale."""
    clamped = clamp_rate(rate_pct)
    prob = clamped / 100.0
    return math.log(prob / (1.0 - prob))


def from_logit(logit_val: float) -> float:
    """Inverts log-odds scale back to percentage in [0, 100]."""
    return 100.0 / (1.0 + math.exp(-logit_val))


def round4(val: float) -> float | int:
    """Rounds to 4 decimal places half-up, matching JavaScript Math.round."""
    shifted = val * 10000.0
    base = math.floor(shifted)
    if shifted - base >= 0.5:
        base += 1.0
    rounded = base / 10000.0
    return int(rounded) if rounded.is_integer() else rounded


def parse_timestamp(iso_str: str) -> float:
    """Parses ISO-8601 UTC timestamp to epoch seconds."""
    clean_str = iso_str.replace("Z", "+00:00")
    dt = datetime.datetime.fromisoformat(clean_str)
    return dt.timestamp()


def get_trophy_prior_logit(trophy: dict) -> float:
    """Returns domain-informed prior logit for a trophy based on tier and type."""
    trophy_id = trophy.get("id")
    tier = trophy.get("tier", "bronze")
    points = trophy.get("points", 15)
    is_secret = trophy.get("secret", False)
    desc = trophy.get("description", "").lower()

    # Prologue / tutorial mission (100% completion by active players)
    if trophy_id == 2 or "gone fishing" in desc or "prologue" in desc:
        return to_logit(99.2)

    # Ultimate platinum trophy
    if tier == "platinum" or trophy_id == 1:
        return to_logit(0.08)

    # 100% game completion
    if "100% completion" in desc:
        return to_logit(0.12)

    # Story chapter progression
    if "chapter 1" in desc:
        return to_logit(52.0)
    if "chapter 2" in desc:
        return to_logit(32.0)
    if "chapter 3" in desc:
        return to_logit(20.0)
    if "chapter 4" in desc:
        return to_logit(12.0)
    if "final mission" in desc:
        return to_logit(6.5)

    # Long-term grinds and high collectible counts
    if "250 stunts" in desc:
        return to_logit(1.2)
    if "100 locations" in desc or "shooting range" in desc or "collectible" in desc:
        return to_logit(1.5)

    # Tier-based baselines
    if tier == "gold":
        return to_logit(2.5)
    if tier == "silver":
        return to_logit(4.2)
    if points <= 15 and not is_secret:
        return to_logit(14.0)

    return to_logit(9.0)


def forecast_single_case(case: dict, filename: str) -> dict:
    """Produces 5-quantile forecasts for each trophy in the given case."""
    if case.get("levels") != LEVELS:
        raise ValueError(f"{filename} requests levels {case.get('levels')}, expected {LEVELS}")

    history = case.get("history", [])
    if not history:
        raise ValueError(f"{filename} contains no history reads")

    trophies = case.get("trophies", [])
    if not trophies:
        raise ValueError(f"{filename} contains no trophies")

    horizon_minutes = float(case.get("horizon_minutes", 90.0))
    horizon_days = horizon_minutes / 1440.0
    horizon_hours = horizon_minutes / 60.0

    last_entry = history[-1]
    last_rates = last_entry["rates"]
    last_tracked = float(last_entry.get("players_tracked", 638))

    player_counts = [float(h.get("players_tracked", 0)) for h in history]
    min_tracked = min(player_counts)
    max_tracked = max(player_counts)

    # Detect if we are in early flat launch night before first Exophase refresh
    is_flat_launch = (max_tracked / max(1.0, min_tracked) < 1.05) and (last_tracked < 2500)

    # Base timestamp for regression
    t_zero = parse_timestamp(history[0]["at"])

    output_rows = []
    for idx, trophy in enumerate(trophies):
        raw_rate = last_rates[idx]
        current_rate = clamp_rate(raw_rate if raw_rate is not None else 0.01)
        current_logit = to_logit(current_rate)
        prior_logit = get_trophy_prior_logit(trophy)

        if is_flat_launch:
            # Launch-night pre-refresh state (e.g. practice cases p00/p01)
            # A batch refresh will expand player denominator from ~600 to ~3500+
            if current_rate >= 98.0:
                # Prologue trophy remains earned by virtually all new players
                median_logit = to_logit(99.5)
                spread = 0.35
            else:
                desc = trophy.get("description", "").lower()
                tier = trophy.get("tier", "bronze")
                points = trophy.get("points", 15)
                is_secret = trophy.get("secret", False)

                # Hard/late game trophies experience full denominator dilution
                if tier == "platinum" or "final mission" in desc or "100% completion" in desc or "chapter 4" in desc:
                    dilution_factor = 5.8
                elif "chapter 1" in desc or (points <= 15 and not is_secret):
                    dilution_factor = 3.2
                else:
                    dilution_factor = 4.8

                refreshed_rate = max(0.04, current_rate / dilution_factor)
                refreshed_logit = to_logit(refreshed_rate)

                # Mixture median accounting for refresh timing uncertainty
                prob_refresh = 0.85
                median_logit = prob_refresh * refreshed_logit + (1.0 - prob_refresh) * current_logit
                spread = 0.88
        else:
            # Active observation window with observed player growth / rate dynamics
            recent_reads = history[-24:] if len(history) >= 24 else history
            times: list[float] = []
            logits: list[float] = []

            for h in recent_reads:
                r_val = h["rates"][idx]
                if r_val is not None:
                    t_hours = (parse_timestamp(h["at"]) - t_zero) / 3600.0
                    times.append(t_hours)
                    logits.append(to_logit(clamp_rate(r_val)))

            velocity = 0.0
            sigma_resid = 0.08
            if len(times) >= 2 and times[-1] > times[0]:
                mean_t = sum(times) / len(times)
                mean_l = sum(logits) / len(logits)
                var_t = sum((t - mean_t) ** 2 for t in times)
                if var_t > 1e-6:
                    cov_tl = sum((t - mean_t) * (l - mean_l) for t, l in zip(times, logits))
                    velocity = cov_tl / var_t

                # Residual standard error of linear trend
                n_pts = len(times)
                sum_sq_err = sum((l - (mean_l + velocity * (t - mean_t))) ** 2 for t, l in zip(times, logits))
                sigma_resid = math.sqrt(sum_sq_err / max(1, n_pts - 2))

            # Exponential velocity damping with 36-hour half life
            decay_rate = math.log(2.0) / 36.0
            effective_dt = (1.0 - math.exp(-decay_rate * horizon_hours)) / decay_rate
            trend_logit = current_logit + velocity * effective_dt

            # Regularize towards domain-specific week 1 prior
            weight_prior = 0.25
            median_logit = (1.0 - weight_prior) * trend_logit + weight_prior * prior_logit

            # Adaptive spread scaled by square-root of horizon days and residual volatility
            spread = max(0.40, 0.52 * math.sqrt(max(0.05, horizon_days)) + 0.30 * sigma_resid)

        # Generate quantiles and ensure finite bounds in [0.0, 100.0]
        raw_quantiles = [from_logit(median_logit + z * spread) for z in Z_SCORES]
        quantiles = [round4(max(0.0, min(100.0, q))) for q in raw_quantiles]

        # Ensure strict monotonicity across rounding
        for k in range(1, 5):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]

        output_rows.append({"id": trophy["id"], "q": quantiles})

    return {"trophies": output_rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    in_files = sorted(INPUT_DIR.glob("*.in"))
    for file_path in in_files:
        content = json.loads(file_path.read_text(encoding="utf-8"))
        forecast = forecast_single_case(content, file_path.name)
        out_filename = f"{file_path.stem}.out"
        out_path = OUTPUT_DIR / out_filename
        out_bytes = json.dumps(forecast, separators=(",", ":")) + "\n"
        out_path.write_text(out_bytes, encoding="utf-8")


if __name__ == "__main__":
    main()
