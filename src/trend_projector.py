"""
src/trend_projector.py
----------------------
Multi-period churn risk trend projector.

Projects a customer's churn probability trajectory over future periods
under a baseline and an optional retention intervention scenario.

Functions:
    project_churn_trend(base_probability, periods, decay_rate, growth_rate) -> list[dict]
    get_trend_summary(projections) -> dict
    project_retention_scenario(base_probability, periods, intervention_decay) -> list[dict]
"""

from typing import List
from src.config import setup_logger

logger = setup_logger("trend_projector")

MAX_PROBABILITY = 0.98
MIN_PROBABILITY = 0.02


def _clamp(value: float) -> float:
    return max(MIN_PROBABILITY, min(MAX_PROBABILITY, value))


def project_churn_trend(
    base_probability: float,
    periods: int = 12,
    decay_rate: float = 0.0,
    growth_rate: float = 0.0,
) -> List[dict]:
    """
    Projects churn probability trajectory over future periods.

    At each period the probability is adjusted by:
        p_next = clamp(p_current * (1 + growth_rate) * (1 - decay_rate))

    Parameters
    ----------
    base_probability : float
        Starting churn probability in [0.0, 1.0].
    periods : int
        Number of future periods to project (default: 12).
    decay_rate : float
        Per-period fractional reduction (0.05 = 5% decrease per period).
        Represents the effect of successful retention actions.
    growth_rate : float
        Per-period fractional increase (0.03 = 3% increase per period).
        Represents natural churn risk escalation without intervention.

    Returns
    -------
    list of dicts: [{ "period", "projected_probability", "trend_direction" }]
    """
    if periods < 1:
        raise ValueError("periods must be >= 1")

    base_probability = _clamp(float(base_probability))
    projections = []
    current = base_probability

    for t in range(1, periods + 1):
        prev = current
        current = _clamp(current * (1 + growth_rate) * (1 - decay_rate))

        delta = current - prev
        if delta > 0.005:
            direction = "Worsening"
        elif delta < -0.005:
            direction = "Improving"
        else:
            direction = "Stable"

        projections.append({
            "period": t,
            "projected_probability": round(current, 4),
            "trend_direction": direction,
        })

    logger.info(
        f"Projected {periods} periods from base={base_probability:.3f} "
        f"(decay={decay_rate}, growth={growth_rate}). "
        f"Final={projections[-1]['projected_probability']}"
    )
    return projections


def get_trend_summary(projections: List[dict]) -> dict:
    """
    Summarises a projection trajectory.

    Parameters
    ----------
    projections : list[dict]
        Output of project_churn_trend().

    Returns
    -------
    dict with keys:
        start_probability, end_probability, peak_probability,
        trough_probability, overall_direction, total_change
    """
    if not projections:
        raise ValueError("projections list is empty.")

    probs = [p["projected_probability"] for p in projections]
    start = probs[0]
    end = probs[-1]
    peak = max(probs)
    trough = min(probs)
    total_change = round(end - start, 4)

    if total_change > 0.02:
        overall_direction = "Worsening"
    elif total_change < -0.02:
        overall_direction = "Improving"
    else:
        overall_direction = "Stable"

    return {
        "start_probability": start,
        "end_probability": end,
        "peak_probability": peak,
        "trough_probability": trough,
        "overall_direction": overall_direction,
        "total_change": total_change,
        "num_periods": len(projections),
    }


def project_retention_scenario(
    base_probability: float,
    periods: int = 12,
    intervention_decay: float = 0.05,
) -> List[dict]:
    """
    Projects the baseline (no action) and retention intervention trajectories.

    Returns
    -------
    list of dicts: [{
        "period",
        "baseline_probability",
        "intervention_probability",
        "probability_saved"
    }]
    """
    baseline = project_churn_trend(base_probability, periods, decay_rate=0.0, growth_rate=0.02)
    retained = project_churn_trend(base_probability, periods, decay_rate=intervention_decay, growth_rate=0.0)

    scenarios = []
    for b, r in zip(baseline, retained):
        scenarios.append({
            "period": b["period"],
            "baseline_probability": b["projected_probability"],
            "intervention_probability": r["projected_probability"],
            "probability_saved": round(b["projected_probability"] - r["projected_probability"], 4),
        })

    return scenarios
