"""
summary.py
----------
Generates a natural-language AI summary and Smart Alerts from timeline stats.
Purely rule-based over the generic event stream, so it works for any
environment without any hardcoded scene knowledge.
"""

from typing import List, Tuple

import pandas as pd

ABNORMAL_TYPES = ["Loitering", "Crowd Detected", "Fast Movement"]


def generate_summary(df: pd.DataFrame, stats: dict) -> str:
    if df.empty:
        return "No persons or events were detected in this video."

    lines = []
    lines.append(
        f"{stats['total_persons']} person(s) detected and "
        f"{stats['total_events']} event(s) generated over "
        f"{int(stats['duration_seconds'])} second(s) of footage."
    )

    zone_counts = stats.get("zone_counts", {})
    if zone_counts:
        busiest_zone = max(zone_counts, key=zone_counts.get)
        lines.append(
            f"{busiest_zone} recorded the highest activity with "
            f"{zone_counts[busiest_zone]} zone-related event(s)."
        )

    if stats.get("longest_dwell"):
        lines.append(stats["longest_dwell"] + ".")

    abnormal_count = sum(
        stats.get("event_type_counts", {}).get(t, 0) for t in ABNORMAL_TYPES
    )
    if abnormal_count == 0:
        lines.append("No abnormal behaviour detected.")
    else:
        lines.append(
            f"{abnormal_count} abnormal or notable event(s) were flagged for "
            f"review (loitering, crowding, or fast movement)."
        )

    if "risk_score" in stats:
        lines.append(
            f"Overall scene risk score: {stats['risk_score']}/100 ({stats['risk_level']})."
        )

    return " ".join(lines)


def generate_alerts(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Returns list of (severity, message). severity in {info, warning, danger}."""
    alerts: List[Tuple[str, str]] = []
    if df.empty:
        return alerts

    for _, row in df[df["Event Type"] == "Loitering"].iterrows():
        alerts.append(("warning", f"{row['Time']} - {row['Description']}"))

    for _, row in df[df["Event Type"] == "Crowd Detected"].iterrows():
        alerts.append(("danger", f"{row['Time']} - {row['Description']}"))

    for _, row in df[df["Event Type"] == "Fast Movement"].iterrows():
        alerts.append(("warning", f"{row['Time']} - {row['Description']}"))

    if not alerts:
        alerts.append(("info", "No abnormal behaviour detected. System operating normally."))

    return alerts
