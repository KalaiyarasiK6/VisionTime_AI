"""
timeline.py
-----------
Converts EventEngine output into a structured timeline (pandas DataFrame),
handles CSV export, and computes summary statistics for the dashboard.
"""

from typing import List

import pandas as pd

from activity import Event

EVENT_LABELS = {
    "entered_scene": "Entered Scene",
    "exited_scene": "Exited Scene",
    "zone_change": "Zone Change",
    "stayed": "Stayed",
    "long_stay": "Long Stay",
    "loitering": "Loitering",
    "fast_movement": "Fast Movement",
    "crowd_detected": "Crowd Detected",
}

# Event types considered "notable" enough for the Important Events panel
IMPORTANT_TYPES = {"Long Stay", "Loitering", "Crowd Detected", "Fast Movement"}

# Risk contribution of each event type (0 = benign, higher = more anomalous).
# Used to compute a session-level "Risk Score" for the AI Summary / dashboard gauge.
RISK_WEIGHTS = {
    "Entered Scene": 0,
    "Exited Scene": 0,
    "Zone Change": 0,
    "Stayed": 3,
    "Long Stay": 15,
    "Loitering": 25,
    "Fast Movement": 18,
    "Crowd Detected": 20,
}


def events_to_dataframe(events: List[Event]) -> pd.DataFrame:
    rows = []
    for e in events:
        event_type = EVENT_LABELS.get(e.event_type, e.event_type)
        rows.append({
            "Time": e.timestamp_str(),
            "Seconds": round(e.timestamp, 2),
            "Person": f"Person {e.track_id}" if e.track_id is not None else "-",
            "Event Type": event_type,
            "Zone": e.zone or "-",
            "Risk Score": RISK_WEIGHTS.get(event_type, 0),
            "Description": e.description,
        })
    df = pd.DataFrame(
        rows,
        columns=["Time", "Seconds", "Person", "Event Type", "Zone", "Risk Score", "Description"],
    )
    if not df.empty:
        df = df.sort_values("Seconds").reset_index(drop=True)
    return df


def export_csv(df: pd.DataFrame, path: str) -> str:
    df.to_csv(path, index=False)
    return path


def important_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["Event Type"].isin(IMPORTANT_TYPES)].reset_index(drop=True)


def compute_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "total_persons": 0,
            "total_events": 0,
            "duration_seconds": 0,
            "zone_counts": {},
            "event_type_counts": {},
            "longest_dwell": None,
            "risk_score": 0,
            "risk_level": "Low",
        }

    persons = df[df["Person"] != "-"]["Person"].unique().tolist()
    duration = df["Seconds"].max()
    zone_counts = df[df["Zone"] != "-"]["Zone"].value_counts().to_dict()
    event_type_counts = df["Event Type"].value_counts().to_dict()

    long_stays = df[df["Event Type"].isin(["Long Stay", "Loitering"])]
    longest_dwell = None
    if not long_stays.empty:
        longest_dwell = long_stays.iloc[-1]["Description"]

    # Session risk score: sum of individual event risk contributions, capped at 100.
    raw_risk = int(df["Risk Score"].sum()) if "Risk Score" in df.columns else 0
    risk_score = min(raw_risk, 100)
    if risk_score >= 60:
        risk_level = "High"
    elif risk_score >= 25:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "total_persons": len(persons),
        "total_events": len(df),
        "duration_seconds": duration,
        "zone_counts": zone_counts,
        "event_type_counts": event_type_counts,
        "longest_dwell": longest_dwell,
        "risk_score": risk_score,
        "risk_level": risk_level,
    }
