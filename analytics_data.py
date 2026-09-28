"""Calculate dashboard values from the current CSV, never from saved images."""
import numpy as np
import pandas as pd


class AnalyticsError(ValueError):
    pass


NUMERIC = ("current_stock", "sales_velocity", "demand_variability",
           "promotion_status", "stockout_within_7_days", "probability")


def read_analytics_data(source):
    try:
        frame = pd.read_csv(source)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise AnalyticsError("Unable to read the CSV. Choose a valid UTF-8 CSV with a header row.") from exc
    if frame.empty:
        raise AnalyticsError("This CSV has no records to analyse.")
    if "current_stock" not in frame:
        raise AnalyticsError("Missing current_stock column. Use processed_inventory_data.csv or a CSV with the same headers.")
    for column in NUMERIC:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all() or (values < 0).any():
            raise AnalyticsError(f"Column {column} must contain finite, non-negative numbers in every row.")
        if column in ("promotion_status", "stockout_within_7_days") and not values.isin([0, 1]).all():
            raise AnalyticsError(f"Column {column} must contain only 0 or 1.")
        if column == "probability" and (values > 1).any():
            raise AnalyticsError("Column probability must contain values between 0 and 1.")
        frame[column] = values
    return frame


def chart(title, values, unit="records", percent=False):
    """Keep exact values alongside proportional bars for accessible charts."""
    maximum = 100 if percent else max([float(v) for v in values.values()] + [1])
    return {"title": title, "unit": unit, "rows": [
        {"label": str(label), "value": float(value),
         "width": float(value) / maximum * 100}
        for label, value in values.items()
    ]}


def histogram(frame, column, title):
    values = frame[column].to_numpy()
    if values.min() == values.max():
        counts = {f"{values[0]:g}": len(values)}
    else:
        frequencies, edges = np.histogram(values, bins=10)
        counts = {
            f"{edges[i]:.6g} to {edges[i + 1]:.6g}" + (" (inclusive)" if i == 9 else " (upper excluded)"): int(count)
            for i, count in enumerate(frequencies)
        }
    return chart(title, counts)


def summarize(frame, thresholds=None):
    thresholds = thresholds or {"medium": 0.35, "high": 0.65}
    stats = [{"label": "Records analysed", "value": f"{len(frame):,}"},
             {"label": "Average current stock", "value": f"{frame.current_stock.mean():,.2f}"}]
    charts = [histogram(frame, "current_stock", "Current stock distribution")]
    for column, label in (("sales_velocity", "Sales velocity"), ("demand_variability", "Demand variability")):
        if column in frame:
            stats.append({"label": f"Average {label.lower()}", "value": f"{frame[column].mean():,.2f}"})
            charts.append(histogram(frame, column, f"{label} distribution"))
    notes = []
    if "stockout_within_7_days" in frame:
        target = frame["stockout_within_7_days"]
        stats.extend([
            {"label": "Derived stockout-positive records", "value": f"{int(target.sum()):,}"},
            {"label": "Derived stockout rate", "value": f"{target.mean() * 100:.2f}%"},
        ])
        charts.append(chart("Derived stockout label distribution", {
            "No derived stockout (0)": int((target == 0).sum()),
            "Derived stockout (1)": int((target == 1).sum()),
        }))
        for column, label in (("category", "category"), ("region", "region"), ("promotion_status", "promotion status")):
            if column in frame:
                groups = frame[column].fillna("Unknown").replace("", "Unknown")
                if column == "promotion_status":
                    groups = groups.map({0: "No promotion", 1: "Promotion"})
                rates = target.groupby(groups).mean().mul(100).sort_values(ascending=False)
                charts.append(chart(f"Derived stockout rate by {label}", rates.to_dict(), "% of group records", percent=True))
        notes.append("Stockout rates describe the CSV's derived seven-day labels, not model probabilities or directly recorded stockout events. Positive records can have overlapping future windows.")
    else:
        notes.append("This data has no stockout_within_7_days labels, so an observed/derived stockout rate cannot be calculated.")
    if "probability" in frame:
        probability = frame["probability"]
        stats.append({"label": "Average predicted probability", "value": f"{probability.mean() * 100:.2f}%"})
        medium, high = float(thresholds["medium"]), float(thresholds["high"])
        charts.append(chart("Predicted risk bands", {
            "Low risk": int((probability < medium).sum()),
            "Medium risk": int(((probability >= medium) & (probability < high)).sum()),
            "High risk": int((probability >= high).sum()),
        }))
        notes.append(f"Predicted risk uses the saved model thresholds: medium from {medium:.0%}, high from {high:.0%}. Predictions are estimates, not confirmed stockouts.")
    return {"stats": stats, "charts": charts, "notes": notes}
