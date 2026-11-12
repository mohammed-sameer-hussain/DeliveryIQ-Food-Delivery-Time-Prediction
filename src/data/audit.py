import json
import pandas as pd

from src.data.loader import ROOT, load_config, load_dataset, clean_dataset


def run_audit():
    config = load_config()
    target = config["target_column"]

    raw = load_dataset()
    data, quality = clean_dataset(raw, target)

    leakage = {
        str(c).strip().lower()
        for c in config.get("leakage_columns", [])
    }
    leakage.add(target.strip().lower())

    suspicious_parts = [
        "average_speed",
        "avg_speed",
        "time_taken",
        "delivery_duration",
        "actual_delivery",
        "delivered_at",
    ]

    suspicious = [
        str(c) for c in data.columns
        if any(part in str(c).lower() for part in suspicious_parts)
    ]

    correlation_flags = []
    numeric = data.select_dtypes(include="number")

    for column in numeric.columns:
        if column == target:
            continue

        if str(column).strip().lower() in leakage:
            continue

        correlation = numeric[column].corr(numeric[target])

        if pd.notna(correlation) and abs(correlation) >= 0.90:
            correlation_flags.append({
                "feature": str(column),
                "correlation_with_target": float(correlation),
                "requires_manual_review": True,
            })

    feature_columns = [
        c for c in data.columns
        if str(c).strip().lower() not in leakage
    ]

    report = {
        "target": target,
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "data_quality": quality,
        "known_leakage_columns": [
            str(c) for c in data.columns
            if str(c).strip().lower() in leakage
        ],
        "suspicious_feature_names": suspicious,
        "high_correlation_review": correlation_flags,
        "duplicate_feature_rows": int(
            data.duplicated(subset=feature_columns).sum()
        ),
        "warning": (
            "High correlation does not prove leakage. Check whether "
            "each feature is actually available when the ETA is requested."
        ),
    }

    output = ROOT / "artifacts" / "reports" / "leakage_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print(json.dumps(report, indent=4))
    print("\nAudit report saved:", output)


if __name__ == "__main__":
    run_audit()
