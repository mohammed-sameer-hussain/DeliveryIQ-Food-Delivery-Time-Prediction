from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.json"

def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

def locate_dataset() -> Path:
    candidates = [
        ROOT / "data" / "raw" / "Dataset.csv",
        ROOT / "data" / "raw" / "cleaned_dataset (1).csv",
        ROOT / "data" / "processed" / "cleaned_dataset.csv",
        ROOT / "data" / "Dataset.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Dataset not found. Copy Dataset.csv into data/raw/Dataset.csv. "
        f"Checked: {', '.join(str(p.relative_to(ROOT)) for p in candidates)}"
    )

def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path else locate_dataset()
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise ValueError("The dataset is empty.")
    return df

def clean_dataset(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, dict]:
    if target_column not in df.columns:
        raise ValueError(
            f"Target '{target_column}' not found. Available columns: {list(df.columns)}"
        )
    original_rows = len(df)
    data = df.copy()
    data = data.drop_duplicates()
    data[target_column] = pd.to_numeric(data[target_column], errors="coerce")
    data = data.dropna(subset=[target_column])
    data = data[data[target_column] > 0]
    report = {
        "rows_before": int(original_rows),
        "rows_after": int(len(data)),
        "duplicates_or_invalid_target_removed": int(original_rows - len(data)),
        "missing_values_by_column": {
            str(k): int(v) for k, v in df.isna().sum().items() if int(v) > 0
        },
    }
    return data, report
