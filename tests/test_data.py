import pandas as pd
from src.data.loader import clean_dataset

def test_clean_dataset_removes_duplicate_and_invalid_targets():
    frame = pd.DataFrame({
        "distance": [2.0, 2.0, 3.0, 4.0],
        "Time_taken_min": [20, 20, None, -2],
    })
    cleaned, report = clean_dataset(frame, "Time_taken_min")
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["distance"] == 2.0
    assert report["rows_before"] == 4
    assert report["rows_after"] == 1
