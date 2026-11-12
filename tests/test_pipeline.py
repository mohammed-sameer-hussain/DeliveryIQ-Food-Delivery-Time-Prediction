import pandas as pd
from src.models.train import build_preprocessor

def test_preprocessor_handles_numeric_and_categorical():
    X = pd.DataFrame({
        "distance": [1.2, 2.4, None, 4.0],
        "traffic": ["Low", "High", "Medium", None],
    })
    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == 4
    assert transformed.shape[1] >= 2
