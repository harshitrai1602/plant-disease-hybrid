import pandas as pd
from pathlib import Path

csv_path = Path("outputs/baseline_cnn_full/training_history.csv")

if not csv_path.exists():
    print(f"Results file not found: {csv_path.resolve()}")
else:
    df = pd.read_csv(csv_path)

    print("\n========== BASELINE CNN EPOCH RESULTS ==========\n")
    print(df.to_string(index=False))

    print("\n========== FIRST 10 EPOCHS ==========\n")
    print(df.head(10).to_string(index=False))

