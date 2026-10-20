
from pathlib import Path

import pandas as pd


def main():
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs"

    majority_path = output_dir / "baseline_results.csv"
    random_forest_path = output_dir / "rgb_baseline_results.csv"
    combined_path = output_dir / "combined_baseline_results.csv"

    majority = pd.read_csv(majority_path)
    random_forest = pd.read_csv(random_forest_path)

    combined = pd.concat(
        [majority, random_forest],
        ignore_index=True,
    )

    combined.to_csv(combined_path, index=False)

    print(combined.to_string(index=False))
    print(f"\nCombined results saved to: {combined_path}")


if __name__ == "__main__":
    main()
