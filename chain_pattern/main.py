import sys
import pandas as pd
from pipeline import build_pipeline


def main():
    if len(sys.argv) != 2:
        raise RuntimeError("Usage: python main.py hh.csv")

    csv_path = sys.argv[1]

    pipeline = build_pipeline(csv_path)
    pipeline.handle(pd.DataFrame())


if __name__ == "__main__":
    main()
