"""Compare two peer empirical methods; no eruption-specific tuning required."""
import argparse
from pathlib import Path

from src.probabilistic_isopach.catalog import resolve_dataset
from .reporting import compare_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Catalogued eruption or a CSV of observations and source coordinates")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--name", help="Report title for a custom CSV")
    args = parser.parse_args(argv)
    compare_dataset(resolve_dataset(args.csv), args.output, name=args.name,
                    progress=lambda message: print(message, flush=True))


if __name__ == "__main__":
    main()
