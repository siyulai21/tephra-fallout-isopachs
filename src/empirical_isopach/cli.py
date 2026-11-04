"""Compare all active empirical methods on one observation hull."""
import argparse
from pathlib import Path
import sys

from .core.catalog import resolve_dataset
from .models.registry import REGISTRY
from .reporting import compare_dataset


def compare(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Catalogued eruption or a CSV of observations and source coordinates")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--name", help="Report title for a custom CSV")
    args = parser.parse_args(argv)
    compare_dataset(resolve_dataset(args.csv), args.output, name=args.name,
                    progress=lambda message: print(message, flush=True))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "compare":
        return compare(argv[1:])
    if argv and argv[0] == "fit":
        from src.probabilistic_isopach.cli import main as historical_fit
        return historical_fit(argv)
    if argv and argv[0] not in ("methods", "-h", "--help"):
        # Preserve the module's original CSV-first regional comparison command.
        return compare(argv)
    parser = argparse.ArgumentParser(description=__doc__,
        epilog="New data: tephra-isopachs compare observations.csv --output results/new-run. "
               "The legacy GP-only command remains available as 'fit'.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("compare", help="Compare every active method; no preferred family")
    subparsers.add_parser("methods", help="List the active empirical model families")
    args = parser.parse_args(argv)
    if args.command == "methods":
        for key, spec in REGISTRY.items():
            print(f"{key}: {spec.label}\n  {spec.description}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
