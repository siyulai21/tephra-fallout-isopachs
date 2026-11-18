"""Regional isopachs and probabilistic whole-Earth fallout from accepted data."""
import argparse
from pathlib import Path
import sys

from .core.catalog import resolve_dataset
from .models.registry import REGISTRY
from .reporting import compare_dataset


def write_global_index(output):
    """Keep the shared global entry point separate from either fitted family."""
    folder = Path(output)/"global"
    folder.mkdir(parents=True, exist_ok=True)
    (folder/"README.md").write_text("""# Whole-deposit fallout: two independent estimates

[Eruption home](../../README.md) · [Regional comparison](../README.md) · [Method relationships](../../../../methods/README.md)

Both families use this case's accepted observations and report conditional whole-Earth **bulk fallout**. Neither is a preferred answer; their volumes and intervals remain separate.

| Family | Report and evidence |
|---|---|
| Directional continuation | [Field, measured isopach fragments, 13 policies, compass budgets and holdouts](../directional/README.md) |
| Spherical mixture | [Field, posterior volume, prior sensitivities and matched geographic holdouts](../spherical-mixture/README.md) |

[Matched evidence across eruptions](../../../../comparisons/global/README.md) explains their different extrapolation assumptions and validation limitations. A narrower interval or smaller total does not establish greater global accuracy. Read each report's numerical status, missing-fan assumptions and sensitivity.

The [regional support report](../support/README.md) diagnoses local evidence. Directional complete-contour decisions are in its case report; the matched study retains the spherical-mixture support audit. A failed support screen does not suppress a conditional global calculation.

This shared index occupies the old spherical-only `global/` location. Its scientific outputs now live in `spherical-mixture/`; original execution paths and hashes remain in the provenance records.
""")


def compare(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Catalogued eruption or a CSV of observations and source coordinates")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--name", help="Report title for a custom CSV")
    parser.add_argument("--regional-only", action="store_true", help="Run the GP/spline regional comparison and local support audit only")
    parser.add_argument("--global-geometry", choices=("displaced", "source-fans", "source-envelope"), help="Override the catalogued spherical-mixture geometry; source-envelope remains experimental")
    args = parser.parse_args(argv)
    compare_dataset(resolve_dataset(args.csv), args.output, name=args.name,
                    progress=lambda message: print(message, flush=True))
    if not args.regional_only:
        from .global_reporting import global_dataset
        global_dataset(resolve_dataset(args.csv), args.output/"spherical-mixture", name=args.name,
                       geometry=args.global_geometry,
                       progress=lambda message: print(message, flush=True))
        from .models.directional.reporting import directional_dataset
        directional_dataset(resolve_dataset(args.csv), args.output/"directional", name=args.name)
        write_global_index(args.output)
        with (args.output/"README.md").open("a") as stream:
            stream.write("\n## Whole-deposit fallout\n\nCompare the [directional estimate from supported isopach fragments](directional/README.md) with the [spherical-mixture estimate](spherical-mixture/README.md). Each reports a whole-Earth bulk volume under its stated assumptions.\n")


def global_analysis(argv=None):
    parser = argparse.ArgumentParser(description="Whole-Earth fallout posterior, assumption sensitivities and geographic validation")
    parser.add_argument("csv", help="Catalogued eruption or observation CSV")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--name")
    parser.add_argument("--resume", action="store_true", help="Resume only checksum-verified matching posterior caches")
    parser.add_argument("--geometry", choices=("directional", "displaced", "source-fans", "source-envelope"), help="Use directional for the common isopach continuation or select a spherical-mixture geometry")
    args = parser.parse_args(argv)
    if args.geometry == "directional":
        from .models.directional.reporting import directional_dataset
        return directional_dataset(resolve_dataset(args.csv), args.output, args.name, args.resume)
    from .global_reporting import global_dataset
    global_dataset(resolve_dataset(args.csv), args.output, name=args.name, resume=args.resume,
                   geometry=args.geometry,
                   progress=lambda message: print(message, flush=True))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "compare":
        return compare(argv[1:])
    if argv and argv[0] == "global":
        return global_analysis(argv[1:])
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
    subparsers.add_parser("global", help="Global fallout posterior with explicit assumption sensitivity")
    subparsers.add_parser("methods", help="List the active empirical model families")
    args = parser.parse_args(argv)
    if args.command == "methods":
        for key, spec in REGISTRY.items():
            print(f"{key}: {spec.label}\n  {spec.description}")
        print("spherical-mixture (global command): Mass-normalized spherical mixture\n  Whole-Earth posterior fallout volume; fixed prior sensitivities and matched geographic holdouts.")
        print("directional: Whole-Earth continuation from locally supported isopachs\n  Source-centred thinning, alternative missing fans and directional volume budgets.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
