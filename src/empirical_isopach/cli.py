"""Regional isopachs and probabilistic whole-Earth fallout from accepted data."""
import argparse
from pathlib import Path
import sys

from .core.catalog import resolve_dataset
from .models.registry import REGISTRY
from .models.nested.report import nested_dataset
from .models.nested.reach_report import reach_dataset
from .reporting import compare_dataset
from .visualization.summary import saved_case_overview
from .visualization.comparison import add_comparison_to_run


def compare(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Catalogued eruption or a CSV of observations and source coordinates")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--name", help="Report title for a custom CSV")
    parser.add_argument("--regional-only", action="store_true", help="Run the GP/spline regional comparison and local support audit only")
    parser.add_argument("--global-geometry", choices=("displaced", "source-fans", "source-envelope"), help="Override the catalogued spherical-mixture geometry; source-envelope remains experimental")
    parser.add_argument("--archives", type=Path, help="Audited archive inspections and named visibility sensitivities")
    args = parser.parse_args(argv)
    if args.regional_only and args.archives:
        parser.error("Archive continuation requires a full comparison or directional run")
    compare_dataset(resolve_dataset(args.csv), args.output, name=args.name,
                    progress=lambda message: print(message, flush=True))
    if not args.regional_only:
        from .global_reporting import global_dataset
        global_dataset(resolve_dataset(args.csv), args.output/"spherical-mixture", name=args.name,
                       geometry=args.global_geometry, archives=args.archives,
                       progress=lambda message: print(message, flush=True))
        from .models.directional.reporting import directional_dataset
        directional_dataset(resolve_dataset(args.csv), args.output/"directional", name=args.name, archives=args.archives)
        nested_dataset(resolve_dataset(args.csv), args.output/"nested-isopachs", name=args.name, archives=args.archives)
        reach_dataset(resolve_dataset(args.csv), args.output/"nested-reach", name=args.name,
                      parent=args.output/"nested-isopachs", archives=args.archives)
        review_continuations(resolve_dataset(args.csv), args.output)
        saved_case_overview(args.output, args.name)
        add_comparison_to_run(args.output)


def review_continuations(path, root):
    from .models.continuation_review import directional_review, spherical_review
    spherical = root/"spherical-mixture"
    directional = root/"directional"
    spherical_review(path, spherical, spherical/"sensitivity-study")
    directional_review(path, directional, directional/"sensitivity-study", directional/"work"/"ensemble")


def global_analysis(argv=None):
    parser = argparse.ArgumentParser(description="Whole-Earth fallout posterior, assumption sensitivities and geographic validation")
    parser.add_argument("csv", help="Catalogued eruption or observation CSV")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--name")
    parser.add_argument("--resume", action="store_true", help="Resume only checksum-verified matching posterior caches")
    parser.add_argument("--geometry", choices=("directional", "nested-isopachs", "nested-reach", "displaced", "source-fans", "source-envelope"), help="Select a complete-deposit method or a spherical mixture geometry")
    parser.add_argument("--archives", type=Path, help="Audited archive inventory for conditional global predictions")
    args = parser.parse_args(argv)
    if args.geometry == "nested-reach":
        return reach_dataset(resolve_dataset(args.csv), args.output, name=args.name, resume=args.resume, archives=args.archives)
    if args.geometry == "nested-isopachs":
        return nested_dataset(resolve_dataset(args.csv), args.output, name=args.name, resume=args.resume, archives=args.archives)
    if args.geometry == "directional":
        from .models.directional.reporting import directional_dataset
        return directional_dataset(resolve_dataset(args.csv), args.output, args.name, args.resume, archives=args.archives)
    from .global_reporting import global_dataset
    global_dataset(resolve_dataset(args.csv), args.output, name=args.name, resume=args.resume,
                   geometry=args.geometry, archives=args.archives,
                   progress=lambda message: print(message, flush=True))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "compare":
        return compare(argv[1:])
    if argv and argv[0] == "global":
        global_analysis(argv[1:])
        return
    if argv and argv[0] not in ("fit", "methods", "-h", "--help"):
        # Preserve the module's original CSV-first regional comparison command.
        return compare(argv)
    parser = argparse.ArgumentParser(description=__doc__,
        epilog="Run tephra-isopachs compare observations.csv --output results/new-run to fit all five families.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("compare", help="Compare every active method; no preferred family")
    subparsers.add_parser("global", help="Global fallout posterior with explicit assumption sensitivity")
    subparsers.add_parser("methods", help="List the active empirical model families")
    args = parser.parse_args(argv)
    if args.command == "methods":
        for key, spec in REGISTRY.items():
            print(f"{key}: {spec.label}\n  {spec.description}")
        print("spherical-mixture (global command): Mass-normalized spherical mixture\n  Whole-Earth posterior fallout volume with bearing-dependent continuation, fixed prior sensitivities and matched geographic holdouts.")
        print("directional: Whole-Earth continuation from locally supported isopachs\n  Uncertain fan direction across six structures. The principal bulk volume integrates the pointwise median isopachs.")
        print("nested-reach: Paired geographic reach check for Bayesian nested isopachs")
        print("nested-isopachs: Joint Bayesian isopach areas and shapes\n  A complete-deposit volume posterior, externally informed thinning, and buffered geographic validation.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
