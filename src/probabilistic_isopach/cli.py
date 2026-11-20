"""Forward explicit workflow commands from the earlier module path."""
import argparse
import sys

from src.empirical_isopach.cli import main as current_main


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] not in {"compare", "global", "methods", "-h", "--help"}:
        argparse.ArgumentParser(prog="tephra-isopachs").error(
            "Use 'tephra-isopachs compare ERUPTION --output DIR' for the current workflow. "
            "The historical GP-only fit is available in the release history."
        )
    return current_main(args)
