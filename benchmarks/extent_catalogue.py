"""Compatibility entry point for rebuilding the current reader-facing catalogue."""
import argparse
from pathlib import Path

from .principal_volumes import refresh_catalogue as summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    summary(parser.parse_args().output)
