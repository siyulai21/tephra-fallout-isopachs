"""Verify or rebuild the YTT inputs from the original workbook and reviewed site decisions."""
import argparse

from .ytt_inputs import prepare_inputs, verify_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Rebuild the reviewed input tables and provenance.")
    args = parser.parse_args(argv)
    if args.prepare_only:
        prepare_inputs()
    result = verify_inputs()
    print(f"Verified {result['audit_records']} YTT source rows and {len(result['cases'])} reviewed selections.")


if __name__ == "__main__":
    main()
