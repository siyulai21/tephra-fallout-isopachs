"""Omit the unresolved Williams Lake Fen two-horizon value from all families.

This is an additional source test of the frozen conditional reference input.
The omission does not assert that either bed is absent or identify a precursor.
"""
import argparse
import csv
from pathlib import Path

from benchmarks import mazama_sensitivity as shared
from benchmarks.mazama import INPUT, sha, write_csv
from src.empirical_isopach.models.nested.study import write_json


CASE = 'omit-williams'
DEFINITION = ('Omit B2020-113, the compiler 46 cm Williams Lake Fen value. '
              'Johnson et al.1994 PNW-GTR-322 p.6 reports two Mazama horizons '
              'separated by9 cm lake sediment; the relationship of46 cm to '
              'individual beds remains unverified. This test assigns no '
              'replacement thickness, age or precursor identity.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--principal', type=Path)
    p.add_argument('--family', required=True,
                   choices=('regional','directional','spherical-mixture','nested-isopachs'))
    args = p.parse_args()
    folder = args.output/CASE
    folder.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(INPUT.open()) if r['EvidenceID'] != 'B2020-113']
    assert len(rows) == 39
    write_csv(folder/'input.csv', rows)
    write_json(folder/'scenario.json', dict(definition=DEFINITION,
        input_sha256=sha(folder/'input.csv'), principal_input_sha256=sha(INPUT),
        scenario_script_sha256=sha(Path(__file__)), shared_script_sha256=sha(Path(shared.__file__))))
    shared.CASES[CASE] = DEFINITION
    shared.run(args.output, CASE, args.family, args.principal)


if __name__ == '__main__':
    main()
