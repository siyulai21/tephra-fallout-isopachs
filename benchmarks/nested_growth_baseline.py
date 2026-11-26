"""Apply the same field and mapped-quartile checks to the retained predecessor."""
import argparse
import json
from pathlib import Path

from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.nested.footprints import diagnostics, map_convergence
from src.empirical_isopach.models.nested.footprints.posterior import FootprintField
from src.empirical_isopach.models.nested.study import digest, write_json


def check(output, baseline):
    output.mkdir(parents=True, exist_ok=True)
    for record in datasets():
        packet = baseline/record['slug']/'footprints/full-selected.npz'
        data, meta = audited_data(record['path'])
        identity = dict(input_sha256=meta['input_sha256'], posterior_sha256=digest(packet),
                        field_diagnostic_sha256=digest(diagnostics.__file__),
                        quantile_diagnostic_sha256=digest(map_convergence.__file__))
        target = output/(record['slug']+'.json')
        if target.exists():
            assert json.loads(target.read_text())['identity'] == identity
            continue
        model = FootprintField.load(packet)
        spatial = diagnostics.population_field_check(model, data)
        quantiles = map_convergence.map_draws(model)
        write_json(target, dict(identity=identity, field_population_check=spatial,
                                mapped_quantile_checks=quantiles))
        print(record['slug'], 'field', spatial['passed'], 'quartiles', quantiles[-1]['passed'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=Path('archive/2026-11-25-nested-growth/previous'))
    args = parser.parse_args()
    check(args.output, args.baseline)
