"""Reproduce the source-backed Aso-4 coordinate/bound revision discovered in November 2026."""
import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.aso_evidence import RAW, case_data, dump, INPUT_SHA256
from src.empirical_isopach.models.mass_mixture.inference import Sampling
from src.empirical_isopach.models.mass_mixture.prior import BASELINE, SENSITIVITIES
from src.empirical_isopach.models.mass_mixture.study import audited_data, fit_study, fit_checked


def revised_table(ledger, coordinate_only=False):
    original=pd.read_csv(RAW)
    assert sha256(RAW.read_bytes()).hexdigest() == INPUT_SHA256
    table,_,_=case_data("offset_sd1" if coordinate_only else "s2_bounds",original)
    # Only the two contradicted coordinates change; retain the supplied precision.
    assert table.loc[3,"Lat"] == 30.5 and table.loc[6,"Lon"] == 140.
    table.loc[3,"Lat"] = 34.5
    table.loc[6,"Lon"] = 146.
    for row in ledger.itertuples():
        if pd.notna(row.s2_reference):
            table.loc[row.csv_row-2,"Refs"] = str(row.s2_reference)+"; compiled by Takarada & Hoshizumi (2020), S2 row "+str(int(row.s2_sheet_row))
    return table


def prepare(output,ledger):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    ledger=pd.read_csv(ledger)
    for name,coordinate_only in (("reviewed",False),("coordinate_only",True)):
        path=output/(name+".csv")
        text=revised_table(ledger,coordinate_only).to_csv(index=False)
        if path.exists() and path.read_text() != text:
            raise ValueError("Existing revision differs. Choose a new directory.")
        path.write_text(text)
    protocol={"original_sha256":INPUT_SHA256,"cases":{
        "reviewed":{"sha256":sha256((output/"reviewed.csv").read_bytes()).hexdigest(),
            "definition":"Correct original CSV row 5 latitude to 34.5 N and row 8 longitude to 146 E from Yamamoto & Kanamatsu (2001), p.56 and coring-location maps. Preserve 7 and 6 cm confirmed by Yamamoto & Aoki (2002), Tables 1/2, p.31 and correlations p.32. Encode CSV 43 upper 10 cm; CSV 52 lower 3 cm; CSV 53 lower 6.7 cm following Table S2, retaining the primary-paper 6.5–7 cm conflict; CSV 55 lower 7 cm as a preservation sensitivity. Keep all other quantitative records and the six excluded pseudo rows. Fill S2 source references without changing row order."},
        "coordinate_only":{"sha256":sha256((output/"coordinate_only.csv").read_bytes()).hexdigest(),
            "definition":"Change the same two coordinates only, preserving all 87 original exact likelihood terms; isolates location effects."}},
        "decision_basis":"Source evidence, not fitted volume. S2 vs Danhara range conflict and uncertain flow/fall records remain explicit. No published volume supplies a prior.",
        "timing":"Additional experiment declared after identifying cruise-record coordinate errors during the source audit; the earlier nine-case protocol remains unchanged.",
        "priors":"Reviewed input receives the existing baseline and all seven fixed prior sensitivities, plus a separate offset SD 1 development comparison. Both baseline and SD 1 use identical new geographic folds from revised coordinates. Coordinate-only receives the existing baseline.",
        "runner_sha256":sha256(Path(__file__).read_bytes()).hexdigest()}
    dump(output/"protocol.json",protocol)
    return protocol


def run(output,mode):
    output=Path(output)
    protocol=json.loads((output/"protocol.json").read_text())
    name="coordinate_only" if mode == "coordinate_only" else "reviewed"
    path=output/(name+".csv")
    assert sha256(path.read_bytes()).hexdigest() == protocol["cases"][name]["sha256"]
    if mode == "coordinate_only":
        data,meta=audited_data(path)
        _,fit=fit_checked(data,meta["vent_lonlat"][1],BASELINE,output/mode/"posteriors",Sampling(seed=20261108),
            progress=lambda t:print(t,flush=True))
        dump(output/mode/"report.json",{"fit":fit,"input":{k:v for k,v in meta.items() if k!="transform_inv"}})
    else:
        prior=replace(BASELINE,offset_sd=1.) if mode == "offset_sd1" else BASELINE
        fit_study(path,output/mode,Sampling(seed=20261108),sensitivity=mode=="baseline",validation=True,
            baseline_prior=prior,sensitivity_priors=SENSITIVITIES,progress=lambda t:print(t,flush=True))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--ledger",type=Path)
    parser.add_argument("--modes",nargs="+",choices=("coordinate_only","baseline","offset_sd1"))
    args=parser.parse_args()
    if args.prepare:
        if args.ledger is None: parser.error("Preparation requires --ledger")
        prepare(args.output,args.ledger)
    for mode in args.modes or []: run(args.output,mode)
