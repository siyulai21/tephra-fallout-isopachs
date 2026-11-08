"""Aso-4 source and extrapolation sensitivities, without a target volume.

Prepare the immutable protocol before fitting. Source alternatives and prior
alternatives are separate experiments, not independent probability samples.
"""
import argparse
from dataclasses import replace
from decimal import Decimal, ROUND_HALF_UP
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod

from benchmarks.aso_input_review import INPUT_SHA256
from src.empirical_isopach.models.mass_mixture.prior import BASELINE
from src.empirical_isopach.models.mass_mixture.inference import Sampling
from src.empirical_isopach.models.mass_mixture.study import audited_data, fit_checked, cross_validate


RAW = Path("data/eruptions/Aso-4_tephra.csv")
S2_SHA = "e03288f9a1074687828baf60031ea5884b9ce420903cf841dc0dd4e19d736781"
S2_DOI = "https://doi.org/10.3389/feart.2020.00170.s002"
CASES = {
    "bounds_only": "CSV 43: upper 10 cm; 52: lower 3 cm; 55: lower 7 cm. CSV 53 retains 6.7 cm pending conflicting source interpretations.",
    "s2_bounds": "bounds_only, with CSV 53 lower 6.7 cm as explicitly reported in Takarada & Hoshizumi Table S2.",
    "primary_6p5": "bounds_only, with CSV 53 fixed at the 6.5 cm endpoint of Danhara et al. Table 4's 6.5–7 cm range.",
    "primary_7": "bounds_only, with CSV 53 fixed at the 7 cm endpoint of that range. Endpoints are sensitivities, not a probability distribution or rigorous bounds on total volume.",
    "omit_ambiguous": "bounds_only, omitting CSV 4, 6, 14, 20: Machida Table 1 labels these afa?, pfl or afa, afa?, afa?. Classification sensitivity, not evidence that all four are flows.",
    "omit_row37": "Original likelihood, omit KH76-2-7 (15 cm) to check the largest downward single-row PSIS influence. A valid influential record is not thereby rejected.",
    "omit_row49": "Original likelihood, omit KH82-4-23 (1 cm) to check the largest upward single-row PSIS influence.",
    "offset_sd1": "Original observations and priors except the SD of each source-relative centre displacement/scale coordinate: 1 instead of 3. Allows secondary maxima; no pseudo-observation or hard centre boundary.",
    "offset_sd6": "Original observations and priors except displacement SD 6 instead of 3. Isolates this prior, unlike the older wide_geography case that also changed scale SD.",
}
SOURCE_NOTES = {
    4: "Machida et al. (1985), Table 1, p.52: site 2a, afa? 75 cm; uncertain fallout classification.",
    6: "Machida et al. (1985), Table 1, p.52: site 2d, pfl or afa 60 cm; flow/fall ambiguity.",
    14: "Machida et al. (1985), Table 1, p.52: site 2i, afa? 45 cm; uncertain fallout classification.",
    20: "Machida et al. (1985), Table 1, p.52: site 2b, afa? 35 cm; uncertain fallout classification.",
    37: "Machida et al. (1985), Table 1, p.53: KH76-2-7 interval 395–410 cm below seafloor; 15 cm is its interval thickness, not the burial depth.",
    43: "Machida et al. (1985), Table 1, p.53: site 5b is <10 cm. Table S2 dropped the inequality.",
    49: "Machida et al. (1985), Table 1, p.53: KH82-4-23 interval 847–848 cm below seafloor supports 1 cm.",
    52: "Nagahashi et al. (2007), Table 1, p.308: TKN2003 at 20.03 m is >3.0 cm; Table S2 calls the core TKN-2004. Match through S2 is secure; naming differs.",
    53: "Conflict retained: Takarada & Hoshizumi S2 F31 gives >6.7 cm; Danhara et al. (2010), Table 4, p.108 gives B43-1, depth 66.9 m, thickness 6.5–7 cm. S2 also cites Takemura & Yokoyama (1989), not independently checked here.",
    55: "Shibata et al. (2025), Sections 3.2/4.1, Loc.1 coordinates match CSV: partly eroded layer 4–7 cm. Treating 7 cm as a lower bound assumes erosion did not increase thickness by redeposition.",
}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def case_data(name, original):
    """Operate by stable raw CSV identities, never by canonical array positions."""
    if name not in CASES:
        raise ValueError(f"Unknown declared case: {name}")
    table = original.copy()
    table["Kind"] = "measurement"
    table["Limit_cm"] = np.nan
    if name in ("bounds_only", "s2_bounds", "primary_6p5", "primary_7", "omit_ambiguous"):
        for row, kind, value in ((43,"upper_limit",10.), (52,"lower_limit",3.), (55,"lower_limit",7.)):
            table.loc[row-2, ["Kind","Limit_cm"]] = [kind, value]
    if name == "s2_bounds":
        table.loc[51, ["Kind","Limit_cm"]] = ["lower_limit",6.7]
    if name.startswith("primary_"):
        table.loc[51,"Thickness"] = 6.5 if name == "primary_6p5" else 7.
    excluded = {"omit_ambiguous":[4,6,14,20], "omit_row37":[37], "omit_row49":[49]}.get(name, [])
    prior = replace(BASELINE, offset_sd={"offset_sd1":1.,"offset_sd6":6.}.get(name, 3.))
    return table, excluded, prior


def prepare(output, sources):
    output, sources = Path(output), Path(sources)
    if sha256(RAW.read_bytes()).hexdigest() != INPUT_SHA256:
        raise ValueError("Re-audit after any accepted input change.")
    if sha256((sources/"Takarada2020-TableS2.xlsx").read_bytes()).hexdigest() != S2_SHA:
        raise ValueError("Published source workbook checksum differs.")
    output.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(RAW)
    values = json.loads((sources/"Takarada2020-TableS2-values.json").read_text())["values"]
    source = [(i+1,r) for i,r in enumerate(values) if i >= 3 and isinstance(r[0],int)]
    assert len(source) == 71
    geod = Geod(ellps="WGS84")
    ledger, matched = [], []
    for i,row in original.iloc[:87].iterrows():
        # Match the recorded rounding itself, not an arbitrarily large search circle.
        def rounded(v):
            return float(Decimal(str(v)).quantize(Decimal(".001"),rounding=ROUND_HALF_UP))
        candidates = [(n,r) for n,r in source if rounded(r[1]) == row.Lon and rounded(r[2]) == row.Lat]
        assert len(candidates) <= 1
        entry = {"csv_row":i+2,"latitude":row.Lat,"longitude":row.Lon,"thickness_cm":row.Thickness,
            "supplied_reference": "" if pd.isna(row.Refs) else row.Refs,
            "s2_sheet_row":"", "s2_number":"", "s2_site":"", "s2_thickness_text":"",
            "s2_reference":"", "setting":"", "rounding_displacement_m":"",
            "source_status":"Supplied citation only; primary thickness/location not independently re-audited in this study.",
            "primary_check":SOURCE_NOTES.get(i+2,"")}
        if candidates:
            n,r = candidates[0]; matched.append(n)
            entry.update(s2_sheet_row=n,s2_number=r[0],s2_site=r[3],s2_thickness_text=r[4],
                s2_reference=r[5],setting=r[6],rounding_displacement_m=geod.inv(row.Lon,row.Lat,r[1],r[2])[2],
                source_status="Matched to published Table S2 by exact three-decimal coordinate rounding; source datum not harmonized.")
        if i+2 == 55:
            entry["source_status"] = "Primary coordinates and field description verified; preservation uncertainty retained."
        ledger.append(entry)
    assert len(matched) == len(set(matched)) == 52
    assert all(e["s2_sheet_row"] for e in ledger[:52])
    pd.DataFrame(ledger).to_csv(output/"row-ledger.csv",index=False)
    unmatched = [{"s2_sheet_row":n,"s2_number":r[0],"longitude":r[1],"latitude":r[2],
        "site":r[3],"thickness_text":r[4],"reference":r[5],"setting":r[6],
        "decision":"Presence without a quantified usable thickness: no fabricated 0.5 cm value, detection threshold, or zero."}
        for n,r in source if n not in matched]
    assert len(unmatched) == 19 and all(r["thickness_text"] in ("pod","tr","?") for r in unmatched)
    pd.DataFrame(unmatched).to_csv(output/"s2-unquantified-presences.csv",index=False)
    protocol = {"input_sha256":INPUT_SHA256,"source_workbook_sha256":S2_SHA,"source_doi":S2_DOI,
        "source_values_sha256":sha256((sources/"Takarada2020-TableS2-values.json").read_bytes()).hexdigest(),
        "matched_original_rows":list(range(2,54)),"additional_rows":list(range(54,89)),
        "source_coverage":"52 of 87 accepted quantitative records trace to S2; all 19 other S2 records are unquantified presence records. Additional 35 records are not certified by this compilation audit.",
        "coordinate_caveat":"S2 H76 says datums differ between papers. Rounded coordinate matching establishes lineage, not datum accuracy. All fitting coordinates remain unchanged.",
        "cases":CASES,"validation_cases":["offset_sd1"],"sampling":Sampling(seed=20261108).__dict__,
        "selection":"No volume target, no model averaging over these cases, no automatic deletion of influential points. offset_sd1 uses the original four buffered geographic folds. The matched older offset_sd3 fit is the reference. offset_sd6 is a prior sensitivity, not a competing validated replacement.",
        "runner_sha256":sha256(Path(__file__).read_bytes()).hexdigest()}
    for name in CASES:
        folder = output/name; folder.mkdir(exist_ok=True)
        table, excluded, prior = case_data(name,original)
        table.to_csv(folder/"input.csv",index=False)
        data, meta = audited_data(folder/"input.csv")
        data = data.subset(~np.isin(data.row_ids,excluded))
        dump(folder/"definition.json", {"name":name,"definition":CASES[name],"excluded_original_csv_rows":excluded,
            "used_original_csv_rows":sorted(data.row_ids.tolist()),"input_sha256":meta["input_sha256"],
            "prior":prior.describe(),"original_input_sha256":INPUT_SHA256})
    path = output/"protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Frozen protocol changed: choose a fresh destination.")
    dump(path,protocol)
    return protocol


def run_case(output,name,validation=False):
    output = Path(output); protocol = json.loads((output/"protocol.json").read_text())
    folder = output/name; definition = json.loads((folder/"definition.json").read_text())
    data,meta = audited_data(folder/"input.csv")
    assert meta["input_sha256"] == definition["input_sha256"]
    data = data.subset(np.isin(data.row_ids,definition["used_original_csv_rows"]))
    _,_,prior = case_data(name,pd.read_csv(RAW))
    assert prior.describe() == definition["prior"]
    settings = Sampling(**protocol["sampling"])
    samples,fit = fit_checked(data,meta["vent_lonlat"][1],prior,folder/"posteriors",settings,
        progress=lambda t:print(t,flush=True))
    # Publish complete draws for every numerical attempt, not volume-only surrogates.
    result = {"definition":definition,"fit":fit,"runner_sha256":sha256(Path(__file__).read_bytes()).hexdigest()}
    dump(folder/"report.json",result)
    if validation:
        if name not in protocol["validation_cases"]:
            raise ValueError("Geographic validation was not declared for this case.")
        result["validation"] = cross_validate(data,meta["vent_lonlat"][1],folder/"validation",settings,
            progress=lambda t:print(t,flush=True),baseline_prior=prior)
        dump(folder/"report.json",result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--sources",type=Path)
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--cases",nargs="+",choices=CASES)
    parser.add_argument("--validation",action="store_true")
    args = parser.parse_args()
    if args.prepare:
        if args.sources is None: parser.error("--sources is required for preparation")
        prepare(args.output,args.sources)
    for name in args.cases or []:
        run_case(args.output,name,args.validation)
