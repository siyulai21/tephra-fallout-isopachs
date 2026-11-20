"""Plot the primary B-Tm count profiles and recover a size-specific Greenland mass ceiling."""
import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.btm_cryptotephra import ROOT, SUIGETSU_YIELD, counts, digest, profile_integrals


def ice_budget():
    source = ROOT/"data/sources/b-tm/lee2024-particles-and-geochemistry.xlsx"
    ns = {"m":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(source) as book:
        manifest = ET.fromstring(book.read("xl/workbook.xml"))
        assert manifest.findall("m:sheets/m:sheet",ns)[2].get("name") == "Extended Data 3"
        sheet = ET.fromstring(book.read("xl/worksheets/sheet3.xml"))
    records = []
    for row in sheet.findall(".//m:sheetData/m:row",ns):
        if int(row.attrib["r"]) < 4:
            continue
        numeric = {c.get("r").rstrip("0123456789"):float(c.find("m:v",ns).text)
                   for c in row.findall("m:c",ns)
                   if c.get("t","n")=="n" and c.find("m:v",ns) is not None}
        records.append([numeric[k] for k in ("B","C","D","E")])
    values = np.array(records)
    depth, year = values[:,:2].T
    concentration = values[:,2:]
    if not (np.all(np.diff(depth)>0) and np.all(np.isfinite(values)) and np.min(concentration)>=0):
        raise ValueError("Particle loading requires ordered water-equivalent depth and complete concentrations.")
    answer = {"source":"Lee et al. (2024), Extended Data 3",
              "doi":"https://doi.org/10.6084/m9.figshare.26549344.v1","sha256":digest(source),
              "bands_micrometres":["2.6–4.5","4.5–9.5"],"units":"micrograms per square centimetre",
              "method":"No background subtraction. Integrate all insoluble particles against water-equivalent depth (cm), using 1 g/cm³ water. Attribution of every particle to B-Tm intentionally bounds its contribution in these two bands. Adjacent-endpoint maxima give a discretization sensitivity. Missing sizes have no inferred mass limit."}
    for key,mask in (("entire_record",np.ones(len(year),bool)),("946_to_948_CE",(year>=946)&(year<948))):
        c=concentration[mask]
        dz=100*np.diff(depth[mask])[:,None]
        a=((c[1:]+c[:-1])*.5*dz).sum(axis=0)
        high=(np.maximum(c[1:],c[:-1])*dz).sum(axis=0)
        answer[key]={"n":len(c),"years":[float(year[mask].min()),float(year[mask].max())],
                     "water_equivalent_interval_m":[float(depth[mask].min()),float(depth[mask].max())],
                     "trapezoid_per_band":a.tolist(),"trapezoid_sum":float(a.sum()),
                     "upper_endpoint_per_band":high.tolist(),"upper_endpoint_sum":float(high.sum())}
    return answer


def kushu_window_check():
    """Independently integrate the coarse scan, including all glass in 50–250 cm."""
    bars = [(86,137,122), (137,189,135), (189,236,192), (236,285,127), (285,309,403),
            (309,358,298), (358,455,112), (455,505,120), (505,531,114)]
    index = sum(200*(b-a)/444*counts(x,111,465,4000) for a,b,x in bars)
    outward = sum(200*(b-a+4)/440*counts(x,111,465,4000,outward=True) for a,b,x in bars)
    high_res = [r["concentration_per_g"] for r in profile_integrals() if r["site"]=="old-084"]
    return {"source":"Chen et al. (2016), Fig. 4, left panel", "scan_depths_cm":[50,250],
            "bars_top_bottom_right_pixels":bars, "calibration_pixels":{"zero_x":111,"4000_x":465,
            "50cm_y":86,"250cm_y":530}, "all_glass_index":index,"outward_reading_index":outward,
            "all_glass_equivalent_cm":index/SUIGETSU_YIELD,
            "outward_all_glass_equivalent_cm":outward/SUIGETSU_YIELD,
            "high_resolution_index_if_samples_1_to_2_cm":[sum(high_res),2*sum(high_res[:-2])+sum(high_res[-2:])],
            "interpretation":"The low-resolution scan covers several distinct horizons. All particles are assigned to B-Tm for this comparison. This is an independent window sensitivity. Neither count series is added to the other. In the width check, the event onset and peak retain their reported 1 cm widths."}


def plot(folder):
    bins=profile_integrals()
    fig,axes=plt.subplots(1,2,figsize=(8.6,4.5),layout="constrained")
    left,right=axes
    depths=np.arange(49,55)
    first=np.zeros(len(depths))
    for colour,ink in (("clear","#087e80"),("brown","#a85f38")):
        series=np.array([r["concentration_per_g"] for r in bins if r["fraction"]==colour])
        left.barh(depths,series,left=first,height=.82,color=ink,label=colour.capitalize())
        first+=series
    left.set(ylabel="Sample depth (cm)",xlabel="Glass shards / g dry sediment",title="Lake Sihailongwan")
    left.legend(frameon=False)
    top=140.
    for r in [r for r in bins if r["site"]=="old-084"]:
        w=r["width_cm"]
        right.barh(top+w/2,r["concentration_per_g"],height=w,color="#087e80",edgecolor="white",linewidth=.6)
        top+=w
    right.set(ylabel="Composite core depth (cm)",xlabel="Glass shards / g dry sediment",title="Lake Kushu")
    for ax in axes:
        ax.invert_yaxis()
        ax.spines[["top","right"]].set_visible(False)
        ax.ticklabel_format(axis="x",style="plain")
    fig.suptitle("B-Tm glass across the sampled horizons",fontweight="bold")
    fig.savefig(folder/"lake-count-profiles.png",dpi=170)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    folder=p.parse_args().output
    folder.mkdir(parents=True,exist_ok=True)
    (folder/"greenland-size-fraction.json").write_text(json.dumps(ice_budget(),indent=2,allow_nan=False)+"\n")
    (folder/"kushu-window-sensitivity.json").write_text(json.dumps(kushu_window_check(),indent=2,allow_nan=False)+"\n")
    plot(folder)
