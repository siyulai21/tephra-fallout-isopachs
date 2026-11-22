"""Read surveyed Calbuco's published distal contour at fixed latitudes and score both fits."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
from pyproj import Geod

from src.Laea import make_laea_transform
from src.empirical_isopach.models.nested.continuation import load_packet
from src.empirical_isopach.models.nested.reach import ReachDraws
from src.empirical_isopach.models.nested.study import digest, write_json

FIGURE = Path("data/sources/nested-reach/calbuco-mingari-fig2.png")
SHA = "038d1827e8ea71ca7619789cba85cf2fb25a4f922f83508d19e2a20a75d9333b"


def crossings():
    if digest(FIGURE) != SHA:
        raise ValueError("Printed Calbuco reference figure changed")
    # Source pixel centres, indexed from the upper left. Fit the printed graticule
    # rather than assuming that the colour-image frame is a geographic boundary.
    xticks = [70.5,217.5,363.5,510.,656.5,802.5,949.5,1095.5,1242.]
    yticks = [131.5,278.5,424.5,571.,717.5,863.5]
    px = np.polyfit(np.arange(-74.,-65.),xticks,1)
    py = np.polyfit(-np.arange(36.,42.),yticks,1)
    figure = np.asarray(Image.open(FIGURE).convert("RGB")).astype(int)
    dark = (figure.max(axis=2)<100) & (np.ptp(figure,axis=2)<28)
    rows = []
    for lat in -np.arange(36.5,41.1,.5):
        y = round(float(np.polyval(py,lat)))
        # Printed locator inset occupies the upper right; the key occupies the lower right.
        left = 65
        right = 935 if y<544 else 900 if y>777 else 1329
        x = np.flatnonzero(dark[y,left:right])+left
        strokes = [np.mean(s) for s in np.split(x,np.flatnonzero(np.diff(x)>2)+1) if len(s)]
        if len(strokes)<2:
            raise ValueError("Printed outer isopach does not have two resolved crossings")
        west,east = strokes[0],strokes[-1]
        rows.append({"latitude":float(lat),"pixel_y":y,"pixel_west":float(west),"pixel_east":float(east),
                     "west_lon":float((west-px[1])/px[0]),"east_lon":float((east-px[1])/px[0])})
    return figure,rows,px,py


def compare(home):
    figure, rows, px, py = crossings()
    base = load_packet(home/"training-parent.npz").reduced(1024)
    new = ReachDraws.load(home/"corridor.npz").reduced(1024)
    forward,_,_=make_laea_transform(-72.618,-41.330)
    geod=Geod(ellps="WGS84")
    lon = np.linspace(-74.,-65.4,1001)
    mid = (lon[1:]+lon[:-1])/2
    cuts = []
    for row in rows:
        lat = row["latitude"]
        xy = np.column_stack(forward.transform(mid,np.full_like(mid,lat)))
        dims = geod.inv(lon[:-1],np.full_like(mid,lat),lon[1:],np.full_like(mid,lat))[2]/1000.
        true = (mid>=row["west_lon"])&(mid<=row["east_lon"])
        rec = dict(row)
        for name,model in (("whole_area",base),("corridor",new)):
            med = np.concatenate([np.median(model.field(batch),axis=0) for batch in np.array_split(xy,8)])
            covered = med>=np.log(.01)  # Figure 2 labels are millimetres, model units are cm.
            rec[name+"_extra_km"]=float(dims[covered&~true].sum())
            rec[name+"_missed_km"]=float(dims[~covered&true].sum())
            rec[name+"_overlap_km"]=float(dims[covered&true].sum())
            rec[name+"_touches_frame"]=bool(covered[0] or covered[-1])
        cuts.append(rec)
    with (home/"mapped-01mm-transects.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(cuts[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(cuts)
    results={"units":"Printed distal limit 0.1 mm = 0.01 cm. Length in WGS84 km along each parallel.",
             "source_sha256":SHA,"pixel_x_from_longitude":px.tolist(),"pixel_y_from_latitude":py.tolist(),
             "latitude_range_south_deg":[36.5,41.],"window_longitude_deg":[float(lon[0]),float(lon[-1])],
             "quadrature_intervals":len(mid),"shared_latitudes":len(rows),
             "comparison":"Equal-weight transects against an author-interpreted contour. No surveyed area or unobserved thickness is asserted."}
    for key in ("whole_area","corridor"):
        results[key]={part:float(np.mean([r[key+"_"+part+"_km"] for r in cuts])) for part in ("extra","missed","overlap")}
        results[key]["touching_frame"]=sum(r[key+"_touches_frame"] for r in cuts)
    write_json(home/"mapped-01mm.json",results)
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12.5,5.9),layout="constrained",gridspec_kw={"width_ratios":[1.2,1.]})
    ax.imshow(figure)
    ax.scatter([v["pixel_west"] for v in rows],[v["pixel_y"] for v in rows],s=26,facecolors="none",edgecolors="#bc258c",linewidths=1.5)
    ax.scatter([v["pixel_east"] for v in rows],[v["pixel_y"] for v in rows],s=26,facecolors="none",edgecolors="#bc258c",linewidths=1.5)
    ax.set_axis_off()
    ax.set_title("Outer 0.1 mm isopach · printed crossings",fontsize=12)
    ys=[-v["latitude"] for v in cuts]
    for key,color,offset in (("whole_area","#a04a2c",-.06),("corridor","#005a67",.06)):
        yp=np.asarray(ys)+offset
        for y,r in zip(yp,cuts,strict=True):
            bx.plot([-r[key+"_missed_km"],r[key+"_extra_km"]],[y,y],color=color,lw=2.)
            bx.scatter([-r[key+"_missed_km"],r[key+"_extra_km"]],[y,y],s=13,color=color)
    bx.axvline(0,color="#596264",lw=.75)
    bx.invert_yaxis()
    bx.set(ylabel="Latitude (°S)",xlabel="← Missed mapped span   |   Extra mapped span → (km)",title="Conditional median footprints")
    bx.legend(handles=[Line2D([],[],color="#a04a2c",label="Whole-area rank"),Line2D([],[],color="#005a67",label="Corridors")],loc="best",fontsize=9)
    fig.suptitle("Calbuco 2015 · Comparison with Romero’s independently mapped deposit")
    fig.savefig(home/"mapped-contour.png",dpi=135)
    plt.close(fig)
    return results
