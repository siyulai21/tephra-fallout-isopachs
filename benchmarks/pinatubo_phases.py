"""Publish source-specific checks for the preclimactic A/B and postclimactic D surveys."""
from collections import Counter, defaultdict
from pathlib import Path
import re

from PIL import Image, ImageDraw

from .pinatubo import AUDIT, SOURCE, src
from .pinatubo_evidence import write
from .pinatubo_maps import BLUE, INK, font

OUT = AUDIT/"other-phases"
CONTEXT = SOURCE/"context"


def reading(raw, unit, figure):
    """Single positive thicknesses alone have a numeric value in the additional-phase index."""
    text = raw.strip("()")
    if figure == "K3" and text == "?":
        return "question mark in source", ""
    if "?" in text or text == "unclear":
        return "reading or marker assignment unresolved", ""
    if text == "0":
        return "reported zero, visibility threshold unspecified", ""
    if text == "dash":
        return "dash at marked site, no numeric thickness", ""
    if text == "+":
        return "published plus sign, no numeric thickness", ""
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return "positive point reading", round(float(text)/(10 if unit == "mm" else 1), 4)
    return "qualified or multiple reading, preserved in source units", ""


def inventory():
    rows = []
    controls = {r["figure"]:r for r in src("context/phase-graticules.csv")}
    def add(identity, phase, figure, image, xy, lonlat, raw, unit, method, linked):
        status, value = reading(raw, unit, figure)
        rows.append(dict(evidence_id=identity, phase=phase, source_figure=figure,
                         original_image=image, source_x=int(xy[0]), source_y=int(xy[1]),
                         reported_or_candidate=raw, unit=unit, review=status, single_positive_cm=value,
                         approximate_longitude=round(float(lonlat[0]), 3), approximate_latitude=round(float(lonlat[1]), 3),
                         position_evidence=method, linked_C_marker=linked, sheet=""))
    for r in src("context/precursor-thickness.csv"):
        add(r["evidence_id"],r["phase"],"K3",r["image"],(r["source_x"],r["source_y"]),
            (r["longitude"],r["latitude"]),r["reported"],r["units"],r["location_method"],r["matched_field_site"])
    for r in src("context/paladio-other-phase-marks.csv"):
        f=r["figure"]
        c={k:float(v) for k,v in controls[f].items() if k!="figure"}
        x,y=int(r["source_x"]),int(r["source_y"])
        lon=c["west_lon"]+(x-c["west_x"])*(c["east_lon"]-c["west_lon"])/(c["east_x"]-c["west_x"])
        lat=c["north_lat"]+(y-c["north_y"])*(c["south_lat"]-c["north_lat"])/(c["south_y"]-c["north_y"])
        add(r["evidence_id"],r["phase"],"P"+f,f"paladio-figure-{f}.gif",(x,y),(lon,lat),
            r["reading_cm"],"cm","Native figure graticule","")
    return rows


def build(output=OUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = inventory()
    groups=defaultdict(list)
    for row in rows:
        groups[(row["source_figure"],row["phase"])].append(row)
    pages=[]
    for (figure,phase), items in groups.items():
        im=Image.open(CONTEXT/items[0]["original_image"]).convert("RGB")
        pages.extend([f"### {figure} · layer {phase}", "",
            f"[Full original plate](../../../sources/pinatubo-1991/context/{items[0]['original_image']})", ""])
        for idx in range(0,len(items),12):
            part=items[idx:idx+12]
            dest=f"{figure.lower()}-{phase.lower()}-{idx//12+1:02d}.png"
            sheet=Image.new("RGB",(1050,60+151*((len(part)+2)//3)),"white")
            drawing=ImageDraw.Draw(sheet)
            drawing.text((15,10),f"Layer {phase}  ·  {figure}  ·  {part[0]['evidence_id']} to {part[-1]['evidence_id']}",
                         font=font(22),fill=INK)
            for j,r in enumerate(part):
                r["sheet"]=dest
                left,top=j%3*350+12,j//3*151+57
                x,y=r["source_x"],r["source_y"]
                drawing.text((left,top),r["evidence_id"]+"   "+r["reported_or_candidate"]+" "+r["unit"],font=font(18),fill=INK)
                patch=im.crop((x-34,y-11,x+35,y+12)).resize((276,92),Image.Resampling.NEAREST)
                sheet.paste(patch,(left+10,top+28))
                drawing.polygon([(left+145,top+23),(left+151,top+23),(left+148,top+28)],fill=BLUE)
                drawing.polygon([(left+4,top+71),(left+4,top+77),(left+10,top+74)],fill=BLUE)
                drawing.text((left+10,top+122),f"Original pixel ({x}, {y})",font=font(13),fill=INK)
            sheet.save(output/dest,optimize=True)
            pages.append(f"- [{part[0]['evidence_id']} through {part[-1]['evidence_id']}]({dest})")
        pages.append("")
    write(output/"readings.csv",rows)
    counts=Counter(r["phase"] for r in rows)
    unresolved=sum(r["review"]=="reading or marker assignment unresolved" for r in rows)
    lines=["# Other Pinatubo fall layers", "",
           f"The additional survey covers **{len(rows)} source entries**. Layer A has {counts['A']}, layer B has {counts['B']} and layer D has {counts['D']}. Each entry links to the original image pixels. {unresolved} labels or marker assignments remain unresolved. A number followed by a question mark on a reading sheet identifies an uncertain transcription. Koyaguchi’s printed question marks and parentheses are preserved in the source column. [Row index with coordinates and reading status](readings.csv)", "",
           "Layer A represents the 12 June morning eruption. Layer B accumulated from late 12 June to early afternoon on 15 June. Layer D accumulated over about six weeks after the climax. Koyaguchi gives millimetres and distinguishes June and December observations with parentheses. Paladio-Melosantos gives centimetres and includes measurements from the earlier surveys. The source entries include repeat observations, mapped zeros, traces and sites with unresolved labels. The principal 15 June Layer C fit uses its own audited observations. [Field stratigraphy and figure captions](https://pubs.usgs.gov/pinatubo/paladio/index.html) · [Koyaguchi’s survey](https://pubs.usgs.gov/pinatubo/koya/index.html)", "",
           "The teal pointers identify the figure pixel. Geographic coordinates have the precision of the original maps. Koyaguchi’s A and B sites match the corresponding C1 field markers. For the three Paladio-Melosantos maps the original axis ticks give linear longitude and latitude scales. The numeric column contains only legible single positive readings in centimetres. The source column retains bounds, multiple values, dashes and reported zeros. [Axis controls](../../../sources/pinatubo-1991/context/phase-graticules.csv) · [Four photographed sections with phase-by-phase thicknesses](../../../sources/pinatubo-1991/context/phase-photo-sections.csv) · [Original photograph plate](../../../sources/pinatubo-1991/context/paladio-figure-4.jpg)", "",
           "At the Gumain River section, photograph 4C gives 23 cm for B and 31 cm for C. The 23 cm B dot (P6-068) falls near the 32 cm C dot (P7-099). The original C map supplies the located measurement. The photograph supplies an independent reading at the named locality. [Layer C reading sheet](../visual/p7-13.png) · [Photograph caption](https://pubs.usgs.gov/pinatubo/paladio/index.html)", "",
           "Figure 2 outlines the April to early June ashfall footprint. It supplies no site-specific bed thickness. [Early footprint](../../../sources/pinatubo-1991/context/paladio-figure-2.gif)", "", *pages]
    (output/"README.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    build()
