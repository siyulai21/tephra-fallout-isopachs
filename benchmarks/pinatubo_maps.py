"""Make the Pinatubo reading sheets from the deposited source pixels, without OCR or resampling the model inputs."""
import csv

from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

from .pinatubo import AUDIT, FRAMES, SOURCE, position, src

OUT = AUDIT / "visual"
FONT_PATH = font_manager.findfont("DejaVu Sans")
BLUE = "#006c85"
INK = "#172b36"


def font(size):
    return ImageFont.truetype(FONT_PATH, size)


def crop_panel(canvas, raw, xy, left, top, marked=False):
    x, y = map(int, xy)
    crop = raw.crop((x-34, y-16, x+35, y+17)).resize((276,132), Image.Resampling.NEAREST)
    canvas.paste(crop,(left,top))
    if marked:
        # Only the margin contains the locator. The published marker and label stay visible.
        g = ImageDraw.Draw(canvas)
        cx, cy = left+138, top+66
        g.polygon([(cx-5,top-6),(cx+5,top-6),(cx,top)],fill=BLUE)
        g.polygon([(left-6,cy-5),(left-6,cy+5),(left,cy)],fill=BLUE)


def sources():
    return [
        ("P2", "Paladio-Melosantos, Table 2 and Figure 7", "paladio-figure-7.gif",
         [("P2-"+r["section"].replace(" ","-"),r["whole_layer_c_cm"]+" cm",[(r["x"],r["y"])]) for r in src("paladio-table-2.csv")]),
        ("K3", "Koyaguchi, Figure 3", "koyaguchi-figure-3.gif",
         [(r["id"],r["c1_mm"]+" and "+r["c2_mm"]+" mm",[(r["c1_x"],r["c1_y"]),(r["c2_x"],r["c2_y"])]) for r in src("koyaguchi-figure-3.csv")]),
        ("P7", "Paladio-Melosantos, Figure 7", "paladio-figure-7.gif",
         [(r["id"],(r["reading_cm"]+" cm" if r["reading_cm"] else "unread") + " · "+r["mark"],[(r["x"],r["y"])]) for r in src("paladio-figure-7.csv")]),
        ("D1", "David and colleagues, Table 1 and Figure 3", "david-figure-3.gif",
         [(r["site"],r["coarse_unit_cm"]+" cm coarse unit",[(r["point_x"],r["point_y"])]) for r in src("david-table-1.csv")]),
    ]


def reading_status(row):
    if row is None:
        return "Coarse unit · location schematic"
    if not row["kind"]:
        return "Outside fitted table"
    sign = {"measurement":"","lower_limit":"≥ ","upper_limit":"≤ "}[row["kind"]]
    return f"Fitted {sign}{float(row['value_cm']):g} cm · CSV row {row['accepted_csv_row']}"


def coordinate_example():
    """Locate a named section against the original longitude and latitude ticks."""
    raw = Image.open(SOURCE/"paladio-figure-7.gif").convert("RGB")
    section = next(r for r in src("paladio-table-2.csv") if r["section"] == "CAB 1")
    x, y = int(section["x"]), int(section["y"])
    lon, lat = position("P7", x, y)
    left, right, top, bottom, _, _ = FRAMES["P7"]
    sheet = Image.new("RGB", (1160, 849), "white")
    mark = ImageDraw.Draw(sheet)
    mark.text((12,10), "Check CAB 1 against the printed coordinate grid", fill=INK, font=font(25))
    mark.text((12,47), "Paladio-Melosantos Figure 7 · original map pixels at native scale", fill=INK, font=font(17))
    ox, oy = 145, 108
    sheet.paste(raw.crop((0,0,1000,681)), (ox,oy))
    for px, label in [(left,"120°00′ E"), (right,"120°45′ E")]:
        mark.text((ox+px,oy-27), label, anchor="mt", font=font(15), fill=BLUE)
        mark.line((ox+px,oy-9,ox+px,oy+4), fill=BLUE, width=2)
    for py, label in [(top,"15°30′ N"), (bottom,"15°00′ N")]:
        mark.text((ox-13,oy+py), label, anchor="rm", font=font(16), fill=BLUE)
        mark.line((ox-9,oy+py,ox+3,oy+py), fill=BLUE, width=2)
    cx, cy = ox+x, oy+y
    for u in range(int(ox+left), cx-15, 14):
        mark.line((u,cy,u+6,cy), fill=BLUE, width=1)
    for v in range(int(oy+top), cy-15, 14):
        mark.line((cx,v,cx,v+6), fill=BLUE, width=1)
    mark.ellipse((cx-10,cy-10,cx+10,cy+10), outline=BLUE, width=2)
    mark.text((12,807), f"CAB 1 · original pixel ({x}, {y}) · {lat:.5f}° N, {lon:.5f}° E · Table 2 thickness {section['whole_layer_c_cm']} cm", fill=INK, font=font(21))
    sheet.save(OUT/"coordinate-check.png", optimize=True)


def distal_sheet():
    """Print the original small-scale symbols with their map locators outside the pixels."""
    readings = src("context/paladio-figure-9-localities.csv")
    raw = Image.open(SOURCE/"paladio-figure-9.gif").convert("RGB")
    sheet = Image.new("RGB", (1000, 68 + 141*((len(readings)+3)//4)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((14, 9), "Figure 9 · Reported distal fallout and named places", font=font(24), fill=INK)
    draw.text((14, 41), "Original pixels ×5. Teal marks indicate the location of the mapped symbol.", font=font(15), fill=INK)
    for i, r in enumerate(readings):
        x, y = int(r["source_x"]), int(r["source_y"])
        left, top = (i % 4)*250+9, (i//4)*141+73
        draw.text((left, top), f"{i+1:02d}  {r['locality']}", font=font(16), fill=INK)
        crop = raw.crop((x-19, y-8, x+20, y+9)).resize((195, 85), Image.Resampling.NEAREST)
        sheet.paste(crop,(left+9,top+24))
        cx,cy = left+106, top+66
        draw.polygon([(cx-3,top+20),(cx+3,top+20),(cx,top+24)],fill=BLUE)
        draw.polygon([(left+4,cy-3),(left+4,cy+3),(left+9,cy)],fill=BLUE)
        draw.text((left, top+111), r["symbol_reading"], font=font(14), fill=INK)
    sheet.save(OUT/"distal-localities.png", optimize=True)


def draw_sheets():
    OUT.mkdir(exist_ok=True,parents=True)
    with (AUDIT/"selection.csv").open() as f:
        lookup = {r["id"]:r for r in csv.DictReader(f)}
    with (AUDIT/"survey-coverage/david-schematic.csv").open() as f:
        schematic = {r["site"]:r for r in csv.DictReader(f)}
    distal_sheet()
    coordinate_example()
    index = []
    guide = ["# Check the Pinatubo measurements by eye", "",
             "Three examples take about a minute to check. The [CAB-1 table entry and mapped section](field-check.md) give a measured 15 cm. On [K-03](visual/k3-01.png), 20 mm is clear and the other value carries a question mark, giving a 2 cm minimum. The [P7-056 symbol](visual/p7-07.png) is ambiguous and was excluded.", "",
             "Find an evidence ID in the [row index](reading-index.csv), then open its sheet below. Each card shows the original pixels enlarged four times. The small teal marks in the margin point to the recorded pixel centre. On the sublayer sheets the left and right panels show the same site in C1 and C2. On the other sheets the second panel repeats the location with the margin marks.", "",
             "P7 filled circles are independent Layer C field readings. Open circles are whole-section totals and bound Layer C from above. Squares report the Koyaguchi or Listanco survey. Triangles locate the sampled sections in Table 2. The section totals are printed in the table. A question mark records an unresolved reading. The [selection ledger](selection.csv) explains each fitting decision.", "",
             "To check a coordinate, open the [worked CAB 1 map](visual/coordinate-check.png). Its triangle is at original pixel (628, 411). It lies 588 of 874 pixels eastward across the 0.75° longitude span, giving 120.50458° E. It lies 388 of 600.5 pixels southward across the 0.5° latitude span, giving 15.17694° N. Each reading card also prints its mapped location. The full-precision coordinates in the row index reflect the map transformation. [Original Figure 7](../../sources/pinatubo-1991/paladio-figure-7.gif) · [Coordinate audit and uncertainty](README.md#map-coordinates)", ""]
    for tag,title,filename,readings in sources():
        raw = Image.open(SOURCE/filename).convert("RGB")
        guide += [f"## {title}", "", f"[Original image](../../sources/pinatubo-1991/{filename})", ""]
        if tag == "K3":
            guide += ["The mapped coordinate on each card comes from the C1 panel. Compare it with the independently registered C2 panel for every accepted pair. [Both sets of ticks, pixels and coordinates](../../../eruptions/pinatubo-1991-layer-c/results/source-checks/paired-map-coordinates/paired-coordinates.csv) · [Effect on the fitted volumes](../../../eruptions/pinatubo-1991-layer-c/results/source-checks/README.md#position-on-the-two-sublayer-maps)", ""]
        if tag == "P2":
            guide += ["[Printed sample names, thicknesses and geographic coordinates](field-check.md)", ""]
        for offset in range(0,len(readings),8):
            subset=readings[offset:offset+8]
            dest=f"{tag.lower()}-{1+offset//8:02d}.png"
            sheet=Image.new("RGB",(1250,65+222*((len(subset)+1)//2)),"white")
            g=ImageDraw.Draw(sheet)
            g.text((18,12),title+"   /   "+subset[0][0]+" to "+subset[-1][0],font=font(23),fill=INK)
            for j,(ident,reading,pair) in enumerate(subset):
                x,y=(j%2)*625+13,(j//2)*222+60
                row = lookup.get(ident)
                status=reading_status(row); color=BLUE if status.startswith("Fitted") else INK
                if row is None:
                    location = schematic[ident]
                    lon, lat = location["approximate_lon"], location["approximate_lat"]
                    precision = "Schematic"
                else:
                    lon, lat = row["longitude"], row["latitude"]
                    precision = "Mapped"
                g.text((x,y),ident+"   "+reading,font=font(17),fill=INK)
                g.text((x+5,y+26),"C1" if len(pair)==2 else "Original",font=font(13),fill=INK)
                g.text((x+306,y+26),"C2" if len(pair)==2 else "Location",font=font(13),fill=INK)
                crop_panel(sheet,raw,pair[0],x+7,y+53,marked=len(pair)==2)
                crop_panel(sheet,raw,pair[-1],x+310,y+53,marked=True)
                g.text((x+5,y+191),status,font=font(16),fill=color)
                g.text((x+306,y+191),f"{precision} {float(lat):.4f}° N, {float(lon):.4f}° E",font=font(14),fill=INK)
                g.line([(x,y+215),(x+598,y+215)],fill="#c9d4d8")
                index.append(dict(evidence_id=ident,source=tag,published_reading=reading,
                                  x1=pair[0][0],y1=pair[0][1],x2=pair[1][0] if len(pair)>1 else "",
                                  y2=pair[1][1] if len(pair)>1 else "",longitude_e=lon,latitude_n=lat,
                                  coordinate_source=precision,decision=status,sheet="visual/"+dest))
            sheet.save(OUT/dest,optimize=True)
            guide.append(f"- [{subset[0][0]} through {subset[-1][0]}](visual/{dest})")
        guide.append("")
    guide.extend(["## Regional ashfall reports", "", "[Nineteen labelled places on Figure 9](visual/distal-localities.png) · [Original figure and reporting qualifications](distal-reports.md)", "", "## Offshore stations and airfields","",
        "The marine bulk-thickness table identifies 11 cores by cruise and station. Compare its values with our [side-by-side station crosswalk](marine-check.md). The original corrected marine map uses station numbers, including 03 and 04. A number at a marine dot identifies its station. Thickness comes from the published table. The [airfield audit](README.md#airfields-and-distal-reports) documents two additional accumulation bounds.",""])
    guide.extend(["The SÜDMEER report identifies SO-114-23 as a 1 cm core. The PDF opens directly at the thickness statement, station chart and depth profiles in the [three-step verification guide](marine-check.md#südmeer-core-23).", ""])
    guide.extend(["## Other fall layers", "", "[Separate layer A, B and D reading sheets and coordinate index](other-phases/README.md)", ""])
    (AUDIT/"VISUAL-CHECK.md").write_text("\n".join(guide))
    with (AUDIT/"reading-index.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=index[0],lineterminator="\n");w.writeheader();w.writerows(index)


if __name__=="__main__":
    draw_sheets()
