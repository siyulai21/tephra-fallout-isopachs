"""Make the Pinatubo reading sheets from the deposited source pixels, without OCR or resampling the model inputs."""
import csv

from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

from .pinatubo import AUDIT, FRAMES, SOURCE, position, src
from .pinatubo_locators import locator_guide, make_locator

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


def reviewed_c1_labels():
    """Show the original characters and nearby compilation at the recorded review pixels."""
    with (AUDIT / "koyaguchi-c1-review.csv").open(newline="") as stream:
        reviews = list(csv.DictReader(stream))
    koya = Image.open(SOURCE / "koyaguchi-figure-3.gif").convert("RGB")
    paladio = Image.open(SOURCE / "paladio-figure-7.gif").convert("RGB")
    columns = [(koya, "c1", 32, 14, 5, "Koyaguchi C1 · mm"),
               (koya, "c2", 32, 14, 5, "Koyaguchi C2 · mm"),
               (paladio, "figure7", 40, 18, 4, "Nearby Figure 7 label · cm")]
    canvas = Image.new("RGB", (1135, 52 + 165 * len(reviews)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (raw, prefix, dx, dy, scale, heading) in enumerate(columns):
        left = 81 + 345 * column
        draw.text((left, 10), heading, fill=INK, font=font(18))
        for index, review in enumerate(reviews):
            top = 50 + 165 * index
            if column == 0:
                draw.text((5, top + 47), review["evidence_id"], fill=INK, font=font(18))
            x, y = int(review[prefix + "_x"]), int(review[prefix + "_y"])
            crop = raw.crop((x - dx, y - dy, x + dx + 1, y + dy + 1))
            canvas.paste(crop.resize((crop.width * scale, crop.height * scale),
                                     Image.Resampling.NEAREST), (left, top))
            cx, cy = left + dx * scale + scale // 2, top + dy * scale + scale // 2
            draw.polygon([(cx - 4, top - 6), (cx + 4, top - 6), (cx, top - 1)], fill=BLUE)
            draw.polygon([(left - 6, cy - 4), (left - 6, cy + 4), (left - 1, cy)], fill=BLUE)
    canvas.save(OUT / "koyaguchi-c1-review.png", optimize=True)


def reviewed_figure7_labels():
    """Rebuild the four review plates from the registered pixels on Figure 7."""
    with (AUDIT / "paladio-figure-7-review.csv").open(newline="") as stream:
        reviews = list(csv.DictReader(stream))
    raw = Image.open(SOURCE / "paladio-figure-7.gif").convert("RGB")
    kind_label = {"measurement": "Measured", "upper_limit": "Upper bound", "lower_limit": "Lower bound"}
    symbol_label = {"solid": "Field circle", "square": "Compilation square", "total": "Open circle",
                    "place": "Place name", "no_reading": "No thickness at pointer", "amb": "Crowded symbols"}
    for start in range(0, len(reviews), 8):
        part = reviews[start:start + 8]
        canvas = Image.new("RGB", (1160, 67 + 220 * ((len(part) + 1) // 2)), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), f"Paladio-Melosantos Figure 7  ·  {part[0]['evidence_id']} to {part[-1]['evidence_id']}",
                  fill=INK, font=font(22))
        draw.text((12, 39), "Original pixels ×5. The teal pointers mark the recorded position. Thicknesses are in centimetres.",
                  fill=INK, font=font(15))
        for i, row in enumerate(part):
            left, top = 14 + (i % 2) * 580, 69 + (i // 2) * 220
            x, y = int(row["source_x"]), int(row["source_y"])
            text = f"{row['evidence_id']}   ·   curator reading {row['curator_reading']}   ·   pixel ({x}, {y})"
            draw.text((left, top), text, fill=INK, font=font(16))
            crop = raw.crop((x - 51, y - 14, x + 52, y + 15))
            l, t = left + 9, top + 32
            canvas.paste(crop.resize((515, 145), Image.Resampling.NEAREST), (l, t))
            cx, cy = l + 257, t + 72
            draw.polygon([(cx-4,t-6), (cx+4,t-6), (cx,t-1)], fill=BLUE)
            draw.polygon([(l-6,cy-4), (l-6,cy+4), (l-1,cy)], fill=BLUE)
            kind = row["model_kind"]
            result = f"{kind_label[kind]} {float(row['model_value_cm']):g} cm" if kind else "Outside fitted input"
            draw.text((left, top + 185), f"{symbol_label[row['reviewed_mark']]}   ·   {result}",
                      fill=BLUE if kind else INK, font=font(16))
        canvas.save(OUT / f"p7-review-{start // 8 + 1:02d}.png", optimize=True)


def reviewed_figure7_context():
    """Compare the Botolan town dot and two coastal compilation marks with the other plates."""
    paladio = Image.open(SOURCE / "paladio-figure-7.gif").convert("RGB")
    koya = Image.open(SOURCE / "koyaguchi-figure-3.gif").convert("RGB")
    records = {r["id"]: r for r in src("paladio-figure-7.csv")}
    pairs = {r["id"]: r for r in src("koyaguchi-figure-3.csv")}
    rows = []
    botolan = records["P7-028"]
    lon, lat = position("P7", botolan["x"], botolan["y"])
    panels = [(paladio, int(botolan["x"]), int(botolan["y"]), "Figure 7 · Layer C")]
    for ticks in src("context/phase-graticules.csv"):
        if ticks["figure"] in ("6", "13"):
            bounds = {k: float(v) for k, v in ticks.items()}
            x = bounds["west_x"] + (lon - bounds["west_lon"]) * (bounds["east_x"] - bounds["west_x"]) / (bounds["east_lon"] - bounds["west_lon"])
            y = bounds["north_y"] + (lat - bounds["north_lat"]) * (bounds["south_y"] - bounds["north_y"]) / (bounds["south_lat"] - bounds["north_lat"])
            other = Image.open(SOURCE / "context" / f"paladio-figure-{ticks['figure']}.gif").convert("RGB")
            panels.append((other, round(x), round(y), f"Figure {ticks['figure']} · same map location"))
    rows.append(("P7-028 · Botolan locality", panels))
    for key, partner in (("P7-162", "K-52"), ("P7-170", "K-48")):
        point, pair = records[key], pairs[partner]
        panels = [(paladio, int(point["x"]), int(point["y"]), "Figure 7 · cm")]
        for field in ("c1", "c2"):
            panels.append((koya, int(pair[field + "_x"]), int(pair[field + "_y"]), partner + " · " + field.upper() + " · mm"))
        rows.append((key + " · field and compilation marks", panels))
    sheet = Image.new("RGB", (1130, len(rows) * 201), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (title, panels) in enumerate(rows):
        top = 201 * i
        draw.text((10, top + 3), title, fill=INK, font=font(18))
        for j, (raw, x, y, label) in enumerate(panels):
            left = 12 + j * 375
            draw.text((left, top + 34), label, fill=INK, font=font(15))
            crop_panel(sheet, raw, (x, y), left + 5, top + 61, marked=True)
    sheet.save(OUT / "p7-review-context.png", optimize=True)


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
    reviewed_c1_labels()
    reviewed_figure7_labels()
    reviewed_figure7_context()
    index = []
    guide = ["# Check the Pinatubo measurements by eye", "",
             "Find an audit ID on either full map. Each numbered figure has a lookup by ID and links to the original-pixel reading sheets. [Koyaguchi Figure 3](#koyaguchi-figure-3) · [Paladio-Melosantos Figure 7](#paladio-melosantos-figure-7)", "",
             "Three examples take about a minute to check. The [CAB-1 table entry and mapped section](field-check.md) give a measured 15 cm. On [K-03](visual/k3-01.png), 20 mm is clear and the other value carries a question mark, giving a 2 cm minimum. The [P7-056 symbol](visual/p7-07.png) is ambiguous and was excluded.", "",
             "Find an evidence ID in the [row index](reading-index.csv), then open its sheet below. Each card shows the original pixels enlarged four times. The small teal marks in the margin point to the recorded pixel centre. On the sublayer sheets the left and right panels show the same site in C1 and C2. On the other sheets the second panel repeats the location with the margin marks.", "",
             "P7 filled circles are independent Layer C field readings. Open circles are whole-section totals and bound Layer C from above. Squares report the Koyaguchi or Listanco survey. Triangles locate the sampled sections in Table 2. The section totals are printed in the table. A question mark records an unresolved reading. The [selection ledger](selection.csv) explains each fitting decision.", "",
             "To check a coordinate, open the [worked CAB 1 map](visual/coordinate-check.png). Its triangle is at original pixel (628, 411). It lies 588 of 874 pixels eastward across the 0.75° longitude span, giving 120.50458° E. It lies 388 of 600.5 pixels southward across the 0.5° latitude span, giving 15.17694° N. Each reading card also prints its mapped location. The full-precision coordinates in the row index reflect the map transformation. [Original Figure 7](../../sources/pinatubo-1991/paladio-figure-7.gif) · [Coordinate audit and uncertainty](README.md#map-coordinates)", ""]
    for tag,title,filename,readings in sources():
        raw = Image.open(SOURCE/filename).convert("RGB")
        guide += [f"## {title}", "", f"[Original image](../../sources/pinatubo-1991/{filename})", ""]
        if tag in ("K3", "P7"):
            image, locations = make_locator(tag, title, SOURCE/filename, readings, OUT, font)
            guide.extend(locator_guide(tag, title, image, locations))
        if tag == "K3":
            guide += ["Four C1 readings were reviewed against both original panels and the neighbouring Figure 7 labels. [Read the four decisions](koyaguchi-c1-review.md) · [Compare the original pixels](visual/koyaguchi-c1-review.png)", ""]
            guide += ["The mapped coordinate on each card comes from the C1 panel. Compare it with the independently registered C2 panel for every accepted pair. [Both sets of ticks, pixels and coordinates](../../../eruptions/pinatubo-1991-layer-c/results/source-checks/paired-map-coordinates/paired-coordinates.csv) · [Effect on the fitted volumes](../../../eruptions/pinatubo-1991-layer-c/results/source-checks/README.md#position-on-the-two-sublayer-maps)", ""]
        if tag == "P2":
            guide += ["[Printed sample names, thicknesses and geographic coordinates](field-check.md)", ""]
        if tag == "P7":
            guide += ["[Review of 32 Figure 7 entries with original crops and fitting decisions](paladio-figure-7-review.md)", ""]
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
