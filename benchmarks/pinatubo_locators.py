"""Number the two original Pinatubo land maps at the pixels used for their reading cards."""
from hashlib import sha256
import json

import numpy as np
from PIL import Image, ImageDraw, PngImagePlugin
from scipy.spatial.distance import cdist
import shapely
from shapely import STRtree

from .pinatubo import src


TEAL = "#005567"
PAPER = "#ffffff"
# Each rectangle follows the printed map frame in the unmodified GIF.
PANELS = {
    "K3": [("C1", 0, 370, (45, 13, 485, 350)),
           ("C2", 370, 738, (46, 386, 486, 723))],
    "P7": [("Figure 7", 0, 1015, (40, 23, 965, 926))],
}


def grid_square(x, y, frame, columns, rows):
    left, top, right, bottom = frame
    col = min(columns - 1, int((x - left) * columns / (right - left)))
    row = min(rows - 1, int((y - top) * rows / (bottom - top)))
    return chr(ord("A") + col) + str(row + 1)


def label_positions(coords, frame, size, source_ink, protected):
    """Reserve space for every indexed mark, badge and leader before drawing."""
    points = np.asarray(coords, float)
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    radius = np.minimum(2.8, 0.32 * distances.min(axis=1))
    symbols = list(shapely.buffer(shapely.points(points), radius))
    placed = []
    result = [None] * len(points)
    width, height = size
    left, top, right, bottom = frame
    angles = 2 * np.pi * np.arange(64) / 64
    direction = np.column_stack((np.cos(angles), np.sin(angles)))
    half = np.array(size) / 2
    rim = np.min(half / np.maximum(np.abs(direction), 1e-10), axis=1)
    gaps = np.array([4, 7, 10, 14, 19, 25, 32, 41, 52, 65, 80, 95, 115])
    offsets = ((gaps[:, None] + rim)[:, :, None] * direction).reshape(-1, 2)
    # First allocate the neighbourhoods in which many sites share little white space.
    for i in np.argsort(-(distances < 33).sum(axis=1), kind="stable"):
        point = points[i]
        corners = np.rint(point + offsets - half).astype(int)
        x, y = corners.T
        valid = (x >= left + 1) & (x + width < right) & (y >= top + 1) & (y + height < bottom)
        corners = corners[valid]
        x, y = corners.T
        ends = np.clip(point, corners, corners + size)
        lines = shapely.linestrings(np.stack((np.broadcast_to(point, ends.shape), ends), axis=1))
        boxes = shapely.box(x - 0.8, y - 0.8, x + width + 0.8, y + height + 0.8)
        others = STRtree(symbols[:i] + symbols[i+1:] + protected + placed)
        free = np.ones(len(boxes), dtype=bool)
        for shapes in (boxes, lines):
            hits, _ = others.query(shapes, predicate="intersects")
            free[hits] = False
        # Count pixels in the entire badge rectangle. Keep dark print visible
        # and favour short leaders and locations clear of pale map linework.
        dark, pale = (source_ink[y+height+1, x+width+1] - source_ink[y, x+width+1]
                      - source_ink[y+height+1, x] + source_ink[y, x]).T
        free &= dark == 0
        cost = np.linalg.norm(ends - point, axis=1) + 0.2 * pale
        if not free.any():
            raise ValueError(f"No clear callout for source pixel {coords[i]}")
        pick = int(np.argmin(np.where(free, cost, np.inf)))
        placed.extend((boxes[pick], lines[pick].buffer(0.55)))
        result[i] = ([int(x[pick]), int(y[pick]), int(x[pick]+width), int(y[pick]+height)],
                     ends[pick].astype(int).tolist(), float(radius[i]))
    return result


def make_locator(tag, title, source, readings, output, font):
    """Render a complete original plate and attach a machine-checkable location index."""
    small = tag == "K3"
    scale, columns, rows = (4, 6, 5) if small else (3, 8, 8)
    size = (13, 8) if small else (20, 10)
    side, header, grid_band = (15, 48, 13) if small else (21, 62, 17)
    with Image.open(source) as im:
        original = im.convert("RGB")
    pixels = np.asarray(original)
    ink = np.stack((pixels.max(axis=2) < 185, pixels.min(axis=2) < 190), axis=2)
    integral = np.pad(ink.cumsum(0).cumsum(1), ((1, 0), (1, 0), (0, 0)))
    panels = PANELS[tag]
    height = header + original.height + grid_band * len(panels) + 4
    canvas = Image.new("RGB", ((original.width + 2*side)*scale, height*scale), PAPER)
    draw = ImageDraw.Draw(canvas)
    draw.text((side*scale, 7*scale), title, fill="#162d37", font=font((16 if small else 22)*scale))
    legend = ("Teal 40 = K-40. Each station appears on C1 and C2." if small else
              "Teal 040 = P7-040. Follow each leader to the open ring at the recorded location.")
    draw.text((side*scale, (29 if small else 37)*scale), legend, fill=TEAL, font=font((9 if small else 12)*scale))
    section_marks = [] if small else [(int(r["x"]), int(r["y"])) for r in src("paladio-table-2.csv")]
    meta = {"source": source.name, "source_sha256": sha256(source.read_bytes()).hexdigest(),
            "source_size": list(original.size), "scale": scale, "columns": columns, "rows": rows, "panels": []}
    for j, (name, start, stop, frame) in enumerate(panels):
        # Retain every native pixel of the plate, including graticules, scale bars and the inter-panel gap.
        base_y = header + (j + 1)*grid_band + start
        def at(x, y):
            return (round((side + x + 0.5)*scale), round((base_y + y - start + 0.5)*scale))
        strip = original.crop((0, start, original.width, stop))
        canvas.paste(strip.resize((strip.width*scale, strip.height*scale), Image.Resampling.NEAREST),
                     (side*scale, base_y*scale))
        x0, y0, x1, y1 = frame
        grid = ImageDraw.Draw(canvas, "RGBA")
        tick_font = font((8 if small else 11)*scale)
        for col in range(columns):
            x = x0 + (x1 - x0)*col/columns
            mid = x + (x1 - x0)/(2*columns)
            draw.text(at(mid, start-grid_band/2), chr(65+col), anchor="mm", font=tick_font, fill=TEAL)
            if col:
                for y in np.arange(y0, y1, 8):
                    grid.line([at(x, y), at(x, min(y+2, y1))], fill=(0, 93, 105, 92), width=max(1,scale//2))
        for row in range(rows):
            y = y0 + (y1 - y0)*row/rows
            mid = y + (y1 - y0)/(2*rows)
            for x in (-side/2, original.width+side/2-1):
                draw.text(at(x, mid), str(row+1), anchor="mm", font=tick_font, fill=TEAL)
            if row:
                for x in np.arange(x0, x1, 8):
                    grid.line([at(x, y), at(min(x+2, x1), y)], fill=(0, 93, 105, 92), width=max(1,scale//2))
        coords = [tuple(map(int, pair[j])) for _, _, pair in readings]
        # Named Table 2 triangles already have section codes on Figure 7.
        extra = [shapely.Point(x, y).buffer(3) for x, y in section_marks
                 if min(np.hypot(x-px, y-py) for px, py in coords) > 3]
        layout = label_positions(coords, frame, size, integral, extra)
        records = []
        for (ident, _, _), (x, y), (box, end, radius) in zip(readings, coords, layout, strict=True):
            delta = np.array(end) - (x, y)
            near = np.array((x, y)) + radius*delta/np.linalg.norm(delta)
            draw.line([at(*near), at(*end)], fill=PAPER, width=scale+1)
            draw.line([at(*near), at(*end)], fill=TEAL, width=max(2, round(0.7*scale)))
            draw.ellipse([at(x-radius, y-radius), at(x+radius, y+radius)], outline=TEAL,
                         width=max(1, round(min(0.75, radius/2)*scale)))
            records.append({"id": ident, "pixel": [x, y], "square": grid_square(x, y, frame, columns, rows),
                            "badge": box, "leader_end": end})
        for r in records:
            l, t, right, b = r["badge"]
            draw.rounded_rectangle([at(l, t), at(right, b)], radius=2*scale, fill=TEAL)
            draw.text(at((l+right)/2, (t+b)/2), r["id"].split("-")[-1], fill=PAPER, anchor="mm",
                      font=font((7 if small else 9)*scale))
        meta["panels"].append({"name": name, "frame": list(frame), "source_y": [start, stop],
                               "paste_origin": [side*scale, base_y*scale], "entries": records})
    info = PngImagePlugin.PngInfo()
    info.add_text("Description", json.dumps(meta, separators=(",", ":")))
    dest = output / ("koyaguchi-figure-3-numbered.png" if small else "paladio-figure-7-numbered.png")
    canvas.save(dest, optimize=True, pnginfo=info)
    return dest.name, meta


def locator_guide(tag, title, image, meta):
    panels = [p["entries"] for p in meta["panels"]]
    small = tag == "K3"
    example = "K-40" if small else "P7-040"
    square = next(r["square"] for r in panels[0] if r["id"] == example)
    link = f"visual/{image}"
    guide = [f"Look up any ID below, then follow its teal number to the open ring on the map. **{example}** is in column **{square[0]}**, row **{square[1:]}**. "
             + ("All 52 stations appear on both sublayer maps. The printed thicknesses are in millimetres." if small else
                "All 172 indexed locations appear, including copied readings, place names and unresolved entries. The printed thicknesses are in centimetres.")
             + f" [Open the full-resolution map]({link})", "",
             "<details>", f"<summary>Find any {'K' if small else 'P7'} ID by map square</summary>", "",
             "| Reading sheet | Audit ID and map square |", "|---|---|"]
    for offset in range(0, len(panels[0]), 8):
        block = panels[0][offset:offset+8]
        entries = []
        for i, r in enumerate(block, offset):
            cell = r["square"]
            if small and cell != panels[1][i]["square"]:
                cell = f"{cell} (C1), {panels[1][i]['square']} (C2)"
            entries.append(f"{r['id']} **{cell}**")
        sheet = f"visual/{tag.lower()}-{offset//8+1:02d}.png"
        guide.append(f"| [{block[0]['id']}–{block[-1]['id']}]({sheet}) | " + " · ".join(entries) + " |")
    return guide + ["", "</details>", "", f"[![{title} with every audit location numbered]({link})]({link})", ""]
