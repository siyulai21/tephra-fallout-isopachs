"""Check the six uploaded crops against the identified thesis scan and draw the reading sheet."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, map_coordinates

HOME = Path(__file__).resolve().parent


def draw_index(clips, output):
    with (HOME.parent / "young1990-map-picks.csv").open() as src:
        picks = {(int(r["pdf_page"]), int(r["site"])): (float(r["pixel_x"]), float(r["pixel_y"]))
                 for r in csv.DictReader(src)}
    for clip in clips:
        for mark in clip["marks"]:
            picks[clip["page"], mark["site"]] = (mark["page_x"], mark["page_y"])
    fig = plt.figure(figsize=(15, 12), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.25])
    panels = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[:, 1])]
    for ax, page, title in zip(panels, (321, 323, 324),
                               ("Proximal precursor map", "Proximal climactic map", "Medial climactic map"), strict=True):
        points = [(site, xy) for (p, site), xy in picks.items() if p == page]
        xy = np.array([v for _, v in points])
        ax.set(xlim=(xy[:, 0].min()-45, xy[:, 0].max()+75), ylim=(xy[:, 1].max()+45, xy[:, 1].min()-50),
               aspect="equal", title=f"{title} · PDF p. {page}", xlabel="Page x (128-dpi pixels)", ylabel="Page y")
        ax.scatter(*xy.T, s=7, c="#445460", zorder=3)
        # Move neighbouring labels apart; the line always ends at the source-pixel position.
        label_x, label_y = [], []
        step = 13 if page == 324 else 18
        for site, (x, y) in sorted(points, key=lambda item: item[1][1]):
            for rise in (0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5):
                tx, ty = x+6, y+rise*step
                if all(abs(tx-i)>step*1.8 or abs(ty-j)>step*.93 for i,j in zip(label_x,label_y)):
                    break
            ax.annotate(str(site), (x,y), (tx,ty), fontsize=7, va="center", color="#243741",
                        arrowprops=dict(arrowstyle="-", color="#899297", lw=.42, shrinkA=0, shrinkB=0))
            label_x.append(tx)
            label_y.append(ty)
        for clip in (c for c in clips if c["page"] == page):
            cx, cy, sx, sy = clip["registration"]
            w, h = np.array(clip["clip_dimensions"])*[sx,sy]
            box = (cx+.5-w/2, cy+.5-h/2)
            ax.add_patch(Rectangle(box,w,h,fill=False,ec="#ab482e",lw=1.3))
            ax.annotate(f"Crop {clip['clip']}",box,xytext=(0,4),textcoords="offset points",
                        color="#97361e",fontsize=8,fontweight="bold",bbox=dict(fc="white",ec="none",pad=.4))
        ax.grid(alpha=.12)
        ax.tick_params(labelsize=8)
    fig.suptitle("Young’s Appendix 1d field sections\nSource section numbers and locations of the six reviewed crops",fontsize=16)
    fig.savefig(output,dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("thesis", type=Path)
    parser.add_argument("--output", type=Path, default=HOME / "source-comparison.png")
    parser.add_argument("--index", type=Path, default=HOME / "appendix-index.png")
    args = parser.parse_args()
    source = json.loads((HOME / "clip-registration.json").read_text())
    assert hashlib.sha256(args.thesis.read_bytes()).hexdigest() == source["source_pdf_sha256"]
    fig, axes = plt.subplots(6, 2, figsize=(13, 13), layout="constrained")
    with tempfile.TemporaryDirectory() as tmp:
        prefix = str(Path(tmp) / "young")
        subprocess.run(["pdftoppm", "-f", "321", "-l", "324", "-r", "400", "-gray",
                        "-png", str(args.thesis), prefix], check=True)
        for clip, (left, right) in zip(source["clips"], axes, strict=True):
            image_path = HOME / clip["file"]
            assert hashlib.sha256(image_path.read_bytes()).hexdigest() == clip["sha256"]
            annotated = np.array(Image.open(image_path).convert("RGB")) / 255
            page = np.array(Image.open(f"{prefix}-{clip['page']}.png").convert("L")) / 255
            height, width = annotated.shape[:2]
            assert [width, height] == clip["clip_dimensions"]
            x0, y0, sx, sy = clip["registration"]
            row, col = np.mgrid[:height, :width]
            page_x = (x0 + (col - (width - 1) / 2) * sx + .5) * 400 / 128 - .5
            page_y = (y0 + (row - (height - 1) / 2) * sy + .5) * 400 / 128 - .5
            original = map_coordinates(page, [page_y, page_x], order=1)
            weight = ~binary_dilation(np.ptp(annotated, axis=2) > .04, iterations=2)
            a = (1 - annotated.mean(axis=2))[weight]
            b = (1 - original)[weight]
            score = 2 * np.sum(a * b) / np.sum(a * a + b * b)
            assert score > .99, (image_path.name, score)
            sites = ", ".join(str(m["site"]) for m in clip["marks"])
            print(f"Crop {clip['clip']}, PDF p. {clip['page']}, sections {sites}, print overlap {score:.5f}")
            left.imshow(annotated)
            right.imshow(original, cmap="gray", vmin=0, vmax=1)
            left.set_title(f"Crop {clip['clip']} · sections {sites}", loc="left", fontsize=11)
            right.set_title(f"Appendix 1d · PDF p. {clip['page']}", loc="left", fontsize=11)
            left.axis("off")
            right.axis("off")
    fig.suptitle("Young’s Mazama field sections\nContributor’s centroids (left) and original map print (right)", fontsize=15)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    draw_index(source["clips"], args.index)


if __name__ == "__main__":
    main()
