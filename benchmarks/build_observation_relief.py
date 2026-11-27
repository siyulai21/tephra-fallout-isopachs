"""Rebuild the observation atlas elevation assets from pinned ETOPO GeoTIFFs."""

import argparse
from collections import defaultdict
from hashlib import file_digest
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from .observation_relief import ASSETS, RECEIPT


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def raster(path):
    with Image.open(path) as image:
        return np.rint(np.maximum(np.asarray(image), 0)).astype(np.uint16)


def rebuild(source_dir):
    receipt = json.loads(RECEIPT.read_text())
    for source in receipt["geotiffs"]:
        path = source_dir / source["file"]
        if digest(path) != source["sha256"]:
            raise ValueError(f"Wrong ETOPO source bytes: {path.name}")
    global_source = source_dir / receipt["geotiffs"][0]["file"]
    fine_sources = [source_dir / item["file"] for item in receipt["geotiffs"][1:]]
    products = defaultdict(dict)
    with TemporaryDirectory() as temporary:
        folder = Path(temporary)
        coarse = folder / "global.tif"
        subprocess.run(["gdal_translate", "-q", "-outsize", "10800", "5400", "-r", "average",
                        global_source, coarse], check=True)
        globe = raster(coarse)
        fine = folder / "fine.vrt"
        subprocess.run(["gdalbuildvrt", "-q", fine, *fine_sources], check=True)
        for tile in receipt["tiles"]:
            west, south, east, north = tile["bounds"]
            if tile["arcseconds"] == 120:
                top, bottom = round((90 - north)*30), round((90 - south)*30)
                columns = np.arange(round((west+180)*30), round((east+180)*30)) % globe.shape[1]
                values = globe[top:bottom, columns]
            else:
                crop = folder / "crop.tif"
                box = [str(value) for value in (west, north, east, south)]
                subprocess.run(["gdal_translate", "-q", "-projwin", *box, fine, crop], check=True)
                values = raster(crop)
            spacing = tile["arcseconds"] / 3600
            assert values.shape == (round((north-south)/spacing), round((east-west)/spacing))
            products[tile["file"]][tile["key"]] = values
    for name, arrays in products.items():
        np.savez_compressed(ASSETS / name, **arrays)
    for tile in receipt["tiles"]:
        tile["sha256"] = digest(ASSETS / tile["file"])
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_directory", type=Path, help="Directory with the GeoTIFFs named in source.json")
    rebuild(parser.parse_args().source_directory)
