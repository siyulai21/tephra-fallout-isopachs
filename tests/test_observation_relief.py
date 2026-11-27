"""Check that terrain sampling and water masking match the cartography."""
import numpy as np

from benchmarks.observation_relief import elevation, grid, inventory, sample, terrain_rgb
from benchmarks.principal_input_figures import packet
from src.empirical_isopach.core.catalog import datasets


def test_relief_chooses_the_resolution_for_the_frame_span():
    available = inventory()["tiles"]
    for arcseconds, span in ((3, 20), (15, 90), (60, 500)):
        tile = next(tile for tile in available if tile["arcseconds"] == arcseconds
                    and any(p.startswith("pinatubo") for p in tile.get("plates", [])))
        west, south, east, north = tile["bounds"]
        margin = arcseconds / 3600
        lon = np.linspace(west + margin, east - margin, 4)
        lat = np.linspace(south + margin, north - margin, 4)
        valid = np.ones(4, dtype=bool)
        np.testing.assert_allclose(elevation(lon, lat, valid, span), sample(tile, lon, lat), atol=.01)


def test_pinatubo_box_a_groups_the_near_vent_detail_plates():
    record = next(item for item in datasets() if item["slug"] == "pinatubo-1991-layer-c")
    xy, _, panels, manifest, _ = packet(record)
    region = manifest["regional_index"]
    assert region["file"] == "near-vent.png" and region["label"] == "A"
    west, south, east, north = region["bounds_km"]
    inside = ((xy >= [west, south]) & (xy <= [east, north])).all(axis=1)
    children = np.concatenate([panels[i-1].indices for i in region["detail_numbers"]])
    assert 100 < region["observations"] < manifest["n_rows"]
    np.testing.assert_array_equal(np.sort(children), np.flatnonzero(inside))


def test_etopo_crops_capture_land_in_the_source_tiles():
    for plate, minimum_peak in (("b-tm:near-vent.png", 2000), ("minoan:detail-04.png", 100)):
        tile = next(item for item in inventory()["tiles"] if plate in item.get("plates", []))
        assert tile["arcseconds"] == 15
        assert grid(tile["file"], tile["key"]).max() > minimum_peak


def test_global_grid_wraps_and_rendered_water_stays_blue():
    lon = np.array([-180., 180.])
    lat = np.array([65., 65.])
    z = elevation(lon, lat, np.ones(2, dtype=bool), 3000)
    assert z[0] == z[1]

    grid_lon = np.tile(np.linspace(-122.16, -122.06, 6), (7, 1))
    grid_lat = np.tile(np.linspace(42.90, 43.00, 7)[:, None], (1, 6))
    mask = np.ones_like(grid_lon, dtype=np.int8)
    mask[0] = 0
    mask[-1] = -1
    rgb = terrain_rgb(grid_lon, grid_lat, mask, np.arange(6), np.arange(7))
    np.testing.assert_allclose(rgb[0], np.tile([.898, .937, .953], (6, 1)))
    assert np.all(rgb[-1] == 1)
    assert np.ptp(rgb[1:-1, :, 0]) > .025
