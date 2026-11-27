"""Check that terrain sampling and water masking match the cartography."""
import numpy as np

from benchmarks.observation_relief import elevation, inventory, sample, terrain_rgb


def test_fine_grid_covers_crater_lake_details():
    lon, lat = np.array([-122.15, -122.1]), np.array([42.9, 42.95])
    tile = next(tile for tile in inventory()["tiles"] if "mazama:detail-06.png" in tile.get("plates", []))
    control = sample(tile, lon, lat)
    assert np.all(control > 1500)
    np.testing.assert_array_equal(elevation(lon, lat, np.ones(2, dtype=bool), 20), control)


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
