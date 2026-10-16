"""Measured colour corroboration: the pixels decide, and only a refuted name changes."""
from __future__ import annotations

import numpy as np
import pytest

from grounded_intent.colour import colour_band, reconcile_colour


def pixels(hue: float, saturation: float, value: float, count: int = 900) -> np.ndarray:
    """An RGB block whose median hue/saturation/value are the arguments."""
    import colorsys

    red, green, blue = colorsys.hsv_to_rgb(hue / 360.0, saturation, value)
    return np.tile(np.array([red, green, blue], dtype=np.float32), (count, 1))


def test_the_measured_band_separates_the_dataset_colours():
    # Every true red in the benchmark measures <= 10.2 deg and every orange >= 18.0 deg.
    assert colour_band(pixels(5.0, 0.7, 0.6)) == "red"
    assert colour_band(pixels(10.2, 0.8, 0.6)) == "red"
    assert colour_band(pixels(18.0, 0.95, 0.6)) == "orange"
    assert colour_band(pixels(54.0, 0.7, 0.55)) == "yellow"
    assert colour_band(pixels(75.0, 0.6, 0.55)) == "green"
    assert colour_band(pixels(20.0, 0.8, 0.4)) == "brown"
    # Achromatic pixels are decided by value, and a borderline saturation decides nothing.
    assert colour_band(pixels(0.0, 0.05, 0.8)) == "white"
    assert colour_band(pixels(0.0, 0.03, 0.5)) == "gray"
    assert colour_band(pixels(0.0, 0.04, 0.1)) == "black"
    assert colour_band(pixels(20.0, 0.18, 0.6)) is None
    assert colour_band(pixels(5.0, 0.7, 0.6, count=50)) is None


@pytest.mark.parametrize("name, band, expected", [
    # The measured mislabel: a ground-truth red cube inventoried as an orange block.
    ("orange", "red", "red"),
    ("red", "red", None),
    # One achromatic family: silver, grey and white are not refuted by each other.
    ("silver", "gray", None),
    ("silver", "white", None),
    ("gray", "black", None),
    ("white", "red", "red"),
    # Compound and unrecognised names describe more (or other) than the band can refute.
    ("white_and_green", "green", None),
    ("teal", "blue", None),
    ("", "red", None),
    ("red", None, None),
])
def test_only_a_refuted_single_word_is_corrected(name, band, expected):
    assert reconcile_colour(name, band) == expected
