"""Measured colour corroboration for observed identities.

A vision model names colours from appearance, and appearance sometimes loses to lighting or
a neighbouring object: measured on this dataset, one ground-truth red cube was inventoried
as an orange block, which made the request "pick up the red cube" unresolvable and cost the
instruction. The object's own pixels decide the colour band, and a single-word name the
pixels REFUTE is replaced with the measured band and reported as a warning naming both.

This is a measurement, not a vocabulary: the bands are hue/saturation/value intervals, the
only words involved are the model's own, and an unrecognised or compound name is left
alone. Nothing here chooses a task answer.
"""
from __future__ import annotations

import numpy as np

# Hue intervals in degrees; the achromatic classes come from saturation/value instead.
_BANDS: tuple[tuple[str, float, float], ...] = (
    ("red", 0.0, 15.0), ("orange", 15.0, 45.0), ("yellow", 45.0, 70.0),
    ("green", 70.0, 165.0), ("cyan", 165.0, 200.0), ("blue", 200.0, 255.0),
    ("purple", 255.0, 290.0), ("pink", 290.0, 335.0), ("red", 335.0, 360.0),
)
_ACHROMATIC = frozenset({"white", "gray", "grey", "silver", "black"})
_CHROMATIC = frozenset({"red", "orange", "brown", "yellow", "green", "cyan",
                        "blue", "purple", "pink"})
# A measurement that is neither clearly chromatic nor clearly achromatic cannot refute a
# name, and one taken over too few pixels is noise.
_DECISIVE_SATURATION = 0.25
_ACHROMATIC_SATURATION = 0.12
_MIN_PIXELS = 200
# Brown is the dark orange band: a model calling it "brown" is describing the same pixels.
_BROWN_MAX_VALUE = 0.5
_ACHROMATIC_BLACK_VALUE = 0.25
_ACHROMATIC_WHITE_VALUE = 0.6


def colour_band(pixels: np.ndarray) -> str | None:
    """The band the pixels show, or None when the measurement cannot decide.

    ``pixels`` is an (N, 3) float array in [0, 1] taken from the object's own region.
    """
    if pixels is None or len(pixels) < _MIN_PIXELS:
        return None
    high = pixels.max(axis=1)
    low = pixels.min(axis=1)
    saturation = np.where(high > 0, (high - low) / np.maximum(high, 1e-6), 0.0)
    median_saturation = float(np.median(saturation))
    median_value = float(np.median(high))
    if median_saturation < _ACHROMATIC_SATURATION:
        if median_value < _ACHROMATIC_BLACK_VALUE:
            return "black"
        return "white" if median_value > _ACHROMATIC_WHITE_VALUE else "gray"
    if median_saturation < _DECISIVE_SATURATION:
        return None  # borderline: neither class can refute the model's word
    red, green, blue = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    delta = high - low + 1e-6
    hue = np.where(
        high == red, ((green - blue) / delta) % 6,
        np.where(high == green, (blue - red) / delta + 2, (red - green) / delta + 4),
    ) * 60
    median_hue = float(np.median(hue))
    for name, start, end in _BANDS:
        if start <= median_hue < end:
            if name == "orange" and median_value < _BROWN_MAX_VALUE:
                return "brown"
            return name
    return None


def polygon_pixels(image: np.ndarray, polygon) -> np.ndarray:
    """The image pixels inside a contour, rasterised with the same rule as the masks."""
    from PIL import Image as _Image
    from PIL import ImageDraw

    canvas = _Image.new("1", (image.shape[1], image.shape[0]), 0)
    ImageDraw.Draw(canvas).polygon([tuple(point) for point in polygon], fill=1)
    return image[np.asarray(canvas, dtype=bool)]


def reconcile_colour(name: str, band: str | None) -> str | None:
    """The corrected colour name, or None to keep the model's.

    Only a single word the measurement contradicts is corrected. A compound name
    ("white_and_green") describes structure the band cannot capture, an achromatic word is
    compatible with every achromatic band (silver, grey and white are one family here), and
    a word outside the measured vocabulary is left untouched rather than replaced by a
    coarser guess.
    """
    if band is None:
        return None
    words = [word for word in str(name or "").casefold().replace("-", "_").split("_") if word]
    if len(words) != 1:
        return None
    word = words[0]
    if word not in _ACHROMATIC and word not in _CHROMATIC:
        return None
    if band in _ACHROMATIC:
        return None if word in _ACHROMATIC else band
    if word in _ACHROMATIC:
        return band
    return None if word == band else band
