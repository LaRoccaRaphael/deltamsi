import numpy as np

from pymsix.processing.kendrick import _parse_formula_to_mass, kendrick_coords


def test_parse_formula_simple_grouping():
    exact, nominal = _parse_formula_to_mass("CH2")
    # Exact mass of CH2: 12 (C) + 2 * 1.00782503223 (H)
    np.testing.assert_allclose(exact, 14.01565006446)
    assert nominal == 14


def test_kendrick_coords_with_string_base():
    masses = [14.01565]
    coords = kendrick_coords(masses, base="CH2")
    # Ensure scale and KM are computed without raising parsing errors
    assert "KM" in coords and "KMD_fraction" in coords
    np.testing.assert_allclose(coords["base_exact"], 14.01565006446)
    np.testing.assert_equal(coords["base_nominal"], 14)
