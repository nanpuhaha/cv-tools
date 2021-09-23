from cv.crop import compute_area, find_largest_rectangle, reset_variables
from cv.io import imread


def test_compute_area():
    assert compute_area(10, 2, 8, 4) == 32


def test_find_largest_rectangle():
    image = imread("image/red.png")
    assert find_largest_rectangle(image) == [10, 2, 11, 8]


def test_reset_variables():
    var = {
        "coord_upper_left": {"x": 0, "y": 0},
        "coord_lower_left": {"x": 0, "y": 10},
        "coord_upper_right": {"x": 10, "y": 0},
        "coord_lower_right": {"x": 10, "y": 10},
        "upper_left": True,
        "lower_left": True,
        "unhindered": ["empty"],
    }
    reset_var = {
        "coord_upper_left": {"x": 0, "y": 0},
        "coord_lower_left": {"x": 0, "y": 0},
        "coord_upper_right": {"x": 0, "y": 0},
        "coord_lower_right": {"x": 0, "y": 0},
        "upper_left": False,
        "lower_left": False,
        "unhindered": [],
    }
    reset_variables(var)
    assert var == reset_var
