from collections import Counter

import numpy as np

from cv.alpha import (
    contains_alpha_channel,
    convert_alpha_image_to_binary,
    convert_alpha_to_black,
    create_canvas,
    get_non_alpha_colors,
)
from cv.io import imread


def test_get_non_alpha_colors():
    result = {
        "colors": [(17, 17, 255), (51, 51, 51), (17, 34, 136), (0, 17, 221)],
        "color_count": 4,
        "counter": Counter(
            {(51, 51, 51): 20, (0, 17, 221): 12, (17, 17, 255): 12, (17, 34, 136): 8}
        ),
        "most_color": (51, 51, 51),
    }

    assert get_non_alpha_colors("image/red.png") == result


def test_contains_alpha_channel():
    image = imread("image/red.png")
    # path = Path("image/red.png")
    # print(f"{path=}")
    # path = path.resolve()
    # print(f"{path=}")
    assert contains_alpha_channel(image) == True


def test_create_canvas():
    black_canvas = create_canvas(width=5, height=5, rgb_color=(0, 0, 0))
    image = np.zeros((5, 5, 3), np.uint8)
    assert (black_canvas == image).all()


def test_convert_alpha_to_black():
    assert (
        convert_alpha_to_black(image=imread("image/red.png"))
        == imread("image/red_black.png")
    ).all()
    assert (
        convert_alpha_to_black(imagepath="image/red.png")
        == imread("image/red_black.png")
    ).all()


def test_convert_alpha_image_to_binary():
    print(
        "\n2) test_convert_alpha_image_to_binary:\n",
        convert_alpha_image_to_binary("image/red.png")
        == imread("image/red_binary.png"),
    )
    assert (
        convert_alpha_image_to_binary("image/red.png") == imread("image/red_binary.png")
    ).all()
