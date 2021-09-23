import os
from pathlib import Path

from cv.io import imread, imwrite, makedirs_if_not_exists


def test_imread():
    img = imread("image/빨강.png")
    assert img is not None


def test_imwrite():
    img = imread("image/red.png")
    output_file = "image/red_한글.png"
    imwrite(output_file, img)
    path = Path(output_file)
    result = path.exists()
    path.unlink()
    assert result


def test_makedirs_if_not_exists():
    temp_dir = "temp/temp2/temp3/"
    path = Path(temp_dir)
    before = path.exists()
    makedirs_if_not_exists(temp_dir)
    after = path.exists()
    print(f"{before=} {after=}")
    os.removedirs(temp_dir)
    assert after is True
