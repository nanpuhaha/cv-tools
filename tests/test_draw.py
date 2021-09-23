from cv.draw import draw_rectangle_on_image
from cv.io import imread, imwrite


def test_draw_rectangle_on_image():
    image = draw_rectangle_on_image("image/red.png", [3, 3, 8, 8], color=(255, 255, 0))
    image_with_rectangle = imread("image/red_rectangle.png")
    assert (image == image_with_rectangle).all()
