from typing import Union

import cv2

from cv.io import imread


def draw_rectangle_on_image(
    image_path: str, box: Union[list, tuple], color=(0, 0, 255)
):
    image = imread(image_path)

    if len(image.shape) == 2:
        # grayscale
        color = (255,)
    else:
        # RGB or RGBA
        channel = image.shape[2]
        if channel == 4:
            color = (*color, 255)

    image_with_rectangle = image.copy()
    cv2.rectangle(
        image_with_rectangle,
        (box[0], box[1]),
        (box[2], box[3]),
        color,
        thickness=1,
    )

    return image_with_rectangle
