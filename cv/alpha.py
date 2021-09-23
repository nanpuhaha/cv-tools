from collections import Counter

import cv2
import numpy as np

from cv.io import imread


class NotExistAlphaChannel(Exception):
    pass


def get_non_alpha_colors(imagepath: str):
    image = imread(imagepath)

    if not contains_alpha_channel(image):
        return NotExistAlphaChannel(f"'{imagepath}' file doesn't contain alpha channel")

    height, width, _ = image.shape
    non_alpha_colors = []
    for x in range(width):
        for y in range(height):
            b, g, r, alpha = image[y, x]  # b, g, r, alpha
            if alpha == 255:
                non_alpha_colors.append((b, g, r))

    return {
        "colors": list(set(non_alpha_colors)),
        "color_count": len(set(non_alpha_colors)),  # len(Counter(non_alpha_colors))
        "counter": Counter(non_alpha_colors),
        "most_color": Counter(non_alpha_colors).most_common(1)[0][0],
    }


def contains_alpha_channel(image: np.ndarray):
    """Check image contains alpha channel"""

    if not isinstance(image, np.ndarray):
        raise ValueError(f"image is NOT np.ndarray: {type(image)}")

    try:
        if len(image.shape) == 3 and image.shape[-1] == 4:
            return True

        return False

    except AttributeError as err:
        raise AttributeError(f"image is None, {err}")


def create_canvas(width: int, height: int, rgb_color: tuple = (0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""

    # Create black canvas image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))

    # Fill image with color
    image[:] = color

    return image


def convert_alpha_to_black(image: np.ndarray = None, imagepath: str = None):
    if image is None and imagepath is None:
        raise ValueError(f"image or imagepath is necessary")
    elif imagepath is not None:
        image = imread(imagepath)

    if not contains_alpha_channel(image):
        print(f"image doesn't contain alpha channel")
        return image

    # # Method 1) Vanila Loop
    # height, width, _ = image.shape
    # black_canvas = create_canvas(width, height, rgb_color=(0, 0, 0))
    # for x in range(width):
    #     for y in range(height):
    #         b, g, r, alpha = image[y, x]
    #         if alpha == 255:
    #             black_canvas[y, x] = (b, g, r)

    # Method 2) Numpy Vectorization
    black_canvas = image.copy()
    alpha = np.where(black_canvas[:, :, 3] != 255)
    black_canvas[alpha] = (0, 0, 0, 255)
    black_canvas = black_canvas[:, :, :3]
    print(f"[ black_canvas.shape = {black_canvas.shape} ]")

    return black_canvas


def convert_alpha_image_to_binary(image_path: str):
    """convert alpha to black, non-alpha to white"""

    image = imread(image_path)

    if not contains_alpha_channel(image):
        print(f"image doesn't contain alpha channel")
        return image

    # # Method 1) Black Canvas
    # height, width, _ = image.shape
    # black_canvas = create_canvas(width, height, rgb_color=(0, 0, 0))
    # for x in range(width):
    #     for y in range(height):
    #         _, _, _, alpha = image[y, x]  # b, g, r, alpha
    #         if alpha == 255:
    #             black_canvas[y, x] = (255, 255, 255)
    # return black_canvas

    # Method 2) Threshold
    _, mask = cv2.threshold(image[:, :, 3], 254, 255, cv2.THRESH_BINARY)

    return mask
