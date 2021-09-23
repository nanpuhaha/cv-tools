import os
from pathlib import Path
from typing import Union

import cv2
import numpy as np


class NoneImageError(Exception):
    pass


def imread(
    filepath: Union[str, Path],
    flags: int = cv2.IMREAD_UNCHANGED,
    dtype: float = np.uint8,
):
    """cv2.imread alternative for Korean filepath"""
    try:
        n = np.fromfile(filepath, dtype)
        img = cv2.imdecode(n, flags)

        if img is None:
            raise NoneImageError(f"{filepath} is None image")

        return img

    except Exception as err:
        raise err


def makedirs_if_not_exists(filepath: str):
    dir = os.path.dirname(filepath)
    if not os.path.exists(dir):
        os.makedirs(dir)


def imwrite(filepath: str, img: np.ndarray, params=None):
    """cv2.imwrite alternative for Korean filepath"""
    try:
        extension = os.path.splitext(filepath)[1]  # png, jpg, ..
        retval, buf = cv2.imencode(extension, img, params)

        if retval:
            makedirs_if_not_exists(filepath)
            with open(filepath, mode="wb") as f:
                buf.tofile(f)
                return True
        else:
            return False

    except Exception as e:
        # TODO: classify exceptions
        raise e
