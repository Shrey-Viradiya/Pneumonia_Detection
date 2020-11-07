from PIL import Image

import numpy as np


def image_center_crop(path):
    """
    Makes a square center crop of an img, which is a [h, w, 3] numpy array.
    Returns [min(h, w), min(h, w), 3] output with same width and height.
    For cropping use numpy slicing.
    """

    image = Image.open(path)
    img = np.asarray(image)

    shape = img.shape
    dim = len(shape)

    if dim == 3:
        h, w, c = shape
    else:
        h, w = shape

    h_crop = min(h, w)

    if dim == 3:
        cropped_img = img[
            ...,
            (h // 2 - h_crop // 2) : (h // 2 + h_crop // 2),
            (w // 2 - h_crop // 2) : (w // 2 + h_crop // 2),
            :,
        ]
        image = Image.fromarray(cropped_img, "RGB")
    else:
        cropped_img = img[
            ...,
            (h // 2 - h_crop // 2) : (h // 2 + h_crop // 2),
            (w // 2 - h_crop // 2) : (w // 2 + h_crop // 2),
        ]
        image = Image.fromarray(cropped_img)

    image.save(path)
