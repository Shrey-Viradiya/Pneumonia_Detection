def image_center_crop(img, rgb_channels=True, batched=True):
    """
    Makes a square center crop of an img, which is a [h, w, 3] numpy array.
    Returns [min(h, w), min(h, w), 3] output with same width and height.
    For cropping use numpy slicing.
    """
    shape = img.shape
    if channels and batched:
        b, h, w, c = shape
    elif channels:
        h, w, c = shape
    elif batched:
        b, h, w = shape
    else:
        h, w = shape

    h_crop = min(h,w)

    if channels:
        cropped_img = img[..., (h//2-h_crop//2):(h//2+h_crop//2), (w//2-h_crop//2):(w//2+h_crop//2), :]
    else:
        cropped_img = img[..., (h//2-h_crop//2):(h//2+h_crop//2), (w//2-h_crop//2):(w//2+h_crop//2)]

    return cropped_img

