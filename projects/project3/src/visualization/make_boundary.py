import numpy as np
import cv2

from PIL import Image


def get_boundary(mask, is_GT=False):

    if is_GT:
        color = (0, 255, 0)
    else:
        color = (255, 0, 0)

    contours, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    out = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    k = -1
    for i, cnt in enumerate(contours):
        if (hier[0, i, 3] == -1):
            k += 1
        cv2.drawContours(out, [cnt], -1, color, 2)

    #cv2.imshow('out', out)
    out2 = Image.fromarray(out).convert("L")
    e = np.asarray(out2)

    return out, e