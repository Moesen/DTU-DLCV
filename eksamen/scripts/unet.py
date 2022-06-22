import numpy as np


def unet_pixel_indexs(output_pixel: np.ndarray = np.array([5, 6])) -> None:
    # Input shape
    input_shape = np.array([514, 514])

    # x, y, padding
    kernel = np.array([3, 3, 0])

    # x, y, stride
    max_pool = np.array([2, 2, 2])

    # Something with up
    up_conv = np.array([2, 2, 0])

    depth = 2

    shape = input_shape.copy()
    for _ in range(depth):
        # conv
        shape -= kernel[:-1] - 1
        # down
        shape = shape / max_pool[:-1]

    # conv
    shape -= kernel[:-1] - 1

    for _ in range(depth):
        # up
        shape = shape * up_conv[:-1]
        # conv
        shape -= kernel[:-1] - 1
    diff = input_shape - shape

    print(f"shape: \t {input_shape} -> {shape}")
    print(f"pixel: \t {output_pixel + diff/2} <- {output_pixel}")

if __name__ == "__main__":
    unet_pixel_indexs()
