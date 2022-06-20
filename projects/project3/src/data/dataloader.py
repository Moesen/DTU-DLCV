from __future__ import annotations

import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from projects.utils import get_project3_root
from sklearn.model_selection import train_test_split
import numpy as np

# CONSTANTS
AUTOTUNE = tf.data.AUTOTUNE
SUPPORTED_FILETYPES = ["jpg", "png"]

# Have to match image_folder imgs, with mask_imgs
# Use regex pattern to find id for each mask
ID_EXPR = re.compile(r"(^\w*?_\d*)")
SEG_EXPR = re.compile(r"(^\w*?_\d*).*?_(\d+)")


class IsicDataSet(object):
    """Tensorflow implementation of isic dataloader"""
    def __init__(
        self,
        image_folder                 : Path,
        mask_folder                  : Path,
        image_channels               : int,
        mask_channels                : int,
        image_file_extension         : str,
        mask_file_extension          : str,
        do_normalize                 : bool,
        image_size                   : tuple[int, int],
        validation_percentage        : float | None = 0.2,
        output_image_path            : bool  | None = False,
        segmentation_type            : str   | None = None,
        validation_segmentation_type : str   | None = None,
        seed                         : int   | None = None,
        flipping                     : str          = "none",
        rotation                     : float | None = 0,
        brightness                   : float | None = 0,
        contrast                     : float | None = 0,
        saturation                   : float | None = 0,
        hue                          : float | None = 0,
        ) -> None:
        # Check if defined filetypes are supported
        assert (
            mask_file_extension in SUPPORTED_FILETYPES
            and image_file_extension in SUPPORTED_FILETYPES
        )

        # Assignment
        self._image_size = image_size
        self._image_channels = image_channels
        self._mask_channels = mask_channels
        self._seed = seed or random.randint(0, 1000)
        self._do_normalize = do_normalize
        self._train_segmentation_type = segmentation_type
        self._image_folder = image_folder
        self._mask_folder = mask_folder
        self._output_image_path = output_image_path
        self._flipping = flipping  # flipping should be either of ["none", "horizontal", "vertical", "horizontal_and_vertical"]
        self._rotation = rotation  # rotation should be in interval [0, 0.5]
        self._brightness = brightness  # brightness should be in interval [0, 1]
        self._contrast = contrast  # contrast should be in interval [0, 1]
        self._saturation = saturation  # saturation should be in interval [0, ?]
        self._hue = hue  # hue should be in interval [0, 0.5]

        if not validation_segmentation_type and segmentation_type:
            self._validation_segmentation_type = segmentation_type
        else:
            self._validation_segmentation_type = validation_segmentation_type
        # Img decoders
        self._image_decoder = (
            tf.image.decode_jpeg
            if image_file_extension == "jpg"
            else tf.image.decode_png
        )

        self._mask_decoder = (
            tf.image.decode_jpeg
            if mask_file_extension == "jpg"
            else tf.image.decode_png
        )

        # Paths
        img_paths = [x for x in image_folder.iterdir() if "isic" in x.name.lower()]
        train_img_paths, val_img_paths = train_test_split(
            img_paths, test_size=validation_percentage, random_state=self._seed
        )

        self._train_image_paths, self._train_mask_paths = self._match_img_mask(
            train_img_paths, self._train_segmentation_type
        )
        self._test_image_paths, self._test_mask_paths = self._match_img_mask(
            val_img_paths, self._validation_segmentation_type
        )

    def _match_img_mask(
        self, image_paths: list[Path], segmentation_type: str | None
    ) -> tuple[list[str], list[str]]:
        img_paths_paired = []
        mask_paths_paired = []

        for image_path, im_fn in [(x, x.name) for x in image_paths]:
            search = ID_EXPR.search(im_fn)
            img_id = search and search.group(1)
            assert img_id is not None

            for mask_path in self._mask_folder.iterdir():
                # Get filename of path, that is
                # /.../.../.../(name.png) <--- this part
                mask_fn = mask_path.name

                # Search for expression, if not a match: continue
                match = SEG_EXPR.search(mask_fn)
                if match is None:
                    continue

                # Extract id and segmentation type.
                # If not matching ids: continue
                mask_id, seg_type = match.groups()
                if mask_id != img_id:
                    continue

                # If going by segmentation type, and not the right: continue
                if segmentation_type and segmentation_type != seg_type:
                    continue

                img_paths_paired.append(image_path.as_posix())
                mask_paths_paired.append(mask_path.as_posix())

        return img_paths_paired, mask_paths_paired

    def _augmentation_func(
        self, image: tf.Tensor, mask: tf.Tensor, image_path: tf.Tensor = None
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor] | tuple[tf.Tensor, tf.Tensor]:
        # To make sure that the mask and the image are rotated in the same way.

        if self._do_normalize:
            value_range = (0, 1)
        else:
            value_range = (0, 255)

        rotation_augmentation_img = tf.keras.Sequential()
        if self._flipping in ["horizontal", "vertical", "horizontal_and_vertical"]:
            rotation_augmentation_img.add(
                tf.keras.layers.RandomFlip(mode=self._flipping, seed=self._seed)
            )
        rotation_augmentation_img.add(
            tf.keras.layers.RandomRotation(
                self._rotation, fill_mode="constant", seed=self._seed
            )
        )  # rotation should be in interval [0, 0.5]

        rotation_augmentation_mask = tf.keras.Sequential()
        if self._flipping in ["horizontal", "vertical", "horizontal_and_vertical"]:
            rotation_augmentation_mask.add(
                tf.keras.layers.RandomFlip(mode=self._flipping, seed=self._seed)
            )
        rotation_augmentation_mask.add(
            tf.keras.layers.RandomRotation(
                self._rotation, fill_mode="constant", seed=self._seed
            )
        )  # rotation should be in interval [0, 0.5]

        image = rotation_augmentation_img(image)
        mask = rotation_augmentation_mask(mask)

        color_augmentation = tf.keras.Sequential()
        color_augmentation.add(
            tf.keras.layers.RandomBrightness(self._brightness, value_range=value_range)
        )  # brightness should be in interval [0, 1]
        color_augmentation.add(
            tf.keras.layers.RandomContrast(self._contrast)
        )  # contrast should be in interval [0, 1]

        image = color_augmentation(image)
        if self._saturation and self._saturation > 0:
            image = tf.image.random_saturation(
                image, 0, self._saturation
            )  # saturation should be in interval [0, ?]
        image = tf.image.random_hue(
            image, self._hue
        )  # hue should be in interval [0, 0.5]

        if self._output_image_path:
            return image, mask, image_path
        else:
            return image, mask

    def _normalize(
        self, image: tf.Tensor, mask: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.cast(image, tf.float32) / 255.0
        mask = tf.cast(mask, tf.float32) / 255.0
        return image, mask

    def _parse_data(
        self, image_path: str, mask_path: str
    ) -> tuple[tf.Tensor, tf.Tensor] | tuple[tf.Tensor, tf.Tensor, str]:
        """Loads, normalizes and returns the images"""
        image_content = tf.io.read_file(image_path)
        mask_content = tf.io.read_file(mask_path)

        image = self._image_decoder(image_content, channels=self._image_channels)
        mask = self._mask_decoder(mask_content, channels=self._mask_channels)
        image_path = tf.convert_to_tensor(image_path)

        if self._do_normalize:
            image, mask = self._normalize(image, mask)

        if self._output_image_path:
            return image, mask, image_path
        else:
            return image, mask

    def _resize_data(
        self, image: tf.Tensor, mask: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Resizes mask and image to same image size given by _image_size"""
        image = tf.image.resize(image, self._image_size)
        mask = tf.image.resize(mask, self._image_size)
        return image, mask

    def _map_function(
        self,
        image_path: str,
        mask_path: str,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Maps the data"""

        if self._output_image_path:
            image, mask, image_path = self._parse_data(image_path, mask_path)
            return tf.py_function(
                self._augmentation_func,
                [image, mask, image_path],
                [tf.float32, tf.float32, tf.string],
            )
        else:
            image, mask = self._parse_data(image_path, mask_path)
            return tf.py_function(
                self._augmentation_func, [image, mask], [tf.float32, tf.float32]
            )

    def get_dataset(
        self, batch_size: int, shuffle: bool = False
    ) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        - Reads the data
        - Normalizes it if normalize=true
        - #TODO Augments the data

        Returns:
            data: A tf dataset object
        """

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (self._train_image_paths, self._train_mask_paths)
        ).map(self._map_function, num_parallel_calls=AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices(
            (self._test_image_paths, self._test_mask_paths)
        ).map(self._parse_data, num_parallel_calls=AUTOTUNE)

        # fmt: off
        if shuffle:
            train_dataset = (train_dataset
                             .prefetch(AUTOTUNE)
                             .shuffle(self._seed)
                             .batch(batch_size))
        else:
            train_dataset = (train_dataset
                             .batch(batch_size)
                             .prefetch(AUTOTUNE))
        # fmt: on
        test_dataset = test_dataset.batch(batch_size).prefetch(AUTOTUNE)

        return train_dataset, test_dataset


if __name__ == "__main__":
    # Example of using dataloader and extracting datasets train and test
    proot = get_project3_root()
    data_root = proot / "data/train_allstyles"
    image_path = data_root / "Images"
    mask_path = data_root / "Segmentations"
    dataset_loader = IsicDataSet(
        image_folder=image_path,
        mask_folder=mask_path,
        image_size=(256, 256),
        image_channels=3,
        mask_channels=1,
        image_file_extension="jpg",
        mask_file_extension="png",
        segmentation_type="0",
        do_normalize=True,
        output_image_path=False,
        flipping="horizontal_and_vertical",
        rotation=0.5,
        hue=0.5,
    )

    train_dataset, test_dataset = dataset_loader.get_dataset(batch_size=1, shuffle=True)
    if dataset_loader._output_image_path:
        image, mask, image_path = next(iter(test_dataset))
        image_path = image_path.numpy()[0].decode("utf-8")
        _, [a, b] = plt.subplots(1, 2)
        a.imshow(image[0])
        b.imshow(mask[0])
        b.set_title(f"max val: {tf.reduce_max(mask)}, min val: {tf.reduce_min(mask)}")
        plt.show()
    else:
        image, mask = next(iter(test_dataset))
        _, [a, b] = plt.subplots(1, 2)
        a.imshow(image[0])
        b.imshow(mask[0])
        b.set_title(f"max val: {tf.reduce_max(mask)}, min val: {tf.reduce_min(mask)}")
        plt.show()
