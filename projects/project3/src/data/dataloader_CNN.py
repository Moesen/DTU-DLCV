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


class IsicDataSet_cnn(object):
    """Tensorflow implementation of isic dataloader"""

    def __init__(
        self,
        lesions_folder: Path,
        background_folder: Path,
        image_channels: int,
        image_file_extension: str,
        do_normalize: bool,
        image_size: tuple[int, int],
        validation_percentage: float | None = 0.2,
        seed: int | None = None,
        flipping: str | None = "none",
        rotation: float | None = 0,
        brightness: float | None = 0,
        contrast: float | None = 0,
        saturation: float | None = 0,
        hue: float | None = 0
    ) -> None:
        # Check if defined filetypes are supported
        assert (
            image_file_extension in SUPPORTED_FILETYPES
        )

        # Assignment
        self._image_size = image_size
        self._image_channels = image_channels
        self._seed = seed or random.randint(0, 1000)
        self._do_normalize = do_normalize
        self._lesions_folder = lesions_folder
        self._background_folder = background_folder
        self._flipping = flipping #flipping should be either of ["none", "horizontal", "vertical", "horizontal_and_vertical"]
        self._rotation = rotation #rotation should be in interval [0, 0.5]
        self._brightness = brightness #brightness should be in interval [0, 1]
        self._contrast = contrast #contrast should be in interval [0, 1]
        self._saturation = saturation #saturation should be in interval [0, ?]
        self._hue = hue #hue should be in interval [0, 0.5]

        # Img decoders
        self._image_decoder = (
            tf.image.decode_jpeg
            if image_file_extension == "jpg"
            else tf.image.decode_png
        )

        # Paths
        img_paths = [x.as_posix() for x in lesions_folder.iterdir() if "isic" in x.name.lower()]
        classes = [1 for x in img_paths]
        img_paths += [x.as_posix() for x in background_folder.iterdir() if "isic" in x.name.lower()]
        classes += [0 for x in background_folder.iterdir() if "isic" in x.name.lower()]

        self._train_image_paths, self._test_image_paths, self._train_classes, self._test_classes = train_test_split(
            img_paths, classes,  test_size=validation_percentage, random_state=self._seed
        )

    def _augmentation_func(
        self, image: tf.Tensor, image_class: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        # TODO: Implement augmentations
        # Link to someone who already did some augments
        # https://github.com/HasnainRaz/SemSegPipeline/blob/master/dataloader.py
        # His augmentation function is inside the map function, but this is neater
        
        seed = np.random.randint(1e8) #To make sure that the mask and the image are rotated in the same way.
        if self._do_normalize:
            value_range = (0,1)
        else:
            value_range = (0,255)

        rotation_augmentation_img = tf.keras.Sequential()
        if self._flipping in ["horizontal","vertical","horizontal_and_vertical"]:
            rotation_augmentation_img.add(tf.keras.layers.RandomFlip(mode=self._flipping, seed=seed))
        rotation_augmentation_img.add(tf.keras.layers.RandomRotation(self._rotation, fill_mode="constant", seed=seed)) #rotation should be in interval [0, 0.5]

        image = rotation_augmentation_img(image)

        color_augmentation = tf.keras.Sequential()
        color_augmentation.add(tf.keras.layers.RandomBrightness(self._brightness, value_range=value_range)) #brightness should be in interval [0, 1]
        color_augmentation.add(tf.keras.layers.RandomContrast(self._contrast)) #contrast should be in interval [0, 1]
        
        image = color_augmentation(image)
        if self._saturation > 0:
            image = tf.image.random_saturation(image, 0, self._saturation) #saturation should be in interval [0, ?]
        image = tf.image.random_hue(image, self._hue) #hue should be in interval [0, 0.5]

        return image, image_class

    def _normalize(
        self, image: tf.Tensor
    ) -> tuple[tf.Tensor]:
        image = tf.cast(image, tf.float32) / 255.0
        return image

    def _parse_data(
        self, image_path: str, image_class: int
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Loads, normalizes and returns the images"""
        image_content = tf.io.read_file(image_path)

        image = self._image_decoder(image_content, channels=self._image_channels)
        image_class = tf.expand_dims(tf.convert_to_tensor(image_class), 0)

        if self._do_normalize:
            image = self._normalize(image)

        return image, image_class

    def _resize_data(
        self, image: tf.Tensor, mask: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Resizes mask and image to same image size given by _image_size"""
        image = tf.image.resize(image, self._image_size)
        mask = tf.image.resize(mask, self._image_size)
        return image, mask

    def _map_function(
        self, image_path: str, image_class: int,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Maps the data"""

        # TODO: Implement data augmentation
        # This file is heavily inspired by link below, which also
        # implements data augmentation so maybe follow that
        # https://github.com/HasnainRaz/SemSegPipeline/blob/master/dataloader.py

        image, image_class = self._parse_data(image_path, image_class)
        return tf.py_function(
            self._augmentation_func, [image, image_class], [tf.float32, tf.int32]
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
            (self._train_image_paths, self._train_classes)
        ).map(self._map_function, num_parallel_calls=AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices(
            (self._test_image_paths, self._test_classes)
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
    lesions_path = proot / "data/isic/train_allstyles/Images"
    background_path = proot / "data/isic/background"
    dataset_loader = IsicDataSet(
        lesions_folder=lesions_path,
        background_folder=background_path,
        image_size=(256, 256),
        image_channels=3,
        image_file_extension="jpg",
        do_normalize=True
    )

    train_dataset, test_dataset = dataset_loader.get_dataset(batch_size=1, shuffle=True)
    train_iter = iter(train_dataset)
    test_iter = iter(test_dataset)

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Training Set')
    for i in range(4):
        j = 1 if i >= 2 else 0
        image, image_class = next(train_iter)
        axs[i % 2, j].imshow(tf.squeeze(image))
        axs[i % 2, j].set_title(f"Class: {image_class.numpy()}")
    plt.show()

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Test Set')
    for i in range(4):
        j = 1 if i >= 2 else 0
        image, image_class = next(test_iter)
        axs[i % 2, j].imshow(tf.squeeze(image))
        axs[i % 2, j].set_title(f"Class: {image_class.numpy()}")
    plt.show()
