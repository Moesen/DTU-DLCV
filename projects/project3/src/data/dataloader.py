from __future__ import annotations

import tensorflow as tf
from pathlib import Path
from projects.utils import get_project3_root
import re
import random
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.AUTOTUNE
SUPPORTED_FILETYPES = ["jpg", "png"]


class IsicDataSet(object):
    """Tensorflow implementation of isic dataloader"""

    def __init__(
        self,
        image_folder: Path,
        mask_folder: Path,
        image_channels: int,
        mask_channels: int,
        image_file_extension: str,
        mask_file_extension: str,
        normalize: bool,
        image_size: tuple[int, int] | None,
        segment_type: int | None,
        seed: int | None = None,
    ):
        # Assignment
        self._image_size = image_size
        self._image_channels = image_channels
        self._mask_channels = mask_channels
        self._seed = seed or random.randint(0, 1000)
        self._do_normalize = normalize

        # Have to match image_folder imgs, with mask_imgs
        # Use regex pattern to find id for each mask
        id_expr = re.compile(r"^\w*?_\d*")

        self._image_paths: list[str] = []
        self._mask_paths: list[str] = []

        for img_path in [x for x in image_folder.iterdir() if "ISIC" in x.name]:
            img_filename = img_path.name
            img_id = id_expr.findall(str(img_filename))

            if len(img_id) != 1:
                raise ValueError(img_filename)

            img_id = img_id[0]
            mask_pairs = [
                mask_path
                for mask_path in mask_folder.iterdir()
                if img_id in mask_path.name
            ]
            for mask_path in mask_pairs:
                self._image_paths.append(img_path.as_posix())
                self._mask_paths.append(mask_path.as_posix())

        if (
            mask_file_extension not in SUPPORTED_FILETYPES
            or image_file_extension not in SUPPORTED_FILETYPES
        ):
            raise ValueError(
                f"""File extensions has to be .jpg or .png
                    Currently:
                        {image_file_extension = }
                        {mask_file_extension = }"""
            )

        self._image_decoder = (
            tf.image.decode_jpeg
            if image_file_extension == ".jpg"
            else tf.image.decode_png
        )

        self._mask_decoder = (
            tf.image.decode_jpeg
            if image_file_extension == ".jpg"
            else tf.image.decode_png
        )

    def _normalize(
        self, image: tf.Tensor, mask: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.cast(image, tf.float32) / 255.0
        mask = tf.cast(mask, tf.float32) / 255.0
        return image, mask

    def _parse_data(self, image_path: str, mask_path: str):
        """Loads, normalizes and returns the images"""
        image_content = tf.io.read_file(image_path)
        mask_content = tf.io.read_file(mask_path)

        image = self._image_decoder(image_content, channels=self._image_channels)
        mask = self._mask_decoder(mask_content, channels=self._mask_channels)

        if self._do_normalize:
            image, mask = self._normalize(image, mask)

        return image, mask

    def _resize_data(self, image, mask):
        """Resizes mask and image to same image size given by _image_size"""
        image = tf.image.resize(image, self._image_size)
        mask = tf.image.resize(mask, self._image_size)
        return image, mask

    def _map_function(self, image_path: str, mask_path: str):
        """Maps the data"""
        # TODO: Possibly implement data augmentation
        # This file is heavily inspired by link below, which also
        # implements data augmentation so maybe follow that
        # https://github.com/HasnainRaz/SemSegPipeline/blob/master/dataloader.py
        image, mask = self._parse_data(image_path, mask_path)
        return image, mask

    def get_dataset(self, batch_size: int, shuffle: bool = False) -> tf.data.Dataset:
        """
        - Reads the data
        - Normalizes it if normalize=true
        - #TODO Augments the data

        Returns:
            data: A tf dataset object
        """
        dataset = tf.data.Dataset.from_tensor_slices((self._image_paths, self._mask_paths))
        dataset = dataset.map(self._map_function, num_parallel_calls=AUTOTUNE)

        if shuffle:
            dataset = dataset.prefetch(AUTOTUNE).shuffle(self._seed).batch(batch_size)
        else:
            dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
        
        return dataset

if __name__ == "__main__":
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
        normalize=True,
    )

    dataset = dataset_loader.get_dataset(batch_size=1, shuffle=True)
