from __future__ import annotations

import tensorflow as tf
from pathlib import Path
from projects.utils import get_project3_root
import re
import random
import matplotlib.pyplot as plt

# CONSTANTS
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
        do_normalize: bool,
        image_size: tuple[int, int],
        segmentation_type: str | None = None,
        seed: int | None = None,
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
        self._segmentation_type = segmentation_type

        # Have to match image_folder imgs, with mask_imgs
        # Use regex pattern to find id for each mask
        id_expr = re.compile(r"(^\w*?_\d*)")
        seg_expr = re.compile(r"(^\w*?_\d*).*?_(\d+)")

        self._image_paths: list[str] = []
        self._mask_paths: list[str] = []

        for im_path, im_fn in [
            (x, x.name) for x in image_folder.iterdir() if "ISIC" in x.name
        ]:
            img_id = id_expr.findall(str(im_fn))

            assert len(img_id) == 1

            img_id = img_id[0]
            for mask_path in mask_folder.iterdir():
                mask_name = mask_path.name

                # Search for expression, if not a match: continue
                match = seg_expr.search(mask_name)
                if match == None:
                    continue

                # Extract id and segmentation type.
                # If not matching ids: continue
                mask_id, seg_type = match.groups()
                if mask_id != img_id:
                    continue

                # If going by segmentation type, and not the right: continue
                if self._segmentation_type and self._segmentation_type != seg_type:
                    continue

                self._image_paths.append(im_path.as_posix())
                self._mask_paths.append(mask_path.as_posix())


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

    def _augmentation_func(
        self, image: tf.Tensor, mask: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        # TODO: Implement augmentations
        # Link to someone who already did some augments
        # https://github.com/HasnainRaz/SemSegPipeline/blob/master/dataloader.py
        # His augmentation function is inside the map function, but this is neater

        return image, mask
        
        
    def _normalize(
        self, image: tf.Tensor, mask: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.cast(image, tf.float32) / 255.0
        mask = tf.cast(mask, tf.float32) / 255.0
        return image, mask

    def _parse_data(
        self, image_path: str, mask_path: str
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Loads, normalizes and returns the images"""
        image_content = tf.io.read_file(image_path)
        mask_content = tf.io.read_file(mask_path)

        image = self._image_decoder(image_content, channels=self._image_channels)
        mask = self._mask_decoder(mask_content, channels=self._mask_channels)

        if self._do_normalize:
            image, mask = self._normalize(image, mask)

        return image, mask

    def _resize_data(
        self, image: tf.Tensor, mask: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Resizes mask and image to same image size given by _image_size"""
        image = tf.image.resize(image, self._image_size)
        mask = tf.image.resize(mask, self._image_size)
        return image, mask

    def _map_function(
        self, image_path: str, mask_path: str
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Maps the data"""

        # TODO: Implement data augmentation
        # This file is heavily inspired by link below, which also
        # implements data augmentation so maybe follow that
        # https://github.com/HasnainRaz/SemSegPipeline/blob/master/dataloader.py

        image, mask = self._parse_data(image_path, mask_path)
        return tf.py_function(self._augmentation_func, [image, mask], [tf.float32, tf.float32])

    def get_dataset(self, batch_size: int, shuffle: bool = False) -> tf.data.Dataset:
        """
        - Reads the data
        - Normalizes it if normalize=true
        - #TODO Augments the data

        Returns:
            data: A tf dataset object
        """
        dataset = tf.data.Dataset.from_tensor_slices(
            (self._image_paths, self._mask_paths)
        )
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
        do_normalize=True,
        segmentation_type="0",
    )

    print(len(dataset_loader._image_paths), len(dataset_loader._mask_paths))
    dataset = dataset_loader.get_dataset(batch_size=1, shuffle=True)

    image, mask = next(iter(dataset)) 
    _, [a, b] = plt.subplots(1, 2)
    a.imshow(image[0])
    b.imshow(mask[0])
    b.set_title(f"max val: {tf.reduce_max(mask)}, min val: {tf.reduce_min(mask)}")
    plt.show()
