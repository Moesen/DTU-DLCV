from __future__ import annotations

from typing import Union

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from projects.utils import get_project11_root


def load_dataset(
    train: bool = True,
    batch_size: int = 32,
    normalize: bool = True,
    image_size: tuple = (32, 32),
    validation_split: float | None = None,  # Good value could be .2
    shuffle: bool = False,
    crop_to_aspect_ratio: bool = False,
    tune_for_perfomance: bool = False,
    use_data_augmentation: bool = False,
    augmentation_flip: str = "horizontal_and_vertical",
    augmentation_rotation: float = 0.5,
    augmentation_contrast: float = 0.5,
    **kwargs,
) -> tf.data.Dataset | tuple[tf.data.Dataset, tf.data.Dataset]:

    # Assert we don't do validation split if dataset is test
    assert validation_split == None or train == True

    proot_path = get_project11_root()
    path = proot_path / "data/hotdog_nothotdog"

    if not path.is_dir():
        raise NotADirectoryError(
            """
            /data/hotdog_nothotdog directory not found in root of project. 
            Either use `make data_local` or download dataset otherwise from hpc.
            """
        )

    dp = path / "train" if train else path / "test"
    seed = np.random.randint(0, 1000)

    main_dataset = tf.keras.utils.image_dataset_from_directory(
        dp,
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        color_mode="rgb",
        shuffle=shuffle,
        seed=seed,
        validation_split=validation_split,
        subset="training" if validation_split is not None else None,
        crop_to_aspect_ratio=crop_to_aspect_ratio,
        **kwargs,
    )

    # Sequential layer of preprocessing applied at mapping stage
    map_layers = tf.keras.Sequential()

    # Normalizing
    if normalize:
        normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
        map_layers.add(normalization_layer)

    # Augmentation
    if use_data_augmentation:
        augmentation_layer = tf.keras.Sequential(
            [
                layers.RandomRotation(augmentation_rotation),
                layers.RandomContrast(augmentation_contrast),
            ]
        )
        if augmentation_flip.lower() != "none":
            augmentation_layer.add(layers.RandomFlip(augmentation_flip))
        map_layers.add(augmentation_layer)

    # Mapping
    main_dataset = main_dataset.map(lambda x, y: (map_layers(x), y))

    # Tuning
    if tune_for_perfomance:
        AUTOTUNE = tf.data.AUTOTUNE
        main_dataset = main_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    if validation_split is None:
        return main_dataset
    else:
        val_dataset = tf.keras.utils.image_dataset_from_directory(
            dp,
            label_mode="int",
            image_size=image_size,
            batch_size=batch_size,
            color_mode="rgb",
            seed=seed,
            validation_split=validation_split,
            subset="validation" if validation_split is not None else None,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            **kwargs,
        )
        return main_dataset, val_dataset


if __name__ == "__main__":
    ts = load_dataset(train=True, batch_size=64, shuffle=True, image_size=(128, 128))
    class_names = ts._input_dataset.class_names  # type: ignore

    img_batch, label = next(iter(ts))
    fig = plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        ax.imshow(img_batch[i])
        if type(class_names) == list:
            ax.set_title(class_names[label[i]])  # type: ignore
    plt.show()
