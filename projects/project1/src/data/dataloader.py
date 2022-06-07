from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers
from src.utils import get_project_root
import matplotlib.pyplot as plt
from typing import Union


def load_dataset(
    train: bool = True,
    batch_size: int = 32,
    normalize: bool = True,
    image_size: tuple = (32, 32),
    shuffle: bool = False,
    crop_to_aspect_ratio: bool = False,
    tune_for_perfomance: bool = False,
    use_data_augmentation: bool = True,
    augmentation_flip: str = "horizontal_and_vertical",
    augmentation_rotation: float = 0.5,
    augmentation_contrast: float = 0.5,
    **kwargs,
) -> tf.data.Dataset:
    """
    Small note: To get class_names do dataset._input_dataset.class_names
    unless normalize = False, then just do dataset.class_names
    """
    proot_path = get_project_root()
    path = proot_path / "data/hotdog_nothotdog"

    if not path.is_dir():
        raise NotADirectoryError(
            """
                /data/hotdog_nothotdog directory not found in root of project. 
                Either use `make data_local` or download dataset otherwise from hpc
                """
        )

    dp = path / "train" if train else path / "test"
    dataset = tf.keras.utils.image_dataset_from_directory(
        dp,
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        color_mode="rgb",
        shuffle=shuffle,
        crop_to_aspect_ratio=crop_to_aspect_ratio,
        **kwargs,
    )

    if normalize and use_data_augmentation:
        normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
        augmentation_layer = tf.keras.Sequential([
            layers.RandomFlip(augmentation_flip),
            layers.RandomRotation(augmentation_rotation),
            layers.RandomContrast(augmentation_contrast)
            ])
        dataset = dataset.map(lambda x, y: (augmentation_layer(normalization_layer(x)), y))

    elif normalize:
        normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
        dataset = dataset.map(lambda x, y: (normalization_layer(x), y))

    elif use_data_augmentation:
        augmentation_layer = tf.keras.Sequential([
            layers.RandomFlip(augmentation_flip),
            layers.RandomRotation(augmentation_rotation),
            layers.RandomContrast(augmentation_contrast)
            ])
        dataset = dataset.map(lambda x, y: (augmentation_layer(x), y))

    if tune_for_perfomance:
        AUTOTUNE = tf.data.AUTOTUNE
        dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
    
    return dataset


if __name__ == "__main__":
    ts = load_dataset(train=True, batch_size=64, shuffle=True, image_size=(128, 128))
    class_names = ts._input_dataset.class_names # type: ignore

    img_batch, label = next(iter(ts))
    fig = plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        ax.imshow(img_batch[i])
        if type(class_names) == list:
            ax.set_title(class_names[label[i]])  # type: ignore
    plt.show()
