from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers
from src.utils import get_project_root
import matplotlib.pyplot as plt
import numpy as np
from typing import Union
import json
import pandas as pd

def make_data_splits(dataset_json, train_data_amount, validation_data_amount, test_data_amount, datapath, seed=800):
    np.random.seed(seed)
    image_paths = [i["file_name"] for i in dataset_json["images"]]
    image_ids = [i["id"] for i in dataset_json["images"]]
    image_annotations_bbox = []
    image_annotations_label = []
    image_annotations_paths = []
    image_annotations_id = []
    catid2supercat = {i["id"]:i["supercategory"] for i in dataset_json["categories"]}
    all_super_cat = list(set(i["supercategory"] for i in dataset_json["categories"]))
    for a in dataset_json["annotations"]:
        image_id = a["image_id"]
        image_id_idx = image_ids.index(image_id)
        super_cat = catid2supercat[a["category_id"]]
        image_annotations_id.append(image_id)
        image_annotations_bbox.append(a["bbox"])
        image_annotations_label.append(super_cat)
        image_annotations_paths.append(image_paths[image_id_idx])

    image_annotations_bbox = np.array(image_annotations_bbox)
    image_annotations_label = np.array(image_annotations_label)
    image_annotations_paths = np.array(image_annotations_paths)
    image_annotations_id = np.array(image_annotations_id)

    image_categories = [image_annotations_label[image_annotations_paths == i] for i in image_paths]

    train_images = {"images":[]}
    validation_images = {"images":[]}
    test_images = {"images":[]}

    train_categories = np.zeros(len(all_super_cat))
    validation_categories = np.zeros(len(all_super_cat))
    test_categories = np.zeros(len(all_super_cat))
    
    dist_weight = 3

    for i, img in enumerate(image_paths):
        scores = np.zeros(3)
        categories = image_categories[i]
        category_indicies = [all_super_cat.index(i) for i in categories]
        img_bbox = image_annotations_bbox[image_annotations_paths == img]
        if sum(train_categories[category_indicies] == 0) > 0:
            for j in category_indicies:
                train_categories[j] += 1
            train_images["images"].append({"path":img, "id":image_ids[i], "bboxs":img_bbox.tolist(), "labels":categories.tolist()})
        else:
            scores[0] += len(train_images["images"]) / train_data_amount
            scores[1] += len(validation_images["images"]) / validation_data_amount
            scores[2] += len(test_images["images"]) / test_data_amount
            scores[0] += sum(train_categories[category_indicies] / (len(train_images["images"])+0.01)) * i / dist_weight
            scores[1] += sum(validation_categories[category_indicies] / (len(validation_images["images"])+0.01)) * i / dist_weight
            scores[2] += sum(test_categories[category_indicies] / (len(test_images["images"])+0.01)) * i / dist_weight
            min_score_idx = np.random.choice(np.flatnonzero(scores == scores.min()))
            if min_score_idx == 0:
                for j in category_indicies:
                    train_categories[j] += 1
                train_images["images"].append({"path":img, "id":image_ids[i], "bboxs":img_bbox.tolist(), "labels":categories.tolist()})
            elif min_score_idx == 1:
                for j in category_indicies:
                    validation_categories[j] += 1
                validation_images["images"].append({"path":img, "id":image_ids[i], "bboxs":img_bbox.tolist(), "labels":categories.tolist()})
            else:
                for j in category_indicies:
                    test_categories[j] += 1
                test_images["images"].append({"path":img, "id":image_ids[i], "bboxs":img_bbox.tolist(), "labels":categories.tolist()})
    

    split_name = ["train","validation","test"]
    for i, split in enumerate([train_images, validation_images, test_images]):
        with open(f'{datapath}/{split_name[i]}_data.json', 'w') as fp:
            json.dump(split, fp)


def load_dataset_rcnn(
    batch_size: int = 32,
    normalize: bool = True,
    image_size: tuple = (32, 32),
    split: str = "train",
    tune_for_perfomance: bool = False,
    use_data_augmentation: bool = False,
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
    path = proot_path / "data/data_wastedetection"

    with open(path / f"{split}_data.json", 'r') as f:
        data_json = json.loads(f.read())

    data = data_json["images"]

    dataset = tf.data.Dataset.from_tensor_slices((image_annotations_paths, image_annotations_bbox, image_annotations_label))

    def read_image(image_file, bbox, labels):
        image = tf.io.read_file(str(path) + "/" + image_file)
        image = tf.image.decode_image(image, channels=3, dtype=tf.float32)
        image = tf.image.crop_to_bounding_box(image, int(bbox[1]), int(bbox[0]), int(bbox[3]), int(bbox[2]))
        image = tf.image.resize(image, image_size)
        return image, labels
    
    map_layers = tf.keras.Sequential()

    if normalize:
        normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
        map_layers.add(normalization_layer)
    
    if use_data_augmentation:
        augmentation_layer = tf.keras.Sequential([
            layers.RandomRotation(augmentation_rotation),
            layers.RandomContrast(augmentation_contrast)
            ])
        if augmentation_flip.lower() != "none":
            augmentation_layer.add(layers.RandomFlip(augmentation_flip))
        map_layers.add(augmentation_layer)

    dataset = dataset.map(read_image).map(lambda x, y: (map_layers(x), y)).batch(batch_size=batch_size)

    if tune_for_perfomance:
        AUTOTUNE = tf.data.AUTOTUNE
        dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
    
    return dataset


if __name__ == "__main__":
    proot_path = get_project_root()
    path = proot_path / "data/data_wastedetection"

    with open(path / "annotations.json", 'r') as f:
        data_json = json.loads(f.read())

    # make_data_splits(data_json, 0.7, 0.15, 0.15, path)
    ts = load_dataset_rcnn(train=True, batch_size=64, shuffle=True, image_size=(128, 128))
    class_names = ts._input_dataset.class_names # type: ignore

    img_batch, label = next(iter(ts))
    fig = plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        ax.imshow(img_batch[i])
        if type(class_names) == list:
            ax.set_title(class_names[label[i]])  # type: ignore
    plt.show()
