from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from PIL import ExifTags, Image
from projects.utils import get_project12_root


def make_data_splits(
    dataset_json: dict,
    train_data_amount: float,
    validation_data_amount: float,
    test_data_amount: float,
    datapath: Path,
    seed: int = 800,
) -> None:
    """
    Creates random split to train, test and test datasets based on
    *_amount parameters and seed
    """
    np.random.seed(seed)
    #Getting all image paths and ids
    image_paths = [i["file_name"] for i in dataset_json["images"]]
    image_ids = [i["id"] for i in dataset_json["images"]]

    #Initializing arrays to save annotations and attributes for annotations
    image_annotations_bbox = []
    image_annotations_label = []
    image_annotations_paths = []
    image_annotations_id = []

    #Finding supercategories
    catid2supercat = {i["id"]: i["supercategory"] for i in dataset_json["categories"]}
    all_super_cat = list(set(i["supercategory"] for i in dataset_json["categories"]))
    all_super_cat.append("Background")

    #Saving dictionaries to map index and category
    cat2id = {cat:i for i, cat in enumerate(all_super_cat)}
    id2cat = {value:key for key, value in cat2id.items()}
    with open(f"{datapath}/cat2id.json", "w") as fp:
        json.dump(cat2id, fp, indent=2)
    with open(f"{datapath}/id2cat.json", "w") as fp:
        json.dump(id2cat, fp, indent=2)

    #Saving annotations and their attributes
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

    image_categories = [
        image_annotations_label[image_annotations_paths == i] for i in image_paths
    ]

    #Initializing train, val, test splits
    train_images = {"images": []}
    validation_images = {"images": []}
    test_images = {"images": []}

    train_categories = np.zeros(len(all_super_cat))
    validation_categories = np.zeros(len(all_super_cat))
    test_categories = np.zeros(len(all_super_cat))

    dist_weight = 3

    #Distirbuting images so that the distribution of categories is as good as possible while having the splits defined at start. 
    #This is done by scores which is calculated from the current length and current amount of the classes already in the split.
    for i, img in enumerate(image_paths):
        scores = np.zeros(3)
        categories = image_categories[i]
        category_indicies = [all_super_cat.index(i) for i in categories]
        img_bbox = image_annotations_bbox[image_annotations_paths == img]
        if sum(train_categories[category_indicies] == 0) > 0:
            for j in category_indicies:
                train_categories[j] += 1
            train_images["images"].append(
                {
                    "path": img,
                    "id": image_ids[i],
                    "bboxs": img_bbox.tolist(),
                    "labels": categories.tolist(),
                }
            )
        else:
            scores[0] += len(train_images["images"]) / train_data_amount
            scores[1] += len(validation_images["images"]) / validation_data_amount
            scores[2] += len(test_images["images"]) / test_data_amount
            scores[0] += (
                sum(
                    train_categories[category_indicies]
                    / (len(train_images["images"]) + 0.01)
                )
                * i
                / dist_weight
            )
            scores[1] += (
                sum(
                    validation_categories[category_indicies]
                    / (len(validation_images["images"]) + 0.01)
                )
                * i
                / dist_weight
            )
            scores[2] += (
                sum(
                    test_categories[category_indicies]
                    / (len(test_images["images"]) + 0.01)
                )
                * i
                / dist_weight
            )
            min_score_idx = np.random.choice(np.flatnonzero(scores == scores.min()))
            if min_score_idx == 0:
                for j in category_indicies:
                    train_categories[j] += 1
                train_images["images"].append(
                    {
                        "path": img,
                        "id": image_ids[i],
                        "bboxs": img_bbox.tolist(),
                        "labels": categories.tolist(),
                    }
                )
            elif min_score_idx == 1:
                for j in category_indicies:
                    validation_categories[j] += 1
                validation_images["images"].append(
                    {
                        "path": img,
                        "id": image_ids[i],
                        "bboxs": img_bbox.tolist(),
                        "labels": categories.tolist(),
                    }
                )
            else:
                for j in category_indicies:
                    test_categories[j] += 1
                test_images["images"].append(
                    {
                        "path": img,
                        "id": image_ids[i],
                        "bboxs": img_bbox.tolist(),
                        "labels": categories.tolist(),
                    }
                )

    #Saving the splits
    split_name = ["train", "validation", "test"]
    for i, split in enumerate([train_images, validation_images, test_images]):
        with open(f"{datapath}/{split_name[i]}_data.json", "w") as fp:
            json.dump(split, fp, indent=2)

def rotate_and_save_images(dataset_json, base_path):
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == "Orientation":
            break
    
    #Getting all image paths and ids
    image_paths = [i["file_name"] for i in dataset_json["images"]]

    for path in image_paths:
        pil_img = Image.open(str(base_path) + "/" + path)
        if pil_img._getexif():  # type: ignore
            exif = dict(pil_img._getexif().items())  # type: ignore
            # Rotate portrait and upside down images if necessary
            if orientation in exif:
                if exif[orientation] == 3:
                    pil_img = pil_img.rotate(180, expand=True)
                    pil_img.save(str(base_path) + "/" + path) 
                if exif[orientation] == 6:
                    pil_img = pil_img.rotate(270, expand=True)
                    pil_img.save(str(base_path) + "/" + path) 
                if exif[orientation] == 8:
                    pil_img = pil_img.rotate(90, expand=True)
                    pil_img.save(str(base_path) + "/" + path)        

def find_n_ground_truths():
    proot_path = get_project12_root()
    path = proot_path / "data/data_wastedetection"

    ground_truth_dict = {}

    for split in ["train", "validation", "test"]:
        with open(path / f"{split}_data.json", "r") as f:
            images_json = json.loads(f.read())  
        images = images_json["images"]
        for img in images:
            ground_truth_dict[img["path"]] = len(img["labels"])
    
    with open(f"{path}/image_n_ground_truths.json", "w") as fp:
        json.dump(ground_truth_dict, fp, indent=2)

def load_dataset_rcnn(
    batch_size: int = 64,
    normalize: bool = True,
    image_size: tuple = (32, 32),
    split: str = "train",
    tune_for_perfomance: bool = False,
    use_data_augmentation: bool = False,
    augmentation_flip: str = "horizontal_and_vertical",
    augmentation_rotation: float = 0.5,
    augmentation_contrast: float = 0.5,
    pct_not_background: float = 0.25,
    **kwargs,
) -> tf.data.Dataset | tf.raw_ops.MapDataset:
    """
    Small note: To get class_names do dataset._input_dataset.class_names
    unless normalize = False, then just do dataset.class_names
    """
    proot_path = get_project12_root()
    path = proot_path / "data/data_wastedetection"

    with open(path / f"{split}_proposals.json", "r") as f:
        data_json = json.loads(f.read())
    
    with open(path / f"{split}_data.json", "r") as f:
        images_json = json.loads(f.read())

    with open(path / f"cat2id.json", "r") as f:
        cat2id_json = json.loads(f.read())

    background_class_idx = cat2id_json["Background"]

    if split == "train":
        proposals = list(data_json.values())
        proposal_labels = []
        proposal_boxes = []
        for p in proposals:
            proposal_labels.append([cat2id_json[i[-1]] for i in p])
            proposal_boxes.append([i[:-1] for i in p])
            
        images = images_json["images"]
        images_ids = [i["id"] for i in images]
        images_paths = [i["path"] for i in images]
        images_paths_proposals = [[images_paths[images_ids.index(int(i))]] for i in data_json.keys()]

        dataset = tf.data.Dataset.from_tensor_slices(
            (images_paths_proposals, proposal_boxes, proposal_labels)
        )

        @tf.function
        def make_img_batch_train(img_path, img_boxes, img_labels):
            base_img = tf.io.read_file(str(path) + "/" + tf.squeeze(img_path))
            base_img = tf.image.decode_image(base_img, channels=3, dtype=tf.float32)
            tensor_batch = []
            tensor_labels = []
            n_not_background = int(batch_size * pct_not_background)
            n_background = batch_size - n_not_background
            not_background_idx = tf.squeeze(tf.where(img_labels != background_class_idx))
            background_idx = tf.squeeze(tf.where(img_labels == background_class_idx))
            not_background_choices = tf.random.shuffle(not_background_idx)
            background_choices = tf.random.shuffle(background_idx)
            for i in range(n_not_background):
                choice = not_background_choices[i]
                bbox = img_boxes[choice]
                label = img_labels[choice]
                img_crop = tf.image.crop_to_bounding_box(
                    base_img, int(bbox[1]), int(bbox[0]), int(bbox[3]), int(bbox[2])
                )
                img_crop = tf.image.resize(img_crop, [image_size[0], image_size[1]])
                tensor_batch.append(img_crop)
                tensor_labels.append(label)
            for i in range(n_background):
                choice = background_choices[i]
                bbox = img_boxes[choice]
                label = img_labels[choice]
                img_crop = tf.image.crop_to_bounding_box(
                    base_img, int(bbox[1]), int(bbox[0]), int(bbox[3]), int(bbox[2])
                )
                img_crop = tf.image.resize(img_crop, [image_size[0], image_size[1]])
                tensor_batch.append(img_crop)
                tensor_labels.append(label)
            
            tensor_batch = tf.convert_to_tensor(tensor_batch)
            tensor_labels = tf.convert_to_tensor(tensor_labels)
            return tensor_batch, tensor_labels
        
        dataset = (
            dataset.map(lambda x, y, z: make_img_batch_train(x, y, z))
        )

    else:
        assert 2000 % batch_size == 0, "When split is train/test 2000 needs to be divideable by batch_size to make sure that one batch only corresponds to one image"
        one_image_n_batches = 2000 // batch_size
        proposals = list(data_json.values())
        proposal_labels = []
        proposal_boxes = []
        for p in proposals:
            proposal_labels.append([cat2id_json[i[-1]] for i in p])
            proposal_boxes.append([i[:-1] for i in p])
            
        images = images_json["images"]
        images_ids = [i["id"] for i in images]
        images_paths = [i["path"] for i in images]
        images_paths_proposals = [[images_paths[images_ids.index(int(i))]] for i in data_json.keys()]

        new_batches_images = []
        new_batches_proposals_boxes = []
        new_batches_proposals_labels = []

        for i, img in enumerate(images_paths_proposals):
            boxes = proposal_boxes[i]
            labels = proposal_labels[i]
            for j in range(one_image_n_batches):
                new_batches_images.append([img[0]])
                new_batches_proposals_boxes.append(boxes[j*batch_size:j*batch_size+batch_size])
                new_batches_proposals_labels.append(labels[j*batch_size:j*batch_size+batch_size])

        dataset = tf.data.Dataset.from_tensor_slices(
            (new_batches_images, new_batches_proposals_boxes, new_batches_proposals_labels)
        )

        @tf.function
        def make_img_batch_test(img_path, img_boxes, img_labels):
            base_img = tf.io.read_file(str(path) + "/" + tf.squeeze(img_path))
            base_img = tf.image.decode_image(base_img, channels=3, dtype=tf.float32)
            tensor_batch = []
            tensor_labels = []
            for i in range(img_boxes.shape[0]):
                bbox = img_boxes[i]
                label = img_labels[i]
                img_crop = tf.image.crop_to_bounding_box(
                    base_img, int(bbox[1]), int(bbox[0]), int(bbox[3]), int(bbox[2])
                )
                img_crop = tf.image.resize(img_crop, [image_size[0], image_size[1]])
                tensor_batch.append(img_crop)
                tensor_labels.append(label)
            
            tensor_batch = tf.convert_to_tensor(tensor_batch)
            tensor_labels = tf.convert_to_tensor(tensor_labels)
            return tensor_batch, tensor_labels, img_path
        
        dataset = (
            dataset.map(lambda x, y, z: make_img_batch_test(x, y, z))
        )

    if tune_for_perfomance:
        AUTOTUNE = tf.data.AUTOTUNE
        dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)

    return dataset


if __name__ == "__main__":
    proot_path = get_project12_root()
    path = proot_path / "data/data_wastedetection"

    with open(path / "annotations.json", "r") as f:
        data_json = json.loads(f.read())

    rotate_and_save_images(data_json, path)
    make_data_splits(data_json, 0.7, 0.15, 0.15, path)
    find_n_ground_truths()
    ts = load_dataset_rcnn(
        train=True, batch_size=40, shuffle=True, image_size=(128, 128)
    )

    # ts_iter = iter(ts)

    # for i in range(50):
    #     img_batch, label, img_path = next(ts_iter)
    # batch_size = len(label)
    # label = list(label)
    # fig = plt.figure(figsize=(10, 10))
    # for i,j in enumerate(range(int(batch_size/9),batch_size,int(batch_size/9))):
    #     ax = plt.subplot(3, 3, i + 1)
    #     ax.imshow(img_batch[j])
    #     if type(label) == list:
    #         ax.set_title(label[j].numpy().decode("UTF-8"))  # type: ignore
    # plt.show()
    # fig = plt.figure(figsize=(10, 10))
    # for i,j in enumerate(range(1,10)):
    #     ax = plt.subplot(3, 3, i + 1)
    #     ax.imshow(img_batch[-j])
    #     if type(label) == list:
    #         ax.set_title(label[-j].numpy().decode("UTF-8"))  # type: ignore
    # plt.show()
