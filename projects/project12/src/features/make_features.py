from __future__ import annotations
from ctypes import resize
from logging import Logger, getLogger

import cv2
import numpy as np
from projects.color_logger import init_logger
from projects.utils import get_project12_root
from pathlib import Path
import json
from PIL import Image
from tqdm import tqdm

import multiprocessing as mp

###############################################################################
#                                                                             #
# ███╗   ███╗██╗   ██╗██╗  ████████╗██╗     ██████╗ ██████╗  ██████╗ ██╗      #
# ████╗ ████║██║   ██║██║  ╚══██╔══╝██║    ██╔════╝██╔═══██╗██╔═══██╗██║      #
# ██╔████╔██║██║   ██║██║     ██║   ██║    ██║     ██║   ██║██║   ██║██║      #
# ██║╚██╔╝██║██║   ██║██║     ██║   ██║    ██║     ██║   ██║██║   ██║██║      #
# ██║ ╚═╝ ██║╚██████╔╝███████╗██║   ██║    ╚██████╗╚██████╔╝╚██████╔╝███████╗ #
# ╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝   ╚═╝     ╚═════╝ ╚═════╝  ╚═════╝ ╚══════╝ #
# this do be pretty cool ___                                                  #
###############################################################################


def calc_iou(bb1: dict[str, int], bb2: dict[str, int]) -> float:
    """
    Compute the intersection-over-union of two sets of bounding boxes.
    Args:
        bb1: (N, 4) dict of bounding boxes for N detections.
        bb2: (M, 4) dict of bounding boxes for M detections.
    Returns:
        iou: float
    """
    # Assert that coordinates are not wrong
    if any(
        [
            bb1["x1"] > bb1["x2"],
            bb1["y1"] > bb1["y2"],
            bb2["x1"] > bb2["x2"],
            bb2["y1"] > bb2["y2"],
        ]
    ):
        return 0

    # Compute intersection areas
    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return iou if 0.0 <= iou <= 1.0 else 0


def xyhw_2_bbox(x, y, h, w) -> dict[str, int]:
    return {"x1": x, "x2": x + w, "y1": y, "y2": y + h}


def fast_bb_proposals(
    image: np.ndarray,
    gt_bboxes,
    gt_labels,
    n_proposals: int = 2000,
    min_iou_proposals: int = 16,
    inc_k: int = 150,
    logger: Logger = getLogger(__file__),
) -> list:

    random_threshold = n_proposals - len(gt_labels) - min_iou_proposals

    if len(gt_bboxes) != len(gt_labels):
        raise ValueError(
            f"Not same length of gt_boxes and labels {len(gt_bboxes) = }, {len(gt_labels) = }"
        )

    logger.debug("Computing proposals")
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()  # type: ignore

    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast(inc_k=inc_k)
    proposals = ss.process()
    np.random.shuffle(proposals)

    samples = []
    iou_count = 0

    for i, proposed_xywh in enumerate(proposals):
        proposed_label = "Background"
        max_iou = 0

        for gt_xywh, gt_label in zip(gt_bboxes, gt_labels):
            proposed_bbox = xyhw_2_bbox(*proposed_xywh)
            gt_bbox = xyhw_2_bbox(*gt_xywh)

            iou = calc_iou(proposed_bbox, gt_bbox)

            if iou > 0.5 and iou > max_iou:
                max_iou = iou
                proposed_label = gt_label

        if len(samples) < random_threshold + iou_count or max_iou > 0.5:
            samples.append([*map(int, proposed_xywh), proposed_label])

        if max_iou > 0.5:
            iou_count += 1

        if i >= n_proposals and iou_count >= min_iou_proposals:
            break

    samples.extend([[*gt_xywh, label] for gt_xywh, label in zip(gt_bboxes, gt_labels)])
    return samples


def proposal_mp_task(
    info: dict, imgs_folder: Path, logger: Logger = getLogger(__file__)
) -> tuple[list, str]:
    orientation_switch = {3: 180, 6: 270, 8: 90}
    ORIENTATION_FLAG = 274

    [img_path, img_id, bboxs, labels] = info.values()
    img_path = imgs_folder / img_path

    with Image.open(img_path) as file:
        ori_flag = file.getexif().get(ORIENTATION_FLAG, None)
        rot = orientation_switch.get(ori_flag, 0)

        img = file.rotate(rot, expand=True)
        img = np.array(img)

        proposals = fast_bb_proposals(img, bboxs, labels)

        num_ious_found = sum([x[-1] != "Background" for x in proposals])
        if num_ious_found < 16:
            logger.warning(
                f"Did not find 16 iou's > .5 in {img_id = }. Found {num_ious_found} iou's in total"
            )
    return proposals, img_id


def generate_proposals(imgs_folder: Path, annot_path: Path):
    with open(annot_path, "r") as f:
        img_info = json.load(f)["images"]

    proposal_dict = {}
    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
        results = [
            pool.apply_async(proposal_mp_task, (info, imgs_folder))
            for info in tqdm(img_info[:20], desc="jobs applied: ")
        ]
        for proposals, img_id in [
            r.get() for r in tqdm(results, desc="jobs processed: ")
        ]:
            proposal_dict[img_id] = proposals
    
    breakpoint()
    return proposal_dict


if __name__ == "__main__":
    # Logging
    log_path = get_project12_root() / "log"
    logger = init_logger(__file__, False, log_path)

    # Datasplit (train, validation, test)
    split = "train"

    data_path = get_project12_root() / "data"
    dataset_path = data_path / "data_wastedetection"

    # Annotation file path with bboxes and labels
    annot_file_path = dataset_path / f"{split}_data.json"

    all_proposals = generate_proposals(
        imgs_folder=dataset_path, annot_path=annot_file_path
    )

    ## Write proposals to json file
    with open(dataset_path / f"{split}_proposals.json", "w") as fp:
        json.dump(all_proposals, fp, indent=4)
