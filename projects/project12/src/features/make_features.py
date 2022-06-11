from __future__ import annotations
import os
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

#################################################################################
#                                                                               #
#  ███╗   ███╗██╗   ██╗██╗  ████████╗██╗     ██████╗ ██████╗  ██████╗ ██╗       #
#  ████╗ ████║██║   ██║██║  ╚══██╔══╝██║    ██╔════╝██╔═══██╗██╔═══██╗██║       #
#  ██╔████╔██║██║   ██║██║     ██║   ██║    ██║     ██║   ██║██║   ██║██║       #
#  ██║╚██╔╝██║██║   ██║██║     ██║   ██║    ██║     ██║   ██║██║   ██║██║       #
#  ██║ ╚═╝ ██║╚██████╔╝███████╗██║   ██║    ╚██████╗╚██████╔╝╚██████╔╝███████╗  #
#  ╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝   ╚═╝     ╚═════╝ ╚═════╝  ╚═════╝ ╚══════╝  #
#    this do be pretty cool ___                                                 #
#################################################################################


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


def xyhw2bbox(x, y, h, w) -> dict[str, int]:
    return {"x1": x, "x2": x + w, "y1": y, "y2": y + h}


def make_bb_proposals(
    image: np.ndarray,
    gt_bboxes,
    gt_labels,
    n_proposals: int = 2000,
    iou_object_thresh: float = 0.4,
    iou_bg_tresh: float = 0.2,
    min_iou_proposals: int = 16,
    inc_k: int = 150,
    logger: Logger = getLogger(__file__),
) -> list:

    if len(gt_bboxes) != len(gt_labels):
        logger.error(
            f"Not same length of gt_boxes and labels {len(gt_bboxes) = }, {len(gt_labels) = }"
        )

    rand_thresh = n_proposals - len(gt_labels) - min_iou_proposals
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()  # type: ignore

    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast(inc_k=inc_k)
    proposals = ss.process()
    np.random.shuffle(proposals)

    samples = []
    iou_count = 0

    for i, proposed_xywh in enumerate(proposals):
        if i >= n_proposals and iou_count >= min_iou_proposals:
            break

        proposed_label = "Background"
        max_iou = 0

        for gt_xywh, gt_label in zip(gt_bboxes, gt_labels):
            proposed_bbox = xyhw2bbox(*proposed_xywh)
            gt_bbox = xyhw2bbox(*gt_xywh)
            iou = calc_iou(proposed_bbox, gt_bbox)

            if iou > iou_object_thresh and iou > max_iou:
                max_iou = iou
                proposed_label = gt_label

        # Checking if iou > object threshold
        obj_check = max_iou > iou_object_thresh and proposed_label != "Background"
        # Checking if iou < background threshold
        # and if there is room in samples list
        bg_check = len(samples) < rand_thresh + iou_count and max_iou < iou_bg_tresh
        if obj_check or bg_check:
            samples.append([*map(int, proposed_xywh), proposed_label])
        if obj_check:
            iou_count += 1

    samples.extend([[*gt_xywh, label] for gt_xywh, label in zip(gt_bboxes, gt_labels)])
    return samples


def make_bb_proposals_np(
    image: np.ndarray,
    gt_bboxes,
    gt_labels,
    img_id: int,
    n_proposals: int = 2000,
    iou_object_thresh: float = 0.4,
    iou_bg_tresh: float = 0.2,
    min_iou_proposals: int = 16,
    inc_k: int = 150,
    logger: Logger = getLogger(__file__),
) -> list:

    if len(gt_bboxes) != len(gt_labels):
        logger.error(
            f"Not same length of gt_boxes and labels {len(gt_bboxes) = }, {len(gt_labels) = }"
        )
        return []

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()  # type: ignore
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast(inc_k=inc_k)
    proposals = ss.process()

    idxs = np.array(
        np.meshgrid(range(len(proposals)), range(len(gt_bboxes)))
    ).T.reshape(-1, 2)

    ious = np.hstack([idxs, np.zeros((idxs.shape[0], 1))])
    for idx, [p_idx, gt_idx] in tqdm(enumerate(idxs), total=idxs.shape[0]):
        p_bbox = xyhw2bbox(*proposals[p_idx])
        gt_bbox = xyhw2bbox(*gt_bboxes[gt_idx])
        ious[idx, 2] = calc_iou(p_bbox, gt_bbox)

    ious_above_thresh = ious[ious[:, 2] > iou_object_thresh]
    ious_below_treshold = ious[ious[:, 2] < iou_bg_tresh]

    if ious_above_thresh.shape[0] < min_iou_proposals:
        logger.warning(
            f"img: {img_id} found {ious_above_thresh.shape[0]} ious above threshold = {iou_object_thresh}"
        )

    rand_above_idx = np.random.choice(ious_above_thresh.shape[0], min_iou_proposals)

    # Finding the the remaining samples below
    num_fills = n_proposals - min_iou_proposals - len(gt_bboxes)
    rand_below_idx = np.random.choice(ious_below_treshold.shape[0], num_fills)

    # Putting it all into a list
    return_list = []
    for idx in rand_above_idx:  # type: ignore
        p_idx, gt_idx, _ = ious[idx]
        proposal_xywh = proposals[int(p_idx)]
        gt_label = gt_labels[int(gt_idx)]
        return_list.append([*map(int, proposal_xywh), gt_label])

    for idx in rand_below_idx:  # type: ignore
        p_idx, gt_idx, _ = ious[idx]
        proposal_xywh = proposals[int(p_idx)]
        gt_label = "Background"
        return_list.append([*map(int, proposal_xywh), gt_label])

    for gt_xywh, gt_label in zip(gt_bboxes, gt_labels):
        return_list.append([*gt_xywh, gt_label])

    return return_list


orientation_switch = {3: 180, 6: 270, 8: 90}
ORIENTATION_FLAG = 274


def proposal_mp_task(
        info: dict, imgs_folder: Path, out_path: Path,logger: Logger = getLogger(__file__)
) -> None:
    [img_path, img_id, bboxs, labels] = info.values()
    img_path = imgs_folder / img_path

    logger.debug(f"Computing proposals for {img_id = }")

    with Image.open(img_path) as file:
        ori_flag = file.getexif().get(ORIENTATION_FLAG, None)
        rot = orientation_switch.get(ori_flag, 0)

        img = file.rotate(rot, expand=True)
        img = np.array(img)

        proposals = make_bb_proposals(img, bboxs, labels)

        ## Write proposals to json file
        with open(out_path / f"{img_id}_proposal.json", "w") as fp:
            logger.info(f"Saving proposals for {img_id = }")
            json.dump(proposals, fp, indent=2)


def generate_proposals(img_folder: Path, out_folder: Path, annot_path: Path):
    with open(annot_path, "r") as f:
        img_info = json.load(f)["images"]

    logger = getLogger(__file__)
    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
        logger.info(f"Spawned pool with {mp.cpu_count()} workers")
        results = [
            pool.apply_async(proposal_mp_task, (info, img_folder, out_folder))
            for info in tqdm(img_info[:10], desc="jobs applied: ")
        ]
        [r.get() for r in tqdm(results, desc="jobs processed: ")]


if __name__ == "__main__":
    # Logging
    log_path = get_project12_root() / "log"
    logger = init_logger(__file__, True, log_path)

    data_path = get_project12_root() / "data/data_wastedetection"
    out_path = get_project12_root() / "proposals"

    if not out_path.is_dir():
        yn = input(
            "Did not find a folder with name proposals in. Do you want me to create one? (y/n)"
        )
        if yn == "y":
            os.mkdir(out_path)
        else:
            raise Exception(
                f"No dir created, can't continue, please make sure {out_path = } exists"
            )

    for split in ["train", "validation", "test"]:
        logger.info(f"Beginning to find proposals for {split = }")
        annot_file_path = data_path / f"{split}_data.json"
        generate_proposals(data_path, out_path, annot_file_path)
