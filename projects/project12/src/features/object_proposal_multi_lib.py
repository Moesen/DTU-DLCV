from __future__ import annotations

from logging import Logger, getLogger

import cv2
import numpy as np

###############################################################################
#                                                                             #
# ███╗   ███╗██╗   ██╗██╗  ████████╗██╗     ██████╗ ██████╗  ██████╗ ██╗      #
# ████╗ ████║██║   ██║██║  ╚══██╔══╝██║    ██╔════╝██╔═══██╗██╔═══██╗██║      #
# ██╔████╔██║██║   ██║██║     ██║   ██║    ██║     ██║   ██║██║   ██║██║      #
# ██║╚██╔╝██║██║   ██║██║     ██║   ██║    ██║     ██║   ██║██║   ██║██║      #
# ██║ ╚═╝ ██║╚██████╔╝███████╗██║   ██║    ╚██████╗╚██████╔╝╚██████╔╝███████╗ #
# ╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝   ╚═╝     ╚═════╝ ╚═════╝  ╚═════╝ ╚══════╝ #
#                                                                             #
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
    assert all(
        [
            bb1["x1"] < bb1["x2"],
            bb1["y1"] < bb1["y2"],
            bb2["x1"] < bb2["x2"],
            bb2["y1"] < bb2["y2"],
        ]
    )

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

    assert iou >= 0.0 and iou <= 1.0
    return iou


def make_bb_proposals(
    image: np.ndarray,
    gt_bboxes: list,
    labels: list,
    n_proposals: int = 2000,
    min_high_iou_proposals: int = 16,
    logger: Logger = getLogger(),
) -> list[list]:
    ## Define ground truth bounding boxes
    assert len(gt_bboxes) == len(labels)

    gtvalues = []
    for gt_label, [x, y, w, h] in zip(labels, [map(int, x) for x in gt_bboxes]):
        gtvalues.append([x, y, w, h, gt_label])

    logger.debug("Constructing ss")
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()  # type: ignore

    ## Compute proposals
    logger.debug("Computing proposals...")

    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast(inc_k=100)
    ssresults = ss.process()

    ## Loop over proposal in the first 2000 proposals
    logger.debug("Computing IoU's...")

    proposal_list = []
    np.random.shuffle(ssresults)

    count_iou = 0
    for i, [x, y, w, h] in enumerate(ssresults):
        # For each proposal, 
        # compute the intersection for all gt_bboxes
        max_iou = 0
        proposal_label = "Background"

        for [gt_x, gt_y, gt_w, gt_h, gt_label] in gtvalues:
            bb1 = {
                "x1": gt_x,
                "x2": gt_x + gt_w,
                "y1": gt_y,
                "y2": gt_y + gt_h,
            }
            bb2 = {"x1": x, "x2": x + w, "y1": y, "y2": y + h}

            iou = calc_iou(bb1, bb2)

            # If the iou is greater than 0.5, 
            # we assign the label of that gt bbox to the proposal
            # iou_temp = 0.0
            if iou > 0.50 and iou > max_iou:
                # logger.debug(i)
                # logger.debug("IoU:", iou)
                max_iou = iou
                proposal_label = gt_label

        if (
            len(proposal_list) < n_proposals - min_high_iou_proposals + count_iou
            or max_iou > 0.5
        ):
            proposal_list.append([*map(int, [x, y, w, h]), proposal_label])

        count_iou += 1 if proposal_label != "Background" else 0

        if i >= n_proposals and count_iou >= min_high_iou_proposals:
            break  # Only do for the first 2000 proposals

    proposal_list.extend(gtvalues)

    return proposal_list
