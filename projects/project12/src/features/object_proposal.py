from __future__ import annotations

"""
Script containing function for object bounding box proposal generation in TF-domain.
"""
# TODO Add timing module around different functions

import json
import os
from logging import Logger

import cv2
import numpy as np
import seaborn as sns
from PIL import Image
from projects.color_logger import init_logger
from projects.utils import get_project12_root
from tqdm import tqdm

sns.set()


class ObjectProposalGenerator:
    def __init__(self, logger: Logger):
        """
        Initialize the ObjectProposalGenerator class.
        """
        self.logger = logger

        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()  # type: ignore
        self.logger.info(f"Model loaded:\n\t {type(self.ss) = }")

    def calc_iou(self, bb1: dict[str, int], bb2: dict[str, int]) -> float:
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

    def make_proposals_image(
        self, image: np.ndarray, gt_bboxes: list, labels: list
    ) -> list[list]:
        ## Define ground truth bounding boxes
        assert len(gt_bboxes) == len(labels)

        gtvalues = []
        for gt_label, [x, y, w, h] in zip(labels, [map(int, x) for x in gt_bboxes]):
            gtvalues.append([x, y, w, h, gt_label])

        ## Compute proposals
        self.logger.debug("Computing proposals...")

        self.ss.setBaseImage(image)
        self.ss.switchToSelectiveSearchFast(inc_k=150)
        ssresults = self.ss.process()

        ## Loop over proposal in the first 2000 proposals
        self.logger.debug("Computing IoU's...")

        n_proposals = 2000
        min_high_iou_proposals = 16

        proposal_list = []
        np.random.shuffle(ssresults)

        count_iou = 0
        for i, [x, y, w, h] in enumerate(ssresults):

            ## For each proposal, compute the intersection for all gt_bboxes
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

                iou = self.calc_iou(bb1, bb2)

                ## If the iou is greater than 0.5, we assign the label of that gt bbox to the proposal
                # iou_temp = 0.0
                if iou > 0.4 and iou > max_iou:
                    # self.logger.debug(i)
                    # self.logger.debug("IoU:", iou)
                    max_iou = iou
                    proposal_label = gt_label

            if i >= n_proposals - len(gtvalues) and count_iou >= min_high_iou_proposals:
                break  # Only do for the first 2000 proposals

            if (
                len(proposal_list) < n_proposals - len(gtvalues) - min_high_iou_proposals + count_iou
                or max_iou > 0.4
                or len(gtvalues) == 0
            ):
                if proposal_label == "Background" and max_iou < 0.2:
                    proposal_list.append([*map(int, [x, y, w, h]), proposal_label])
                elif proposal_label != "Background":
                    proposal_list.append([*map(int, [x, y, w, h]), proposal_label])

            count_iou += 1 if proposal_label != "Background" else 0

        proposal_list.extend(gtvalues)

        return proposal_list

    def make_all_proposals(self, image_base_path, annotation_file_path):
        ## Read annotations
        with open(annotation_file_path, "r") as f:
            dataset = json.loads(f.read())["images"]

        # for orientation in ExifTags.TAGS.keys():
        #     if ExifTags.TAGS[orientation] == "Orientation":
        #         break
        orientation_flag = 0x0112

        ## Create d
        out_dict = {}
        for im in tqdm(dataset[:3]):
            ## Load image and rotate if necessary
            image_path = os.path.join(image_base_path, im["path"])
            pil_img = Image.open(image_path)
            if pil_img._getexif():
                exif = dict(pil_img._getexif().items())
                # Rotate portrait and upside down images if necessary
                if orientation_flag in exif:
                    if exif[orientation_flag] == 3:
                        pil_img = pil_img.rotate(180, expand=True)
                    if exif[orientation_flag] == 6:
                        pil_img = pil_img.rotate(270, expand=True)
                    if exif[orientation_flag] == 8:
                        pil_img = pil_img.rotate(90, expand=True)

            image = np.array(pil_img)

            ## Get image bboxes and labels
            image_id = im["id"]
            bboxes = im["bboxs"]
            labels = im["labels"]

            try:
                proposals = self.make_proposals_image(
                    image=image, gt_bboxes=bboxes, labels=labels
                )
            except Exception as err:
                self.logger.debug(f"error in {image_path}\n{err = }")
                continue

            self.logger.debug(f"{len(proposals) = }")
            out_dict[str(image_id)] = proposals

        return out_dict


if __name__ == "__main__":
    logger = init_logger(
        __file__,
        True,
    )

    data_path = get_project12_root() / "data"
    dataset_path = data_path / "data_wastedetection"
    split = "test"

    annot_file_path = dataset_path / f"{split}_data.json"
    out_path = dataset_path / f"{split}_proposals.json"

    OPG = ObjectProposalGenerator(logger)

    all_proposals = OPG.make_all_proposals(
        image_base_path=dataset_path,
        annotation_file_path=annot_file_path,
    )

    ## Write proposals to json file
    with open(dataset_path / f"{split}_proposals.json", "w") as fp:
        json.dump(all_proposals, fp)
