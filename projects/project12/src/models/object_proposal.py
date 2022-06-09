"""
Script containing function for object bounding box proposal generation in TF-domain.
"""


import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
# # import cv2
import matplotlib.pyplot as plt


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import os
from src.utils import get_project_root

from PIL import Image, ExifTags
from pycocotools.coco import COCO
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
import colorsys
import random
import pylab

import cv2

from tqdm import tqdm


class ObjectProposalGenerator:
    def __init__(self, fast_selector=False, cv2_model=True):
        """
        Initialize the ObjectProposalGenerator class.
        """
        print("Loading model...")
        if not cv2_model:
            self.detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
        else:
            self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
            self.fast_selector = fast_selector
        print("Model loaded.")


    def object_porposal_cv2(self, image):
        """
        input: rgb image
        output: list of bounding box coordinates
        """
        self.ss.setBaseImage(image)
        self.ss.switchToSelectiveSearchFast()
        print("Running selective search...")
        proposals = self.ss.process()
        print("Selective search done.")
        
        return proposals


    def get_iou(self, bb1, bb2):
        """
        Compute the intersection-over-union of two sets of bounding boxes.
        Args:
            bb1: (N, 4) array of bounding boxes for N detections.
            bb2: (M, 4) array of bounding boxes for M detections.
        Returns:
            iou: (N, M) array of IoU for N detections in `bb1` and M detections in `bb2`.
        """

        # Compute intersection areas
        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb2['x1'] < bb2['x2']
        assert bb2['y1'] < bb2['y2']    
        
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])    
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0    
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)    
        
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])    
        
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou





    def make_proposals_image(self, image, gt_bboxes, labels):
        
        ## Define ground truth bounding boxes
        gtvalues=[]
        for i in range(len(gt_bboxes)):
            box = gt_bboxes[i]
            x = int(box[0])
            y = int(box[1])
            w = int(box[2])
            h = int(box[3])
            gt_label = labels[i]
            gtvalues.append([x, y, w, h, gt_label])

        ## Compute proposals
        print("Computing proposals...")
        self.ss.setBaseImage(image)
        self.ss.switchToSelectiveSearchQuality()
        ssresults = self.ss.process()

        ## Loop over proposal in the first 2000 proposals
        print("Computing IoU's...")
        proposal_list = []
        subset = random.choices(ssresults, k=2000)

        for e, result in tqdm(enumerate(subset)):
            if e-(len(gtvalues)+1) < 2000:  # Only do for the first 2000 proposals
                
                ## For each proposal, compute the intersection for all gt_bboxes
                for gtval in gtvalues:
                    gt_x, gt_y, gt_w, gt_h = gtval[0], gtval[1], gtval[2], gtval[3]
                    x, y, w, h = result
                    iou = self.get_iou(bb1={"x1":gt_x,"x2":gt_x+gt_w,"y1":gt_y,"y2":gt_y+gt_h},bb2={"x1":x,"x2":x+w,"y1":y,"y2":y+h})

                        
                    ## If the iou is greater than 0.5, we assign the label of that gt bbox to the proposal
                    proposal_label = 'Background'
                    iou_temp = 0.0
                    if iou > 0.50 and iou > iou_temp:
                        print(e)
                        print("IoU:", iou)
                        iou_temp = iou
                        proposal_label = gtval[-1]
                
                proposal_list.append([x, y, w, h, proposal_label])
        
        ## Lastly, append ground truth bboxes to the proposal list
        for gtval in gtvalues:
            proposal_list.append(gtval)

        return proposal_list
    

    def make_all_proposals(self, image_base_path, annotation_file_path, out_path):
        
        ## Read annotations
        with open(annotation_file_path, 'r') as f:
            dataset = json.loads(f.read())['images']

        all_images_proposals = {}
        for im in tqdm(dataset):
            ## Load image
            image_path = os.path.join(image_base_path, im['path'])
            image = cv2.imread(image_path)
            
            image_id = im['id']
            ## Get image bboxes and labels
            bboxes = im['bboxs']
            labels = im['labels']

            try:
                proposals = self.make_proposals_image(image=image, gt_bboxes=bboxes, labels=labels)
            except Exception as e:
                print(e)
                print("error in "+image_path)
                continue
            print(len(proposals))
            all_images_proposals[str(image_id)] = proposals
        return all_images_proposals


if __name__ == "__main__":
    os.chdir(get_project_root())
    dataset_path = 'data/data_wastedetection'
    # anns_file_path = dataset_path + '/' + 'annotations.json'
    annotation_file_path = dataset_path + '/' + 'testing_proposal_data.json'
    out_path = dataset_path + '/' + 'proposals.json'
    OPG = ObjectProposalGenerator()

    all_proposals = OPG.make_all_proposals(image_base_path=dataset_path, annotation_file_path=annotation_file_path, out_path=dataset_path)
    
    ## Write proposals
    with open(out_path, 'w') as f:
        dataset = json.loads(f.read())['images']

    
    
    # # Read annotations
    # with open(anns_file_path, 'r') as f:
    #     dataset = json.loads(f.read())

    # categories = dataset['categories']
    # anns = dataset['annotations']
    # imgs = dataset['images']

    # # Load specific image for testing function
    # image_filepath = 'batch_11/000028.jpg'
    # # pylab.rcParams['figure.figsize'] = (28,28)

    # # Obtain Exif orientation tag code
    # for orientation in ExifTags.TAGS.keys():
    #     if ExifTags.TAGS[orientation] == 'Orientation':
    #         break

    # # Loads dataset as a coco object
    # coco = COCO(anns_file_path)

    # # Find image id
    # img_id = -1
    # for img in imgs:
    #     if img['file_name'] == image_filepath:
    #         img_id = img['id']
    #         break

    # # Show image and corresponding annotations
    # if img_id == -1:
    #     print('Incorrect file name')
    # else:
    #     # Load image
    #     print(image_filepath)
    #     I = Image.open(dataset_path + '/' + image_filepath)

    #     # Load and process image metadata
    #     if I._getexif():
    #         exif = dict(I._getexif().items())
    #         # Rotate portrait and upside down images if necessary
    #         if orientation in exif:
    #             if exif[orientation] == 3:
    #                 I = I.rotate(180,expand=True)
    #             if exif[orientation] == 6:
    #                 I = I.rotate(270,expand=True)
    #             if exif[orientation] == 8:
    #                 I = I.rotate(90,expand=True)

    #     # Show image
    #     fig,ax = plt.subplots(1)
    #     plt.axis('off')
    #     plt.imshow(I)

    #     # Load mask ids
    #     annIds = coco.getAnnIds(imgIds=img_id, catIds=[], iscrowd=None)
    #     anns_sel = coco.loadAnns(annIds)

    #     # Show annotations
    #     for ann in anns_sel:
    #         [x, y, w, h] = ann['bbox']
    #         rect = Rectangle((x,y),w,h,linewidth=2,edgecolor="red",
    #                         facecolor='none', alpha=0.7, linestyle = '--')
    #         ax.add_patch(rect)

    #     # Show image with label rectangle
    #     # plt.show()





    #     ### Now the image is loaded, and the boxes can be generated
    #     proposal_generator = ObjectProposalGenerator(fast_selector=True, cv2_model=True)

    #     pred_boxes = proposal_generator.object_porposal_cv2(image=np.array(I))

    #     print(pred_boxes)
    #     # Show annotations
    #     for box in pred_boxes:
    #         [x, y, w, h] = box
    #         rect = Rectangle((x,y),w,h,linewidth=2,edgecolor="green", facecolor='none', alpha=0.7, linestyle = '-')
    #         ax.add_patch(rect)
    #     plt.show()
        








    # def object_proposal_hub(self, rgb_tensor):
    #     """
    #     input: rgb image tensor
    #     output: tensor of bounding box coordinates
    #     """

    #     # Creating prediction
    #     print("Creating prediction...")
    #     boxes, scores, classes, num_detections = self.detector(rgb_tensor)
    #     print(num_detections)
        
    #     # Processing outputs
    #     pred_boxes = boxes.numpy()[0].astype('int')
    #     proposal = []

    #     # Putting the boxes and labels on the image
    #     for (ymin,xmin,ymax,xmax) in pred_boxes:
    #         # img_boxes = cv2.rectangle(rgb_tensor, (xmin, ymax),(xmax, ymin),(0,255,0),2)      
    #         proposal.append([xmin, ymin, xmax, ymax])
        
    #     return proposal
