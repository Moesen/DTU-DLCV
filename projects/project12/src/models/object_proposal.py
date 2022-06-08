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


class ObjectProposalGenerator:
    def __init__(self, out_width=256, out_height=256):
        """
        Initialize the ObjectProposalGenerator class.
        """
        self.detector = detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
        self.out_width = out_width
        self.out_height = out_height

    def object_proposal(self, rgb_tensor):
        """
        input: rgb image tensor
        output: tensor of bounding box coordinates
        """

        # #Load image by Opencv2
        # img = cv2.imread('image_2.jpg')
        # #Resize to respect the input_shape
        # inp = cv2.resize(img, (width , height ))
        # #Convert img to RGB
        # rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        # # COnverting to uint8
        # rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)

        # Creating prediction
        boxes, scores, classes, num_detections = self.detector(rgb_tensor)
        
        # Processing outputs
        pred_boxes = boxes.numpy()[0].astype('int')

        proposal = []
        # Putting the boxes and labels on the image
        for (ymin,xmin,ymax,xmax) in pred_boxes:
            # img_boxes = cv2.rectangle(rgb_tensor, (xmin, ymax),(xmax, ymin),(0,255,0),2)      
            proposal.append([xmin, ymin, xmax, ymax])
        
        # output = tf.image.crop_and_resize(rgb_tensor, boxes, box_indices, CROP_SIZE)

        # plt.imshow(img_boxes)

        return proposal




if __name__ == "__main__":



    os.chdir(get_project_root())
    dataset_path = 'data/data_wastedetection'
    anns_file_path = dataset_path + '/' + 'annotations.json'
    # Read annotations
    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())

    categories = dataset['categories']
    anns = dataset['annotations']
    imgs = dataset['images']

    # Load specific image for testing function
    image_filepath = 'batch_11/000030.jpg'
    pylab.rcParams['figure.figsize'] = (28,28)

    # Obtain Exif orientation tag code
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            break

    # Loads dataset as a coco object
    coco = COCO(anns_file_path)

    # Find image id
    img_id = -1
    for img in imgs:
        if img['file_name'] == image_filepath:
            img_id = img['id']
            break

    # Show image and corresponding annotations
    if img_id == -1:
        print('Incorrect file name')
    else:
        # Load image
        print(image_filepath)
        I = Image.open(dataset_path + '/' + image_filepath)

        # Load and process image metadata
        if I._getexif():
            exif = dict(I._getexif().items())
            # Rotate portrait and upside down images if necessary
            if orientation in exif:
                if exif[orientation] == 3:
                    I = I.rotate(180,expand=True)
                if exif[orientation] == 6:
                    I = I.rotate(270,expand=True)
                if exif[orientation] == 8:
                    I = I.rotate(90,expand=True)

        # Show image
        fig,ax = plt.subplots(1)
        plt.axis('off')
        plt.imshow(I)

        # Load mask ids
        annIds = coco.getAnnIds(imgIds=img_id, catIds=[], iscrowd=None)
        anns_sel = coco.loadAnns(annIds)

        # Show annotations
        for ann in anns_sel:
            [x, y, w, h] = ann['bbox']
            rect = Rectangle((x,y),w,h,linewidth=2,edgecolor="red",
                            facecolor='none', alpha=0.7, linestyle = '--')
            ax.add_patch(rect)

        # Show image with label rectangle
        # plt.show()


    ### Begin making own rectangle
    image_tensor = tf.convert_to_tensor(I, dtype=tf.uint8)
    image_tensor = tf.expand_dims(image_tensor , 0)
    print(image_tensor.shape)


    proposal_generator = ObjectProposalGenerator()
    box = proposal_generator.object_proposal(rgb_tensor=image_tensor)

    for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores, pred_boxes, pred_labels):
        if score < 0.5:
            continue

        score_txt = f'{100 * round(score)}%'
        img_boxes = cv2.rectangle(rgb,(xmin, ymax),(xmax, ymin),(0,255,0),2)      
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_boxes, label,(xmin, ymax-10), font, 1.5, (255,0,0), 2, cv2.LINE_AA)
        cv2.putText(img_boxes,score_txt,(xmax, ymax-10), font, 1.5, (255,0,0), 2, cv2.LINE_AA)

