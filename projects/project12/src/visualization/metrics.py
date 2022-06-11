

from typing import List
import numpy as np
import tensorflow as tf
from keras import backend as K

import matplotlib.image as img
import json

from projects.project12.src.data.dataloader import load_dataset_rcnn
from projects.project12.src.models.post_processing import NMS

from projects.utils import get_project12_root

from projects.project12.src.metrics.BoundingBox import BoundingBox
from projects.project12.src.metrics.BoundingBoxes import BoundingBoxes
from projects.project12.src.metrics.Evaluator import *
from projects.project12.src.metrics.utils import *


PROJECT_ROOT = get_project12_root()
model_name = "trash_conv_20220611135130" #'hotdog_conv_20220604214318' #'hotdog_conv_20220604190940'
model_path = PROJECT_ROOT / "models" / model_name

new_model = tf.keras.models.load_model(model_path)

# Check its architecture
new_model.summary()


### Mean average Precision (mAP) 

batch_size = 100
img_size = (128,128)

val_data = load_dataset_rcnn(
    split="validation",
    normalize=False,
    use_data_augmentation=False,
    batch_size=batch_size,
    tune_for_perfomance=False,
    image_size=img_size,
    validation_mode = "object"
)

path = PROJECT_ROOT / "data/data_wastedetection"
with open(path / "annotations.json", "r") as f:
    dataset_json = json.loads(f.read())
catid2supercat = {i["id"]: i["supercategory"] for i in dataset_json["categories"]}
all_super_cat = list(set(i["supercategory"] for i in dataset_json["categories"]))
all_super_cat.append("Background")
labels = all_super_cat
b, classes = np.unique(labels, return_inverse=True)


BB_all_predicted = []
bb_class = []
bb_confidence = []

myBoundingBoxes = BoundingBoxes()


test_img, tensor_labels, img_path0, BB = next(iter(val_data))
img_path0 = img_path0[0].numpy().decode("UTF-8")

with open(path / f"validation_data.json", "r") as f:
    val_json = json.loads(f.read())  

img_path_json = val_json["images"]

with open(path / f"annotations.json", "r") as f:
    anno_json = json.loads(f.read())  

anno_json_dict = anno_json["images"]


print("Predicting all batches for all images")

n_batches = 1000/batch_size
from tqdm import tqdm

for n,(bb_img, tensor_labels, img_path, BB) in tqdm(
        enumerate(val_data), total=len(val_data)
    ):
    
    #print('Processing image batch: %i' % n)

    #compute classification predictions on the bb cropped images with bounding boxes = BB
    logits = new_model(bb_img, training=False)
    probs = tf.nn.softmax(logits,axis=1)
    predicted = K.cast(K.argmax(logits, axis=1), "uint8")

    BB_all_predicted.append( BB.numpy() )
    bb_confidence.append( probs.numpy().max(1) )
    bb_class.append( predicted.numpy() )
    
    #only perform the below if all bounding boxes are predicted in a single image
    if (n+1) % n_batches == 0:
        #print("Adding bounding boxes")

        img_path = img_path[0].numpy().decode("UTF-8")

        for im in img_path_json:
            if im["path"] == img_path:
                GT_bb = np.array(im["bboxs"])
                GT_class = im["labels"]

        for im in anno_json_dict:
            if im["file_name"] == img_path:
                imgSize = (im["width"], im["height"])

        #GT_bb = np.array([[30,30,15,15], [32,32,15,15]])
        #GT_class = ["hotdog","hotdog"] #[1,1]

        #add GT bbs 
        for gtb,gtc in zip(GT_bb,GT_class):
            gt_boundingBox = BoundingBox(imageName=img_path, classId=gtc, 
                                    x=gtb[0], y=gtb[1], w=gtb[2], h=gtb[3], typeCoordinates=CoordinatesType.Absolute,
                                bbType=BBType.GroundTruth, format=BBFormat.XYWH, imgSize=imgSize)
            myBoundingBoxes.addBoundingBox(gt_boundingBox)

        #reformat from shape (n_batches, batch_size, 4) -> (n_batches*batch_size, 4) = (2000,4)
        BB_all_predicted = np.array(BB_all_predicted)
        BB_all_predicted = BB_all_predicted.reshape((-1,BB_all_predicted.shape[-1]))

        #reformat from shape (n_batches, batch_size, 1) -> (n_batches*batch_size, 1) = (2000,1)
        bb_class = np.array(bb_class)
        bb_class = bb_class.flatten()[:,np.newaxis]

        #reformat from shape (n_batches, batch_size, 1) -> (n_batches*batch_size, 1) = (2000,1)
        bb_confidence = np.array(bb_confidence)
        bb_confidence = bb_confidence.flatten()[:,np.newaxis]

        #cast to tensorflow tensors
        BB_all_predicted = tf.convert_to_tensor(BB_all_predicted,dtype=tf.float32)
        bb_confidence = tf.convert_to_tensor(bb_confidence,dtype=tf.float32)
        bb_class = tf.convert_to_tensor(bb_class,dtype=tf.float32)

        # NMS post processing
        bb_selected, bb_confidence_selected, bb_class_selected = NMS(BB_all_predicted, bb_class, bb_confidence, classes, iout = 0.5, st = 0.4, max_out = 10)

        bb_labels_selected = [labels[int(i)] for i in bb_class_selected.numpy().squeeze().tolist()]

        #add detected bbs
        for db,dc,conf in zip(bb_selected, bb_labels_selected, bb_confidence_selected):
            detected_boundingBox = BoundingBox(imageName=img_path, classId=dc, classConfidence=conf,
                                    x=db[0], y=db[1], w=db[2], h=db[3], typeCoordinates=CoordinatesType.Absolute,
                                    bbType=BBType.Detected, format=BBFormat.XYWH, imgSize=imgSize)

            myBoundingBoxes.addBoundingBox(detected_boundingBox)

        BB_all_predicted = []
        bb_class = []
        bb_confidence = []
        
        


boundingboxes = myBoundingBoxes
evaluator = Evaluator()



# Plot Precision x Recall curve
evaluator.PlotPrecisionRecallCurve(
    boundingboxes,  # Object containing all bounding boxes (ground truths and detections)
    IOUThreshold=0.3,  # IOU threshold
    method=MethodAveragePrecision.EveryPointInterpolation,  # As the official matlab code
    showAP=True,  # Show Average Precision in the title of the plot
    showInterpolatedPrecision=True, # Plot the interpolated precision curve
    showGraphic=False,
    savePath= PROJECT_ROOT / "reports/figures/" 
    ) 

# Get metrics with PASCAL VOC metrics
metricsPerClass = evaluator.GetPascalVOCMetrics(
    boundingboxes,  # Object containing all bounding boxes (ground truths and detections)
    IOUThreshold=0.3,  # IOU threshold
    method=MethodAveragePrecision.EveryPointInterpolation)  # As the official matlab code
print("Average precision values per class:\n")


AP = []

#  Loop through classes to obtain their metrics
for mc in metricsPerClass:
    # Get metric values per each class
    c = mc['class']
    precision = mc['precision']
    recall = mc['recall']
    average_precision = mc['AP']
    ipre = mc['interpolated precision']
    irec = mc['interpolated recall']
    # Print AP per class
    print('%s: %f' % (c, average_precision))

    AP.append(average_precision)

mAP = np.array(AP).mean()

print('Mean Average Precison: %f' % (mAP))






"""
gt_boundingBox_5 = BoundingBox(imageName='000003', classId='bench', x=0.546, y=0.48133333333333334,
                               w=0.136, h=0.13066666666666665, typeCoordinates=CoordinatesType.Relative,
                               bbType=BBType.GroundTruth, format=BBFormat.XYWH, imgSize=(500,375))
# Detected bounding boxes of 000001.jpg
detected_boundingBox_1 = BoundingBox(imageName='000001', classId='person', classConfidence= 0.893202, 
                                     x=52, y=4, w=352, h=442, typeCoordinates=CoordinatesType.Absolute, 
                                     bbType=BBType.Detected, format=BBFormat.XYX2Y2, imgSize=(353,500))


gt_boundingBox_1 = BoundingBox(imageName='000003', classId='person', 
                                x=100, y=100, w=30, h=30, typeCoordinates=CoordinatesType.Absolute,
                               bbType=BBType.GroundTruth, format=BBFormat.XYWH, imgSize=(500,375))

gt_boundingBox_2 = BoundingBox(imageName='000003', classId='bench', 
                                x=150, y=150, w=30, h=30, typeCoordinates=CoordinatesType.Absolute,
                               bbType=BBType.GroundTruth, format=BBFormat.XYWH, imgSize=(500,375))



detected_boundingBox_0 = BoundingBox(imageName='000003', classId='bench', classConfidence= 0.901,
                                    x=150, y=150, w=27, h=27, typeCoordinates=CoordinatesType.Absolute,
                                    bbType=BBType.Detected, format=BBFormat.XYWH, imgSize=(500,375))

detected_boundingBox_01 = BoundingBox(imageName='000003', classId='bench', classConfidence= 0.91,
                                    x=300, y=300, w=25, h=25, typeCoordinates=CoordinatesType.Absolute,
                                    bbType=BBType.Detected, format=BBFormat.XYWH, imgSize=(500,375))



detected_boundingBox_1 = BoundingBox(imageName='000003', classId='person', classConfidence= 0.7,
                                    x=100, y=100, w=28, h=28, typeCoordinates=CoordinatesType.Absolute,
                                    bbType=BBType.Detected, format=BBFormat.XYWH, imgSize=(500,375))

detected_boundingBox_2 = BoundingBox(imageName='000003', classId='person', classConfidence= 0.8,
                                    x=10, y=10, w=9, h=9, typeCoordinates=CoordinatesType.Absolute,
                                    bbType=BBType.Detected, format=BBFormat.XYWH, imgSize=(500,375))

detected_boundingBox_3 = BoundingBox(imageName='000003', classId='person', classConfidence= 0.9,
                                    x=300, y=300, w=25, h=25, typeCoordinates=CoordinatesType.Absolute,
                                    bbType=BBType.Detected, format=BBFormat.XYWH, imgSize=(500,375))



myBoundingBoxes = BoundingBoxes()

# Add all bounding boxes to the BoundingBoxes object:
myBoundingBoxes.addBoundingBox(gt_boundingBox_1)
myBoundingBoxes.addBoundingBox(gt_boundingBox_2)

myBoundingBoxes.addBoundingBox(detected_boundingBox_0)
myBoundingBoxes.addBoundingBox(detected_boundingBox_01)
myBoundingBoxes.addBoundingBox(detected_boundingBox_1)
myBoundingBoxes.addBoundingBox(detected_boundingBox_2)
myBoundingBoxes.addBoundingBox(detected_boundingBox_3)


boundingboxes = myBoundingBoxes


# Uncomment the line below to generate images based on the bounding boxes
# createImages(dictGroundTruth, dictDetected)
# Create an evaluator object in order to obtain the metrics
evaluator = Evaluator()

##############################################################
# VOC PASCAL Metrics
##############################################################

# Plot Precision x Recall curve
evaluator.PlotPrecisionRecallCurve(
    boundingboxes,  # Object containing all bounding boxes (ground truths and detections)
    IOUThreshold=0.3,  # IOU threshold
    method=MethodAveragePrecision.EveryPointInterpolation,  # As the official matlab code
    showAP=True,  # Show Average Precision in the title of the plot
    showInterpolatedPrecision=True)  # Plot the interpolated precision curve

# Get metrics with PASCAL VOC metrics
metricsPerClass = evaluator.GetPascalVOCMetrics(
    boundingboxes,  # Object containing all bounding boxes (ground truths and detections)
    IOUThreshold=0.3,  # IOU threshold
    method=MethodAveragePrecision.EveryPointInterpolation)  # As the official matlab code
print("Average precision values per class:\n")


AP = []

#  Loop through classes to obtain their metrics
for mc in metricsPerClass:
    # Get metric values per each class
    c = mc['class']
    precision = mc['precision']
    recall = mc['recall']
    average_precision = mc['AP']
    ipre = mc['interpolated precision']
    irec = mc['interpolated recall']
    # Print AP per class
    print('%s: %f' % (c, average_precision))

    AP.append(average_precision)

mAP = np.array(AP).mean()

print('Mean Average Precison: %f' % (mAP))
"""