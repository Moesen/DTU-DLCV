

###########################################################################################
#                                                                                         #
# This sample shows how to evaluate object detections applying the following metrics:     #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012                   #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012                   #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: May 24th 2018                                                 #
###########################################################################################

#import _init_paths
#import cv2

from projects.project12.src.metrics.BoundingBox import BoundingBox
from projects.project12.src.metrics.BoundingBoxes import BoundingBoxes
from projects.project12.src.metrics.Evaluator import *
from projects.project12.src.metrics.utils import *
import glob
import os
import numpy as np





gt_boundingBox_5 = BoundingBox(imageName='000003', classId='bench', x=0.546, y=0.48133333333333334,
                               w=0.136, h=0.13066666666666665, typeCoordinates=CoordinatesType.Relative,
                               bbType=BBType.GroundTruth, format=BBFormat.XYWH, imgSize=(500,375))
# Detected bounding boxes of 000001.jpg
detected_boundingBox_1 = BoundingBox(imageName='000001', classId='person', classConfidence= 0.893202, 
                                     x=52, y=4, w=352, h=442, typeCoordinates=CoordinatesType.Absolute, 
                                     bbType=BBType.Detected, format=BBFormat.XYX2Y2, imgSize=(353,500))


myBoundingBoxes = BoundingBoxes()
# Add all bounding boxes to the BoundingBoxes object:
myBoundingBoxes.addBoundingBox(gt_boundingBox_5)
myBoundingBoxes.addBoundingBox(detected_boundingBox_1)


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

