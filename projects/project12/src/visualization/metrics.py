
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
import matplotlib.patches as patches
from projects.project11.src.data.dataloader import load_dataset
from projects.project12.src.models.post_processing import NMS

from projects.utils import get_project12_root

from projects.project12.src.metrics.BoundingBox import BoundingBox
from projects.project12.src.metrics.BoundingBoxes import BoundingBoxes
from projects.project12.src.metrics.Evaluator import *
from projects.project12.src.metrics.utils import *


PROJECT_ROOT = get_project12_root()
model_name = 'hotdog_conv_20220604214318' #'hotdog_conv_20220604190940'
#model_path = PROJECT_ROOT / "models" / model_name
model_path = "/Users/simonyamazaki/Documents/2_M/DTU-DLCV/projects/project1/models/hotdog_conv_20220604214318"

new_model = tf.keras.models.load_model(model_path)

# Check its architecture
new_model.summary()


### Mean average Precision (mAP) 

batch_size = 2
img_size = (64,64)

test_data = load_dataset(
        train=False,
        normalize=True,
        batch_size=batch_size,
        tune_for_perfomance=False,
        image_size=img_size,
    )

labels = test_data._input_dataset.class_names
b, classes = np.unique(labels, return_inverse=True)


BB_all_predicted = []
bb_class = []
bb_confidence = []

img_id_now = -1 

myBoundingBoxes = BoundingBoxes()

#### LOOP ####
for _ in range(1):
    (bb_img, y) = next(iter(test_data))

    img_id = 0
    GT_bb = np.array([[30,30,15,15], [32,32,15,15]])
    GT_class = ["hotdog","hotdog"] #[1,1]

    imgSize = tuple(bb_img.shape[1:3])

    #predicted proposal bounding boxes in the current batch
    BB = np.array([[30,30,15,15], [32,32,15,15]])

    #compute classification predictions on the bb cropped images with bounding boxes = BB
    logits = new_model(bb_img, training=False)
    probs = tf.nn.softmax(logits,axis=1)
    predicted = K.cast(K.argmax(logits, axis=1), "uint8")

    BB_all_predicted.append(BB)
    bb_confidence.append( probs.numpy().max(1) )
    bb_class.append( predicted.numpy() )

    #only perform the below if all bounding boxes are predicted in a single image
    if True: #img_id_now != img_id:
        #add GT bbs 
        for gtb,gtc in zip(GT_bb,GT_class):
            gt_boundingBox = BoundingBox(imageName=img_id, classId=gtc,#'person', 
                                    x=gtb[0], y=gtb[1], w=gtb[2], h=gtb[3], typeCoordinates=CoordinatesType.Absolute,
                                bbType=BBType.GroundTruth, format=BBFormat.XYWH, imgSize=imgSize)
            myBoundingBoxes.addBoundingBox(gt_boundingBox)

        #reformat from shape (n_batches, batch_size, 4) -> (n_batches*batch_size, 4) = (2000,4)
        BB_all_predicted = np.array(BB_all_predicted)
        BB_all_predicted = BB_all_predicted.reshape((-1,BB_all_predicted.shape[-1]))

        #reformat from shape (n_batches, batch_size, 1) -> (n_batches*batch_size, 1) = (2000,1)
        bb_class = np.array(bb_class)
        bb_class = bb_class.reshape((-1,bb_class.shape[-1]))

        #reformat from shape (n_batches, batch_size, 1) -> (n_batches*batch_size, 1) = (2000,1)
        bb_confidence = np.array(bb_confidence)
        bb_confidence = bb_confidence.reshape((-1,bb_confidence.shape[-1]))

        #cast to tensorflow tensors
        BB_all_predicted = tf.convert_to_tensor(np.array(BB_all_predicted),dtype=tf.float32)
        bb_confidence = tf.convert_to_tensor(np.array(bb_confidence),dtype=tf.float32)
        bb_class = tf.convert_to_tensor(np.array(bb_class),dtype=tf.float32)

        # NMS post processing
        bb_selected, bb_confidence_selected, bb_class_selected = NMS(BB_all_predicted, bb_class, bb_confidence, classes, iout = 0.5, st = 0.2, max_out = 10)

        bb_labels_selected = [labels[int(i)] for i in bb_class_selected.numpy().squeeze().tolist()]

        #add detected bbs
        for db,dc,conf in zip(bb_selected, bb_labels_selected, bb_confidence_selected):
            detected_boundingBox = BoundingBox(imageName=img_id, classId=dc, classConfidence=conf,
                                    x=db[0], y=db[1], w=db[2], h=db[3], typeCoordinates=CoordinatesType.Absolute,
                                    bbType=BBType.Detected, format=BBFormat.XYWH, imgSize=imgSize)

            myBoundingBoxes.addBoundingBox(detected_boundingBox)

        BB_all_predicted = []
        bb_class = []
        bb_confidence = []
        
    #img_id_now = img_id


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