import tensorflow as tf

from keras import backend as K
from src.data.dataloader import load_dataset
from src.utils import get_project_root
from tensorflow import keras
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import numpy as np 
import ssl
ssl._create_default_https_context = ssl._create_unverified_context








def bb_intersection_over_union(boxA, boxB):
    #boxes should be tensorflow tensors 

	# determine the (x, y)-coordinates of the intersection rectangle
	xA = K.maximum(boxA[0], boxB[0])
	yA = K.maximum(boxA[1], boxB[1])
	xB = K.minimum(boxA[2], boxB[2])
	yB = K.minimum(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = K.maximum(0, xB - xA + 1) * K.maximum(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / K.cast(boxAArea + boxBArea - interArea,"float32")
	# return the intersection over union value
	return iou




from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

def compute_bb_iou(boxes_a, boxes_b):
  """Calculates the overlap (iou - intersection over union) between boxes_a and boxes_b.
  Args:
    boxes_a: a tensor with a shape of [batch_size, N, 4]. N is the number of
      boxes per image. The last dimension is the pixel coordinates in
      [ymin, xmin, ymax, xmax] form.
    boxes_b: a tensor with a shape of [batch_size, M, 4]. M is the number of
      boxes. The last dimension is the pixel coordinates in
      [ymin, xmin, ymax, xmax] form.
  Returns:
    intersection_over_union: a tensor with as a shape of [batch_size, N, M],
    representing the ratio of intersection area over union area (IoU) between
    two boxes
  """
  with ops.name_scope('bbox_overlap'):
    a_y_min, a_x_min, a_y_max, a_x_max = array_ops.split(
        value=boxes_a, num_or_size_splits=4, axis=2)
    b_y_min, b_x_min, b_y_max, b_x_max = array_ops.split(
        value=boxes_b, num_or_size_splits=4, axis=2)

    # Calculates the intersection area.
    i_xmin = math_ops.maximum(
        a_x_min, array_ops.transpose(b_x_min, [0, 2, 1]))
    i_xmax = math_ops.minimum(
        a_x_max, array_ops.transpose(b_x_max, [0, 2, 1]))
    i_ymin = math_ops.maximum(
        a_y_min, array_ops.transpose(b_y_min, [0, 2, 1]))
    i_ymax = math_ops.minimum(
        a_y_max, array_ops.transpose(b_y_max, [0, 2, 1]))
    i_area = math_ops.maximum(
        (i_xmax - i_xmin), 0) * math_ops.maximum((i_ymax - i_ymin), 0)

    # Calculates the union area.
    a_area = (a_y_max - a_y_min) * (a_x_max - a_x_min)
    b_area = (b_y_max - b_y_min) * (b_x_max - b_x_min)
    EPSILON = 1e-8
    # Adds a small epsilon to avoid divide-by-zero.
    u_area = a_area + array_ops.transpose(b_area, [0, 2, 1]) - i_area + EPSILON

    # Calculates IoU.
    intersection_over_union = i_area / u_area

    return intersection_over_union