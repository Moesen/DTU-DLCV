import ssl
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import backend as K
from projects.utils import get_project12_root
from tensorflow import keras


def NMS(BB, predicted, probs, classes, iout = 0.5, st = 0.2, max_out = 10):

    # NMS post processing 
    iou_threshold = tf.convert_to_tensor(iout,dtype=tf.float32)
    score_threshold = tf.convert_to_tensor(st,dtype=tf.float32)
    max_output_size = tf.convert_to_tensor(max_out,dtype=tf.int32)
    scores = tf.convert_to_tensor(probs.numpy().max(1),dtype=tf.float32)
    predicted = tf.squeeze(predicted)

    c_selected_boxes = []
    c_selected_probs = []
    c_selected_preds = []

    for c in classes:

        idx = K.cast(np.where(predicted.numpy()==c)[0], tf.int32)

        if len(idx.numpy())>1:
            pred_c = tf.gather(predicted, idx)
            bb_c = tf.gather(BB, idx)
            scores_c = tf.gather(scores,idx)
        else:
            pred_c = predicted
            bb_c = BB
            scores_c = scores

        selected_indices = tf.image.non_max_suppression(
            bb_c, scores=scores_c, max_output_size=max_output_size, 
            iou_threshold=iou_threshold, score_threshold=score_threshold)
        
        selected_boxes = tf.gather(bb_c, selected_indices).numpy()
        selected_probs = tf.gather(scores_c, selected_indices).numpy()
        selected_preds = tf.gather(pred_c, selected_indices).numpy()

        c_selected_boxes.append(selected_boxes)
        c_selected_probs.append(selected_probs)
        c_selected_preds.append(selected_preds)


    #all NMS processed bb
    c_selected_boxes_filtered = [y for y in c_selected_boxes if 0 not in y.shape]
    if len(c_selected_boxes_filtered) > 0:
        c_selected_boxes_filtered = np.vstack(c_selected_boxes_filtered)
    else:
        c_selected_boxes_filtered = np.array(c_selected_boxes_filtered)

    all_selected_boxes = K.cast(tf.convert_to_tensor(c_selected_boxes_filtered), tf.float32)

    #all NMS processed probs
    c_selected_probs_filtered = [y for y in c_selected_probs if 0 not in y.shape]
    if len(c_selected_probs_filtered) > 0:
        c_selected_probs_filtered = np.hstack(c_selected_probs_filtered)
    else:
        c_selected_probs_filtered = np.array(c_selected_probs_filtered)
    all_selected_probs = K.cast(tf.convert_to_tensor(c_selected_probs_filtered), tf.float32)

    #all NMS processed preds (class predictions)
    c_selected_preds_filtered = [y for y in c_selected_preds if 0 not in y.shape]
    if len(c_selected_preds_filtered) > 0:
        c_selected_preds_filtered = np.hstack(c_selected_preds_filtered)
    else:
        c_selected_preds_filtered = np.array(c_selected_preds_filtered)
    all_selected_preds = K.cast(tf.convert_to_tensor(c_selected_preds_filtered), tf.int32)

    return all_selected_boxes, all_selected_probs, all_selected_preds