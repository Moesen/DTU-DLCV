from ctypes.wintypes import PLARGE_INTEGER

import matplotlib as mpl
import matplotlib.patches as patches
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

import os 
os.chdir("/Users/simonyamazaki/Documents/2_M/DTU-DLCV/projects/project1/")

PROJECT_ROOT = get_project12_root()
model_name = 'hotdog_conv_20220604214318' #'hotdog_conv_20220604190940'
#model_path = PROJECT_ROOT / "models" / model_name
model_path = "/Users/simonyamazaki/Documents/2_M/DTU-DLCV/projects/project1/models/hotdog_conv_20220604214318"

new_model = tf.keras.models.load_model(model_path)

# Check its architecture
new_model.summary()

batch_size=1
img_size = (64,64)

test_data = load_dataset(
        train=False,
        normalize=True,
        batch_size=batch_size,
        tune_for_perfomance=False,
        image_size=img_size,
    )

(test_img, y) = next(iter(test_data))

test_img = test_img[0,:,:,:]
test_img = tf.expand_dims(test_img, axis=0)
test_img = tf.concat([test_img,test_img], axis=0)

#proposal BB cropped images 
#yield: prop_imgs, BB
# with the first dimension as the number of proposals 

BB = np.array([[30,30,15,15], [32,32,15,15]])
BB = tf.convert_to_tensor(BB,dtype=tf.float32)


logits = new_model(test_img, training=False)
probs = tf.nn.softmax(logits,axis=1)
predicted = K.cast(K.argmax(logits, axis=1), "uint8")


labels = test_data._input_dataset.class_names
b, classes = np.unique(labels, return_inverse=True)

all_selected_boxes, all_selected_probs, all_selected_preds = NMS(BB, predicted, probs, classes, iout = 0.5, st = 0.2, max_out = 10)

fig, axs = plt.subplots(1,2, figsize=(15,8))

cmap = mpl.cm.get_cmap('Spectral')

BBs = [BB,all_selected_boxes]
probss = [probs,all_selected_probs]
predss = [predicted,all_selected_preds]

tits = ["before NMS","after NMS"]

for m in range(test_img.shape[0]):
    axs[m].imshow(test_img[m,:,:,:].numpy().squeeze())

    BB_plot = BBs[m]
    prob_plot = probss[m]
    pred_plot = predss[m]

    for n, (bb,pred,prob) in enumerate(zip(BB_plot.numpy(),pred_plot.numpy().squeeze(),prob_plot.numpy())):
        
        rgba = cmap(n/prob_plot.shape[0])

        rect = patches.Rectangle((bb[0],bb[1]), bb[2], bb[3], linewidth=1,
                            edgecolor=rgba, facecolor="none")
        axs[m].add_patch(rect)

        pred_prob = np.max(prob)
        
        pred_label = labels[int(pred)]
        object_text = f"{pred_label}, p={pred_prob:.2f}"

        axs[m].text(bb[0],bb[1], object_text, color='red', 
            bbox=dict(facecolor='white', edgecolor='black'))

        axs[m].title.set_text(tits[m])



fig_path = PROJECT_ROOT.parent / "project12/reports/figures/Objects_detected.png" 
plt.savefig(fig_path)

