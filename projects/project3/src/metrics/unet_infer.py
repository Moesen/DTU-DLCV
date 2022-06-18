
import tensorflow as tf
import keras 

from glob import glob
import os 

import matplotlib.pyplot as plt
import numpy as np
#import cv2

from tqdm import tqdm
from collections import defaultdict


from projects.utils import get_project3_root
#from projects.project3.src.data.simple_dataloader import basic_loader
from projects.project3.src.data.dataloader import IsicDataSet
from projects.project3.src.models.Networks import Pix2Pix_Unet
from projects.project3.src.metrics.losses import *
from projects.project3.src.metrics.eval_metrics import *



PROJECT_ROOT = get_project3_root()
model_name = ""
model_path = PROJECT_ROOT / "models" / model_name

#new_model = tf.keras.models.load_model(model_path)

# Check its architecture
#new_model.summary()


BATCH_SIZE = 16
IMG_SIZE = (256,256) #(256,256,3)

proot = get_project3_root()
data_root = proot / "data/isic/test_style0"
image_path = data_root / "Images"
mask_path = data_root / "Segmentations"


dataset_loader = IsicDataSet(
    image_folder=image_path,
    mask_folder=mask_path,
    image_size=IMG_SIZE,
    image_channels=3,
    mask_channels=1,
    image_file_extension="jpg",
    mask_file_extension="png",
    do_normalize=True,
#    validation_percentage=.2
)


test_dataset = dataset_loader.get_dataset(batch_size=BATCH_SIZE, shuffle=False)


(test_img, mask) = next(iter(test_dataset))

test_img_plot = test_img
mask_img_plot = mask


fig, axs = plt.subplots(1,4, figsize=(8,15))

import matplotlib.colors as colors
palette = plt.cm.gray.with_extremes(over='r', under='g', bad='b')

for (img, mask, ax) in zip(test_img_plot.numpy(), mask_img_plot.numpy(), axs.ravel()):
    ax.imshow(img)

    ax.imshow(mask, interpolation='nearest',
                cmap=palette,
                norm=colors.BoundaryNorm([-1, -0.5, -0.2, 0, 0.2, 0.5, 1],
                                         ncolors=palette.N),
                aspect='auto',
                origin='lower',
                extent=[x0, x1, y0, y1])

    iou = 1
    ax.set_title(f"Prediction: {iou:.2f}",fontsize=24,x=0.5,y=1.05)


plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
plt.show()

fig_path = PROJECT_ROOT / "reports/figures/1_boundary.png"
plt.savefig(fig_path)


"""hotdog_certain_idx = np.argmax( probs[:,0] )
nothotdog_certain_idx = np.argmax( probs[:,1] )

confidences = np.max(probs,1)
uncertain_idx = np.argpartition(np.abs(confidences-0.5), 2)[:2]
#uncertain_idx = np.argmin( np.abs(confidences-0.5) )

idx2plot = K.cast(np.concatenate((hotdog_certain_idx, nothotdog_certain_idx, uncertain_idx), axis=None), tf.int32)

test_img_plot = tf.gather(test_img, idx2plot)

probs_plot = probs[idx2plot.numpy().tolist(),:]
predicted_plot = predicted[idx2plot.numpy().tolist()]"""





