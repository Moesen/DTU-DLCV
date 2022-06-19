
import tensorflow as tf
import keras 

from glob import glob
import os 
os.environ['DISPLAY'] = ':0'

import matplotlib.patches as mpatches
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cv2

from tqdm import tqdm
from collections import defaultdict
from PIL import Image, ImageFilter

from projects.utils import get_project3_root
#from projects.project3.src.data.simple_dataloader import basic_loader
from projects.project3.src.data.dataloader import IsicDataSet
from projects.project3.src.models.Networks import Pix2Pix_Unet
from projects.project3.src.metrics.losses import *
from projects.project3.src.metrics.eval_metrics import *
from projects.project3.src.visualization.make_boundary import get_boundary


PROJECT_ROOT = get_project3_root()
model_name = "unet_20220618123600"
model_path = PROJECT_ROOT / "models" / model_name

loss_fn = focal_loss()

unet = tf.keras.models.load_model(model_path, custom_objects={"loss": loss_fn })
unet.summary()


BATCH_SIZE = 16
IMG_SIZE = (256,256) #(256,256,3)

proot = get_project3_root()
data_root = proot / "data/isic/test_style0" #train_allstyles" #test_style0"
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
    validation_percentage=.2
)


train_dataset, val_dataset = dataset_loader.get_dataset(batch_size=BATCH_SIZE, shuffle=False)

test_img, mask = list(iter(val_dataset))[1]

#use these images 
idx = tf.constant([0,4,8,12])
test_img_plot = tf.gather(test_img, idx)
mask_img_plot = tf.gather(mask, idx)

print("Plotting...")
fig, axs = plt.subplots(1,4, figsize=(15,8))


for (img, mask, ax) in zip(test_img_plot, mask_img_plot, axs.ravel()):
    
    pred_logits = unet.predict(tf.expand_dims(img, 0))
    pred_probs = tf.keras.activations.sigmoid(pred_logits)
    pred_mask = tf.math.round(pred_probs)

    img = img.numpy()
    mask = mask.numpy()

    #change color for boundary of GT mask 
    out, b_idx = get_boundary(mask, is_GT=True)
    img[b_idx>1,:] = out[b_idx>1,:]
    
    #change color for bounadry of prediction
    out, b_idx = get_boundary(pred_mask.numpy().squeeze(), is_GT=False)
    img[b_idx>1,:] = out[b_idx>1,:]

    gc = mpl.colors.to_rgba((0,1,0))
    rc = mpl.colors.to_rgba((1,0,0))

    green_patch = mpatches.Patch(color=gc, label='GT')
    red_patch = mpatches.Patch(color=rc, label='Pred')
    ax.legend(handles=[green_patch,red_patch],bbox_to_anchor=(1.2, 0.5))

    ax.imshow(img)
    #ax.get_xaxis().set_ticks([])
    #ax.get_yaxis().set_ticks([])

    compute_IoU = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])
    img_iou = compute_IoU(pred_mask, mask)

    ax.set_title(f"IoU: {img_iou:.2f}",fontsize=16,x=0.5,y=1.05)
    ax.grid(False)
    ax.axis('off')

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)

fig_path = PROJECT_ROOT / "reports/figures/1_boundary.png"
plt.savefig(fig_path)


fig, axs = plt.subplots(1,4, figsize=(15,8))

for n, (mask, ax) in enumerate(zip(mask_img_plot, axs.ravel())):
    ax.imshow(mask)

fig_path = PROJECT_ROOT / "reports/figures/GT_boundary.png"
plt.savefig(fig_path)

#compute metrics for model
total_iou = []

## DO THIS PER IMAGE INSTEAD
for (x_batch_val, true_mask) in val_dataset:
    for (val_img, val_GT_mask) in zip(x_batch_val, true_mask):
        val_logits = unet(tf.expand_dims(val_img, 0), training=False)
        val_probs = tf.keras.activations.sigmoid(val_logits)
        pred_mask = tf.math.round(val_probs)

        compute_IoU = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])
        batch_iou = compute_IoU(pred_mask, val_GT_mask)
        total_iou.append( batch_iou )

print("IoU for entire test set: ",np.array(total_iou).mean())

