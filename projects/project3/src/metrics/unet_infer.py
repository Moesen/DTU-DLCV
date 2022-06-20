
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

## load the base model
PROJECT_ROOT = get_project3_root()
model_name = "unet_20220618123600"
model_path = PROJECT_ROOT / "models" / model_name

loss_fn = focal_loss()

unet = tf.keras.models.load_model(model_path, custom_objects={"loss": loss_fn })
unet.summary()

### the models trained on other segmentation types 
model_name0 = "unet0_20220619215224"
model_path0 = PROJECT_ROOT / "models" / model_name0
unet0 = tf.keras.models.load_model(model_path0, custom_objects={"loss": loss_fn })

model_name1 = "unet1_20220619205508"
model_path1 = PROJECT_ROOT / "models" / model_name1
unet1 = tf.keras.models.load_model(model_path1, custom_objects={"loss": loss_fn })

model_name2 = "unet2_20220619212343"
model_path2 = PROJECT_ROOT / "models" / model_name2
unet2 = tf.keras.models.load_model(model_path2, custom_objects={"loss": loss_fn })


unet_models = [unet, unet0, unet1, unet2]
unet_seg_type = ["All", "Type 0", "Type 1", "Type 2"]
###


#Initialize dataloader
BATCH_SIZE = 16
IMG_SIZE = (256,256)

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
    validation_percentage=.1,
    seed=69,
)

test_dataset, _ = dataset_loader.get_dataset(batch_size=BATCH_SIZE, shuffle=False)

print("Loading first batch...")
test_img, mask = list(iter(test_dataset))[0]

#use these images 
idx = tf.constant([0,4,8,12])
test_img_plot1 = tf.gather(test_img, idx)
mask_img_plot1 = tf.gather(mask, idx)

"""img_list = []
mask_list = []
for _ in range(len(unet_models)):
    img_list.append(test_img_plot1)
    mask_list.append(mask_img_plot1)

test_img_plot = tf.concat(img_list, axis=0)
mask_img_plot = tf.concat(mask_list, axis=0)"""

test_img_plot = test_img_plot1
mask_img_plot = mask_img_plot1

print("Plotting...")
fig, axs = plt.subplots(len(unet_models), 4, figsize=(15,15) )


for m, model in enumerate(unet_models):
    print("Plotting model ",m)
    img_np = []
    for n,(img, mask) in enumerate(zip(test_img_plot, mask_img_plot)):

        #make segmentation predictions
        pred_logits = model.predict(tf.expand_dims(img, 0))
        pred_probs = tf.keras.activations.sigmoid(pred_logits)
        pred_mask = tf.math.round(pred_probs)

        img_np = img.numpy()
        mask_np = mask.numpy()

        #change color for boundary of GT mask 
        out, b_idx = get_boundary(mask_np, is_GT=True)
        img_np[b_idx>1,:] = out[b_idx>1,:]
        
        #change color for bounadry of prediction
        out, b_idx = get_boundary(pred_mask.numpy().squeeze(), is_GT=False)
        img_np[b_idx>1,:] = out[b_idx>1,:]

        axs[m,n].imshow(img_np / 255.)

        if n==0:
            axs[m,n].set_ylabel(unet_seg_type[m], rotation='horizontal', fontsize=16, ha='right')

        #make manual legend
        if n==2 and m==len(unet_models)-1:
            gc = mpl.colors.to_rgba((0,1,0))
            rc = mpl.colors.to_rgba((1,0,0))
            green_patch = mpatches.Patch(color=gc, label='GT')
            red_patch = mpatches.Patch(color=rc, label='Pred')
            axs[m,n].legend(handles=[green_patch,red_patch], bbox_to_anchor=(0.2, -0.05), ncol=2)

        #plot the iou and area difference in title
        compute_IoU = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])
        pred_mask = tf.cast( pred_mask, tf.uint32)
        mask = tf.cast( mask, tf.uint32) 
        img_iou = compute_IoU(pred_mask, mask)

        breakpoint()
        n_seg_pixels_mask = tf.math.reduce_sum(mask).numpy()
        n_seg_pixels_pred = tf.math.reduce_sum(pred_mask).numpy()

        p_diff = (n_seg_pixels_pred - n_seg_pixels_mask) / n_seg_pixels_mask

        axs[m,n].set_title(f"IoU: {img_iou:.2f}, Area diff: {p_diff:.2f}",fontsize=16,x=0.5,y=1.05)
        axs[m,n].grid(False)
        axs[m,n].get_xaxis().set_ticks([])
        axs[m,n].get_yaxis().set_ticks([])
        #ax.axis('off')

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)

fig_path = PROJECT_ROOT / "reports/figures/1_boundary.png"
plt.savefig(fig_path,bbox_inches='tight')


#plot a figure of the GT segmentations to check implementation
fig, axs = plt.subplots(1,4, figsize=(15,8))
for n, (mask, ax) in enumerate(zip(mask_img_plot, axs.ravel())):
    ax.imshow(mask)
fig_path = PROJECT_ROOT / "reports/figures/GT_boundary.png"
plt.savefig(fig_path)


#compute metrics for model

print("Computing final metrics...")
for m, model in enumerate(unet_models):
    total_iou = []
    total_p_diff = []
    print("... for model " + unet_seg_type[m])
    for (x_batch_val, true_mask) in test_dataset:
        for (val_img, val_GT_mask) in zip(x_batch_val, true_mask):
            val_logits = model(tf.expand_dims(val_img, 0), training=False)
            val_probs = tf.keras.activations.sigmoid(val_logits)
            pred_mask = tf.math.round(val_probs)

            compute_IoU = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])
            batch_iou = compute_IoU(pred_mask, val_GT_mask)
            total_iou.append( batch_iou )

            n_seg_pixels_mask = tf.math.reduce_sum(val_GT_mask).numpy()
            n_seg_pixels_pred = tf.math.reduce_sum(pred_mask).numpy()
            p_diff = (n_seg_pixels_pred - n_seg_pixels_mask) / n_seg_pixels_mask
            total_p_diff.append( p_diff )

    print(unet_seg_type[m])
    print("IoU for entire test set: ",np.array(total_iou).mean())
    print("Pixel diff entire test set: ",np.array(total_p_diff).mean())

