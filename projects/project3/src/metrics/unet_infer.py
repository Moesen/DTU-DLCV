
import tensorflow as tf
import keras 

from glob import glob
import os 
os.environ['DISPLAY'] = ':0'

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



PROJECT_ROOT = get_project3_root()
model_name = "unet_20220618123600"
model_path = PROJECT_ROOT / "models" / model_name

loss_fn = focal_loss()

unet = tf.keras.models.load_model(model_path, custom_objects={"loss": loss_fn })
unet.summary()


BATCH_SIZE = 16
IMG_SIZE = (256,256) #(256,256,3)

proot = get_project3_root()
data_root = proot / "data/isic/train_allstyles" #test_style0"
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

idx = tf.constant([0,4,8,12])

#test_img_plot = tf.image.resize( tf.gather(test_img, idx), [256,256])
#mask_img_plot = tf.image.resize( tf.gather(mask, idx), [256,256])
test_img_plot = tf.gather(test_img, idx)
mask_img_plot = tf.gather(mask, idx)

print("Plotting...")
fig, axs = plt.subplots(1,4, figsize=(15,8))

for (img, mask, ax) in zip(test_img_plot.numpy(), mask_img_plot.numpy(), axs.ravel()):
    
    """mask = Image.fromarray(mask.squeeze(),mode="L")
    #img_gray = mask.convert("L")
    edges = mask.filter(ImageFilter.FIND_EDGES)
    e = np.asarray(edges)
    edge_coord = np.where(e==255)
    plt.plot(edge_coord[0], edge_coord[1], color="red", linewidth=3)"""

    contours, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    out = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    k = -1
    for i, cnt in enumerate(contours):
        if (hier[0, i, 3] == -1):
            k += 1
        cv2.drawContours(out, [cnt], -1, colors[k], 2)


    #cv2.imshow('out', out)
    img[out>0] = out[out>0]

    ax.imshow(img)
    #ax.get_xaxis().set_ticks([])
    #ax.get_yaxis().set_ticks([])

    #out2 = Image.fromarray(out).convert("L")

    #e = np.asarray(out2)
    #e[e>0] = 255
    #edge_coord = np.squeeze(np.where(e == 255))

    #plt.scatter(edge_coord[0,:], edge_coord[1,:], color="red")

    iou = 1
    ax.set_title(f"Prediction: {iou:.2f}",fontsize=24,x=0.5,y=1.05)
    ax.grid(False)
    ax.axis('off')

#plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)

fig_path = PROJECT_ROOT / "reports/figures/1_boundary.png"
plt.savefig(fig_path)

#compute metrics for model
total_iou = []

for (x_batch_val, true_mask) in val_dataset:
    val_logits = unet(x_batch_val, training=False)
    val_probs = tf.keras.activations.sigmoid(val_logits)
    pred_mask = tf.math.round(val_probs)

    compute_IoU = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])
    batch_iou = compute_IoU(pred_mask, true_mask)
    total_iou.append( batch_iou )

print("IoU for entire test set: ",np.array(total_iou).mean())





