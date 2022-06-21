#from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

import datetime
import os
from collections import defaultdict
from glob import glob

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#from projects.project3.src.data.simple_dataloader import basic_loader
from projects.project3.src.data.dataloader import IsicDataSet
from projects.project3.src.features.Memory import get_model_memory_usage
from projects.project3.src.metrics.eval_metrics import *
from projects.project3.src.metrics.losses import *
from projects.project3.src.models.Networks import Pix2Pix_Unet
from projects.utils import get_project3_root
from tensorflow.python.client import device_lib
from tqdm import tqdm

# built tensorflow with GPU

print("TENSORFLOW BUILT WITH CUDA: ", tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

print("TENSORFLOW VISIBLE DEVIES: ", device_lib.list_local_devices())

method = "GPU"

if method == "GPU":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if len(tf.config.list_physical_devices("GPU")) > 0:
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True





if __name__ == '__main__':
    
    BATCH_SIZE = 8
    IMG_SIZE = (256,256) #(256,256,3)
    GF = 32

    ##### TRAIN MODEL ##### 
    save_model = True

    num_epochs = 80

    proot = get_project3_root()
    data_root = proot / "data/isic/train_allstyles"
    image_path = data_root / "Images"
    mask_path0 = proot / "data/isic/train_style0/Segmentations"
    mask_path1 = proot / "data/isic/train_style1/Segmentations"
    mask_path2 = proot / "data/isic/train_style2/Segmentations"
    mask_path_all = data_root / "Segmentations"

    mask_path_list = [mask_path0, mask_path1, mask_path2]#, mask_path_all]
    style_names = ["0","1","2","all"]

    for sn, mask_path in zip(style_names, mask_path_list):

        print("Initiating model for style " + sn)

        unet = Pix2Pix_Unet(loss_f=weighted_cross_entropy(),#focal_loss(),#tf.keras.losses.BinaryCrossentropy(),  #tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                    train_dataset=[],
                    test_data=[],
                    img_size=(*IMG_SIZE, 3),
                    gf=GF,
                    num_conv=1,
                    depth=5,
                    batchnorm=False,
                    dropout_percent=0.3,
                    )

        unet.unet.summary()
        print("MEMORY USAGE in GB: ",get_model_memory_usage(BATCH_SIZE, unet.unet))


        dataset_loader = IsicDataSet(
            image_folder=image_path,
            mask_folder=mask_path,
            image_size=(256, 256),
            image_channels=3,
            mask_channels=1,
            image_file_extension="jpg",
            mask_file_extension="png",
            do_normalize=True,
            output_image_path=False,
            validation_percentage=.2,
            seed=69,
            flipping="vertical",
            rotation=0.2,
            brightness=0.1,
            #hue=0.05,
        )

        train_dataset, val_dataset = dataset_loader.get_dataset(batch_size=BATCH_SIZE, shuffle=True)

        early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=1,
        mode="min",
        restore_best_weights=True,
        )

        model_history = unet.unet.fit(train_dataset, epochs=num_epochs,validation_data=val_dataset,callbacks=early_stopping)
        
        # Saving model
        print("Saving model...")
        model_name = 'unet_'+sn+'_'+datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        if save_model:
            model_path = proot / "models" / model_name
            unet.unet.save(model_path)



    # Compute IoU for the final model
    """total_iou = []
    print("Computing final metrics...")
    for (x_batch_val, true_mask) in val_dataset:
        for (val_img, val_GT_mask) in zip(x_batch_val, true_mask):
            val_logits = unet.unet(tf.expand_dims(val_img, 0), training=False)
            val_probs = tf.keras.activations.sigmoid(val_logits)
            pred_mask = tf.math.round(val_probs)

            compute_IoU = tf.keras.metrics.BinaryIoU()#tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])
            batch_iou = compute_IoU(pred_mask, val_GT_mask)

            total_iou.append( batch_iou )

    print("IoU for entire validation set: ",np.array(total_iou).mean())"""

    
    