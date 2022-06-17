#from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

import tensorflow as tf
from tensorflow.python.client import device_lib
import keras 

from glob import glob
import datetime
import os 

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from collections import defaultdict


from projects.utils import get_project3_root
#from projects.project3.src.data.simple_dataloader import basic_loader
from projects.project3.src.data.dataloader import IsicDataSet
from projects.project3.src.models.Networks import Pix2Pix_Unet
from projects.project3.src.metrics.losses import *
from projects.project3.src.metrics.eval_metrics import *



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
    GF = 16

    proot = get_project3_root()
    data_root = proot / "data/isic/train_allstyles"
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
    )

    train_dataset = dataset_loader.get_dataset(batch_size=BATCH_SIZE, shuffle=True)


    ##### TRAIN MODEL ##### 
    save_model = False

    num_epochs = 2
    sample_img_interval = 20

    #metric = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0]) #keras.metrics.SparseCategoricalAccuracy()

    unet = Pix2Pix_Unet(loss_f=focal_loss(),  #tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                        train_dataset=train_dataset,
                        test_data=[],
                        img_size=(*IMG_SIZE, 3),
                        gf=GF,
                        )

    unet.unet.summary()

    model_history = unet.unet.fit(train_dataset, epochs=num_epochs)
    
    #unet.train(epochs=num_epochs,sample_interval_epoch=sample_img_interval )

    final_preds = unet.unet.predict(train_dataset)

    breakpoint()

    model_name = 'unet_'+datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    if save_model:
        model_path = proot / "models" / model_name
        unet.save(model_path)
    
    ######
    #model_history = unet.unet.fit(dataset['train'], epochs=num_epochs,)
    #                                validation_data=dataset['val'])
    ######
    