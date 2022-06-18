import ssl

import tensorflow as tf
from keras import backend as K
from keras import regularizers

# img_size_loader = (128,128)
# img_size = (128,128,3)
# batch_size = 64



# train_dataset = load_dataset(
#     train=True,
#     normalize=False,
#     shuffle = True,
#     use_data_augmentation=True,
#     batch_size=batch_size,
#     tune_for_perfomance=False,
#     image_size=img_size_loader,
# )
# test_data = load_dataset(
#     train=False,
#     normalize=False,
#     batch_size=batch_size,
#     tune_for_perfomance=False,
#     use_data_augmentation=False,
#     image_size=img_size_loader,
# )

# #batch_imgs = next(iter(train_dataset))
# #feature_batch = base_model(batch_imgs)


# #base_model.trainable = True

# # Fine-tune from this layer onwards
# #fine_tune_at = 15

# # Freeze all the layers before the `fine_tune_at` layer

# #for layer in base_model.layers[:fine_tune_at]:
# #    layer.trainable = False




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




def create_CNN(IMG_SIZE):

    base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=IMG_SIZE,
        pooling=None,
        classes=2,
        classifier_activation='softmax',
        include_preprocessing=True
        )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE)
    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x) ##
    x = tf.keras.layers.Dense(200,kernel_regularizer=regularizers.l2(1e-3), kernel_initializer='he_normal')(x) ##
    x = tf.keras.layers.Dropout(0.2)(x)
    predictions = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, predictions)


    """x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)"""



    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.summary()
    return model
    
        




if __name__ == '__main__':

    BATCH_SIZE = 16
    IMG_SIZE = (256,256,3)
    GF = 32

    # Example of using dataloader and extracting datasets train and test
    proot = get_project3_root()
    data_root = proot / "data/ISIC18"
    image_path = data_root / "ISIC18_Task3_ims"
    mask_path = data_root / "Segmentations"
    dataset_loader = IsicDataSet(
        image_folder=image_path,
        mask_folder=mask_path,
        image_size=(256, 256),
        image_channels=3,
        mask_channels=1,
        image_file_extension="jpg",
        mask_file_extension="png",
        do_normalize=True,
        segmentation_type="0",
    )

    train_dataset, test_dataset = dataset_loader.get_dataset(batch_size=1, shuffle=True)
    image, mask = next(iter(test_dataset))
    _, [a, b] = plt.subplots(1, 2)
    a.imshow(image[0])
    b.imshow(mask[0])
    b.set_title(f"max val: {tf.reduce_max(mask)}, min val: {tf.reduce_min(mask)}")


    train_dataset, val_dataset = dataset_loader.get_dataset(batch_size=BATCH_SIZE, shuffle=True)
    

    # plt.imshow(train_dataset[0][0][0])
    cnn_model = create_CNN(IMG_SIZE=IMG_SIZE)

    ##### TRAIN MODEL ##### 
    save_model = True

    num_epochs = 100
    sample_img_interval = 20

    history = cnn_model.fit(train_dataset, epochs=100, validation_data=val_dataset)
    #loss0, accuracy0 = cnn_model.evaluate(test_data)
    
    
    # # Compute acc for the final model
    # (x_batch_val, true_mask) = next(iter(val_dataset))
    # pred_logits = unet.unet.predict(x_batch_val)
    # pred_mask = tf.keras.activations.sigmoid(pred_logits)

    # compute_IoU = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])
    # best_iou = compute_IoU(pred_mask,true_mask)
    # print("Best model IoU: ",best_iou)


    # Saving model
    model_name = 'CNN_'+datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if save_model:
        model_path = proot / "models" / model_name
        cnn_model.save(model_path)
    