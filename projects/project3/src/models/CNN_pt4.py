import datetime
import os
import ssl
from collections import defaultdict
from glob import glob

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import regularizers
from keras.layers import Conv2D
#from projects.project3.src.data.simple_dataloader import basic_loader
from projects.project3.src.data.dataloader_CNN import IsicDataSet
from projects.project3.src.metrics.eval_metrics import *
from projects.project3.src.metrics.losses import *
from projects.utils import get_project3_root
from tensorflow.python.client import device_lib
from tqdm import tqdm

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
    #x = Conv2D(32, kernel_size=3, padding='same', strides=1)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x) ##
    x = tf.keras.layers.Dense(200,kernel_regularizer=regularizers.l2(1e-3), kernel_initializer='he_normal')(x) ##
    x = tf.keras.layers.Dropout(0.2)(x)
    predictions = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, predictions)


    """x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)"""



    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

                # metrics=['accuracy']
                )

    model.summary()

    return model
    
        




if __name__ == '__main__':

    BATCH_SIZE = 16
    IMG_SIZE = (256,256,3)

    # Example of using dataloader and extracting datasets train and test
    proot = get_project3_root()
    data_root = proot / "data/isic"
    lesions_path = data_root / "train_allstyles/Images"
    background_path = data_root / "background"

    dataset_loader = IsicDataSet(
        lesions_folder=lesions_path,
        background_folder=background_path,
        image_size=IMG_SIZE,
        image_channels=3,
        # mask_channels=1,
        image_file_extension="jpg",
        # mask_file_extension="png",
        do_normalize=False,
        # segmentation_type="0",
    )

    # train_dataset, test_dataset = dataset_loader.get_dataset(batch_size=1, shuffle=True)
    # image, image_label = next(iter(test_dataset))
    # _, [a, b] = plt.subplots(1, 2)
    # a.imshow(image[0])
    # b.imshow(mask[0])
    # b.set_title(f"max val: {tf.reduce_max(mask)}, min val: {tf.reduce_min(mask)}")


    train_dataset, val_dataset = dataset_loader.get_dataset(batch_size=BATCH_SIZE, shuffle=True)
    

    # plt.imshow(train_dataset[0][0][0])
    cnn_model = create_CNN(IMG_SIZE=IMG_SIZE)

    ##### TRAIN MODEL ##### 
    save_model = True

    num_epochs = 20
    #sample_img_interval = 20

    print("Training...")
    history = cnn_model.fit(train_dataset, epochs=num_epochs, validation_data=val_dataset)
    #loss0, accuracy0 = cnn_model.evaluate(test_data)
    
    
    # Compute acc for the final model
    acc = cnn_model.evaluate(val_dataset)
    print(f"Accuracy: {acc}")
    

    # Saving model
    model_name = 'CNN_model_'+datetime.datetime.now().strftime("%Y%m%d%H%M%S")#+'.h5'
    weights_name = 'CNN_weights_'+datetime.datetime.now().strftime("%Y%m%d%H%M%S")#+'.h5'
    if save_model:
        model_path = proot / "models" / model_name
        #weights_path = proot / "models" / weights_name
        cnn_model.save(model_path)
        # save
        #cnn_model.save_weights(weights_path)
    