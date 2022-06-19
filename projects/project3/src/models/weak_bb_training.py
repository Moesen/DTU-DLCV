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


def get_model_memory_usage(batch_size, model):
    import numpy as np
    try:
        from keras import backend as K
    except:
        from tensorflow.keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes


if __name__ == '__main__':
    
    BATCH_SIZE = 16
    IMG_SIZE = (256,256) #(256,256,3)
    GF = 32
    seed = 69

    proot = get_project3_root()
    data_root = proot / "data/isic/train_allstyles"
    image_path = data_root / "Images"
    mask_path = data_root / "Segmentations"


    ### Setup first model
    save_model = True
    num_epochs = 100
    sample_img_interval = 20
    num_weak_runs = 10

    ### Run U-Net training multiple times, after each run saving a new set of masks
    for run_ind in range(num_weak_runs):
        
        if run_ind == 0:
            new_mask_path = data_root / "Segmentations"
        else:
            new_mask_path = data_root / "Weak_BB" / ("run_" + str(run_ind-1))

        dataset_loader = IsicDataSet(
            image_folder=image_path,
            mask_folder=new_mask_path,
            image_size=IMG_SIZE,
            image_channels=3,
            mask_channels=1,
            image_file_extension="jpg",
            mask_file_extension="png",
            do_normalize=True,
            validation_percentage=.2,
            segmentation_type="1",
            output_image_path=False,
            validation_segmentation_type="2",
            seed=seed,
        )
        train_dataset_weak, val_dataset_strong = dataset_loader.get_dataset(batch_size=BATCH_SIZE, shuffle=True)

        ## Init U-Net and train
        unet = Pix2Pix_Unet(loss_f=focal_loss(),  #tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                            train_dataset=train_dataset_weak,
                            test_data=[],
                            img_size=(*IMG_SIZE, 3),
                            gf=GF,
                            num_conv=3,
                            depth=5,
                            batchnorm=True,
                            )

        unet.unet.summary()
        print("MEMORY USAGE: ",get_model_memory_usage(BATCH_SIZE, unet.unet))

        dataset_loader._output_image_path = True

        model_history = unet.unet.fit(train_dataset_weak, epochs=num_epochs,validation_data=val_dataset_strong)
        
        train_dataset_weak, val_dataset_strong = dataset_loader.get_dataset(batch_size=BATCH_SIZE, shuffle=True)
        #unet.train(epochs=num_epochs,sample_interval_epoch=sample_img_interval )

        # Compute IoU for the final model
        total_iou = []
        for (x_batch_val, true_mask, img_path) in val_dataset_strong:
            val_logits = unet.unet(x_batch_val, training=False)
            val_probs = tf.keras.activations.sigmoid(val_logits)
            pred_mask = tf.math.round(val_probs)

            compute_IoU = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])
            batch_iou = compute_IoU(pred_mask, true_mask)
            total_iou.append( batch_iou )

        print("IoU for entire validation set: ",np.array(total_iou).mean())

        
        # Saving model
        model_name = f'weak_unet_{run_ind}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'

        if save_model:
            model_path = proot / "models" / model_name
            unet.unet.save(model_path)
        
        ### Save predicted masks in new folder
        new_mask_path = data_root / "Weak_BB" / ("run_" + str(run_ind))
        if not new_mask_path.exists():
            new_mask_path.mkdir()

        ### Now we save the predicted label segmentations of training set
        for (x_batch_train, _, img_path) in train_dataset_weak:
            train_logits = unet.unet(x_batch_train, training=False)
            train_probs = tf.keras.activations.sigmoid(train_logits)
            train_pred_mask = tf.math.round(train_probs)

            pred_mask = tf.cast(train_pred_mask, tf.uint8)
            for i, img in enumerate(x_batch_train):
                pred_mask_i = pred_mask[i]
                pred_mask_i = tf.image.encode_png(pred_mask_i)
                pred_mask_i = pred_mask_i.numpy()

                img_path_i = img_path[i].numpy().decode("utf-8")
                img_path_i = "ISIC" + img_path_i.split("ISIC")[1]
                img_path_i = img_path_i[:-4]
                pred_mask_file = new_mask_path / (img_path_i + "_seg_2_expert_f" + ".png")
                with open(pred_mask_file, "wb") as f:
                    f.write(pred_mask_i)
    
    