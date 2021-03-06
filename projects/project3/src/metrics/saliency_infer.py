from keras import backend as K
from tf_keras_vis.saliency import Saliency
# from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore

from matplotlib import cm
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow import keras


import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

from projects.utils import get_project3_root
import os
from tensorflow.keras.models import Model
import numpy as np

from projects.project3.src.data.dataloader import IsicDataSet
from projects.project3.src.data.dataloader_CNN import IsicDataSet_cnn

from tqdm import tqdm
from timeit import default_timer as timer


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




if __name__ == "__main__":

    # Load model
    print("Testing GradCam Implementation")
    proot = get_project3_root()
    model_path = proot / "models/" / "CNN_model_20220620112147"
    cnn_model = tf.keras.models.load_model(model_path)

    print(cnn_model.summary())

    # Load image
    IMG_SIZE = (256,256)
    #lesions_path = proot / "data/isic" / "train_allstyles/Images" / "ISIC_0000013.jpg"


    ## compute CNN acucracy 
    data_root = proot / "data/isic"
    lesions_path = data_root / "train_allstyles/Images"
    background_path = data_root / "background"

    dataset_loader = IsicDataSet_cnn(
        lesions_folder=lesions_path,
        background_folder=background_path,
        image_size=IMG_SIZE,
        image_channels=3,
        # mask_channels=1,
        image_file_extension="jpg",
        # mask_file_extension="png",
        do_normalize=False,
        validation_percentage=0.2,
        seed=69,
        # segmentation_type="0",
    )

    _, val_dataset = dataset_loader.get_dataset(batch_size=16, shuffle=True)

    n_correct = 0
    n_datapoints = 0

    for (x_batch_val, y_label) in val_dataset:
        val_logits = cnn_model(x_batch_val, training=False)
        predicted = K.cast(K.argmax(val_logits, axis=1), "uint8").numpy()

        true_labels = y_label.numpy().squeeze()

        n_correct += (true_labels==predicted).sum()
        n_datapoints += len(true_labels)

    print("Accuracy: ", n_correct/n_datapoints)

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
        do_normalize=False,
        validation_percentage=.1,
        seed=69,
    )

    test_dataset, _ = dataset_loader.get_dataset(batch_size=16, shuffle=False)


print("Computing final metrics...")

total_iou = []
total_p_diff = []


for n,(x_batch_val, true_mask) in enumerate(test_dataset):
    print("New batch...",n)
    print("out of ",len(test_dataset))
    for (val_img, val_GT_mask) in zip(x_batch_val, true_mask):
        img = tf.expand_dims(val_img, 0)
        val_logits = cnn_model(img, training=False)
        #val_probs = tf.keras.activations.sigmoid(val_logits)
        #pred_mask = tf.math.round(val_probs)

        predicted = K.cast(K.argmax(val_logits, axis=1), "uint8").numpy()
        class_pred = predicted[0]

        score = CategoricalScore([class_pred])
        saliency = Saliency(cnn_model, clone=True)

        print("Generating saliency map...")
        start = timer()

        saliency_map = saliency(
            score,
            img,
            smooth_samples=50,  # The number of calculating gradients iterations.
            smooth_noise=0.2,
        )  # noise spread level.

        end = timer()
        print("Time spend computing saliency map:", end - start)

        heatmap = saliency_map.squeeze()

        img_np = img.numpy().squeeze()

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        #predict a mask 
        pred_mask = heatmap>50
        pred_mask = pred_mask*1
        pred_mask = tf.convert_to_tensor(pred_mask, dtype=tf.int64) 
        val_GT_mask = tf.convert_to_tensor(val_GT_mask.numpy() / 255., dtype=tf.int64) 

        compute_IoU = tf.keras.metrics.BinaryIoU() #tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])
        batch_iou = compute_IoU(pred_mask, val_GT_mask)
        total_iou.append( batch_iou )

        n_seg_pixels_mask = int(val_GT_mask.numpy().sum())
        n_seg_pixels_pred = int(pred_mask.numpy().sum())

        p_diff = ((n_seg_pixels_pred - n_seg_pixels_mask) / n_seg_pixels_mask)*100
        total_p_diff.append( p_diff )

print("IoU for entire test set: ",np.array(total_iou).mean())
print("Pixel diff entire test set in %: ",np.array(total_p_diff).mean())

