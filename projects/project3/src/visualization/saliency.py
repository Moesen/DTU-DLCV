from keras import backend as K
from tf_keras_vis.saliency import Saliency
# from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore

from matplotlib import cm
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow import keras

from tensorflow.keras.layers import Conv2D

import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

from projects.utils import get_project3_root
import os
from tensorflow.keras.models import Model
import numpy as np

from projects.project3.src.data.dataloader import IsicDataSet



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




def get_saliency_map(model, image, class_idx):
    with tf.GradientTape() as tape:
        tape.watch(image)
        logits = model(image)
        loss = logits[:, class_idx]

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, image)

    # take maximum across channels
    gradient = tf.reduce_max(gradient, axis=-1)

    # convert to numpy
    gradient = gradient.numpy()

    # normaliz between 0 and 1
    min_val, max_val = np.min(gradient), np.max(gradient)
    smap = (gradient - min_val) / (max_val - min_val + K.epsilon())

    return smap


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        #if pred_index is None:
        #    pred_index = tf.argmax(preds[0])
        #class_channel = preds[:, pred_index]
        class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()



if __name__ == "__main__":

    # Load model
    print("Testing GradCam Implementation")
    proot = get_project3_root()
    #model_path = proot / "models" / "CNN_1" / "saved_model.pb"

    model_path = proot / "models/" / "CNN_model_20220620112147"
    cnn_model = tf.keras.models.load_model(model_path)

    #cnn_model = tf.keras.models.load_model('/home/augustsemrau/drive/M1semester/02514_DLinCV/DTU-DLCV/projects/project3/models/CNN_20220619212559.h5')
    # cnn_model = cnn_model.load_weights('/home/augustsemrau/drive/M1semester/02514_DLinCV/DTU-DLCV/projects/project3/models/CNN_weights_20220619212559.h5')
    print(cnn_model.summary())

    # Load image
    IMG_SIZE = (256,256)
    #lesions_path = proot / "data/isic" / "train_allstyles/Images" / "ISIC_0000013.jpg"

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

    img_idx = 1
    imgs, mask = next(iter(test_dataset))
    img = tf.expand_dims(imgs[img_idx,...], 0)
    mask = tf.expand_dims(mask[img_idx,...], 0)

    from keras import backend as K
    from tf_keras_vis.saliency import Saliency
    from tf_keras_vis.utils.scores import CategoricalScore

    logits = cnn_model(img)
    predicted = K.cast(K.argmax(logits, axis=1), "uint8").numpy()
    class_pred = predicted[0]

    score = CategoricalScore([class_pred])
    saliency = Saliency(cnn_model, clone=True)

    saliency_map = saliency(
        score,
        img,
        smooth_samples=100,  # The number of calculating gradients iterations.
        smooth_noise=0.5,
    )  # noise spread level.

    heatmap = saliency_map.squeeze()

    img_np = img.numpy().squeeze()

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_np.shape[1], img_np.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.5 + img_np
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    #predict a mask 
    pred_mask = heatmap>150
    pred_mask = pred_mask*1

    

    cmap = mpl.cm.jet
    fig, axs = plt.subplots(1,4,figsize=(15,8))
    axs[0].imshow(img_np)
    axs[1].imshow(jet_heatmap)
    axs[2].imshow(superimposed_img)
    axs[3].imshow(pred_mask)

    saliency_fig_path = proot / "reports/figures/gradcam_saliency.png"
    plt.savefig(saliency_fig_path)

    breakpoint()



