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


"""class GradCamModel:
    def __init__(self, new_model, layer_name):
        self.gradcam = Gradcam(model=new_model, clone=True)  # (model=new_model, model_modifier=replace2linear, clone=True)
        self.layer_name = layer_name

    # def __call__(self, x):
        # return Gradcam(self.model, self.layer_name)(x)

    # Generate heatmap with GradCAM
    def get_saliency_map(self, score, img, class_idx):
        cam = self.gradcam(score, img, penultimate_layer=-1)
        return cam"""


"""def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array"""


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
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
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

    model_path = proot / "models/" / "CNN_model_20220620105359"
    cnn_model = tf.keras.models.load_model(model_path)

    #cnn_model = tf.keras.models.load_model('/home/augustsemrau/drive/M1semester/02514_DLinCV/DTU-DLCV/projects/project3/models/CNN_20220619212559.h5')
    # cnn_model = cnn_model.load_weights('/home/augustsemrau/drive/M1semester/02514_DLinCV/DTU-DLCV/projects/project3/models/CNN_weights_20220619212559.h5')
    print(cnn_model.summary())

    # Load image
    IMG_SIZE = (256,256)#,3)
    lesions_path = proot / "data/isic" / "train_allstyles/Images" / "ISIC_0000013.jpg"

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

    imgs, mask = next(iter(test_dataset))
    img = tf.expand_dims(imgs[0,...], 0)
    #img = get_img_array(img_path=lesions_path, size=IMG_SIZE)

    # img = tf.reshape(img, [-1] + img.shape.as_list())

    #logits = cnn_model(img)

    #logits = cnn_model(img, training=False)

    #predicted = K.cast(K.argmax(logits, axis=1), "uint8").numpy()
    #class_pred = predicted[0]

    #score = CategoricalScore([class_pred])

    # saliency = GradCamModel(new_model=cnn_model, clone=True)

    # saliency_map = saliency.get_saliency_map(score=score, img=img, )
        # smooth_samples=20,  # The number of calculating gradients iterations.
        # smooth_noise=0.20,)  # noise spread level.
    # Remove last layer's softmax
    
    cnn_model.layers[-1].activation = None
    lcl = "global_average_pooling2d"

    heatmap = make_gradcam_heatmap(img_array=img.numpy(), model=cnn_model, last_conv_layer_name=lcl, pred_index=None)


    cmap = mpl.cm.jet
    fig, axs = plt.subplots(1,2,figsize=(15,8))
    axs[0].imshow(img.numpy().squeeze())
    axs[1].imshow(img.numpy().squeeze())
    axs[1].imshow(heatmap.squeeze(), cmap=cmap, alpha=0.5)
    saliency_fig_path = proot / "reports/figures/gradcam_saliency.png"
    plt.savefig(saliency_fig_path)
