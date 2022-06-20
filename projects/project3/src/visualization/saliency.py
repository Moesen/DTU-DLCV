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
from projects.project3.src.visualization.make_boundary import get_boundary

from keras import backend as K
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.scores import CategoricalScore

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
    IMG_SIZE = (256,256)#,3)
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

    img_idx = 2 
    imgs, mask = next(iter(test_dataset))
    img = tf.expand_dims(imgs[img_idx,...], 0)
    mask = tf.expand_dims(mask[img_idx,...], 0)
    img_np = img.numpy().squeeze()
    img_boundary = img.numpy().squeeze()

    logits = cnn_model(img)
    predicted = K.cast(K.argmax(logits, axis=1), "uint8").numpy()
    class_pred = predicted[0]

    score = CategoricalScore([class_pred])
    saliency = Saliency(cnn_model, clone=True)

    # img = tf.reshape(img, [-1] + img.shape.as_list())
    print("Generating saliency map...")
    start = timer()

    saliency_map = saliency(score,
                            img,
                            smooth_samples=100,  # The number of calculating gradients iterations.
                            smooth_noise=0.2,
                            )  # noise spread level.

    end = timer()
    print("Time spend computing saliency map:", end - start)

    heatmap_out = saliency_map.squeeze()

    #logits = cnn_model(img, training=False)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap_out)

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
    pred_mask = heatmap>50
    pred_mask = pred_mask*1

    mask_np = mask.numpy().squeeze()
    breakpoint()
    #change color for boundary of GT mask 
    out, b_idx = get_boundary(mask_np, is_GT=True)
    img_boundary[np.logical_and(b_idx>1,b_idx<255),:] = out[np.logical_and(b_idx>1,b_idx<255),:]
    
    #change color for bounadry of prediction
    out, b_idx = get_boundary(pred_mask, is_GT=False)
    img_boundary[b_idx>1,:] = out[b_idx>1,:]



    cmap = mpl.cm.jet
    fig, axs = plt.subplots(1,5,figsize=(15,8))
    axs[0].imshow(img_np/255)
    axs[1].imshow(heatmap_out.squeeze(), cmap=cmap)
    axs[2].imshow(superimposed_img)
    axs[3].imshow(pred_mask, cmap="gray")
    axs[4].imshow( img_boundary/255 )


    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')
    axs[3].axis('off')
    axs[4].axis('off')

    saliency_fig_path = proot / "reports/figures/smoothgrad_saliency.png"
    plt.savefig(saliency_fig_path,bbox_inches='tight')
