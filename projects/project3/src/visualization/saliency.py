from keras import backend as K
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.scores import CategoricalScore

from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam

import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt


class GradCamModel:
    def __init__(self, new_model, layer_name):
        self.gradcam = Gradcam(model=new_model, clone=True)  # (model=new_model, model_modifier=replace2linear, clone=True)
        self.layer_name = layer_name

    # def __call__(self, x):
        # return Gradcam(self.model, self.layer_name)(x)

    # Generate heatmap with GradCAM
    def get_saliency_map(self, score, img, class_idx):
        cam = self.gradcam(score, img, penultimate_layer=-1)
        return cam


if __name__ == "__main__":
    # Load model
    print("Testing GradCam Implementation")

    model_path = ""
    new_model = tf.keras.models.load_model(model_path)
    img_idx = 0
    img = test_img[img_idx, ...]
    img = tf.reshape(img, [-1] + img.shape.as_list())

    logits = new_model(img)

    logits = new_model(test_img, training=False).numpy()

    predicted = K.cast(K.argmax(logits, axis=1), "uint8").numpy()
    class_pred = predicted[0]

    score = CategoricalScore([class_pred])

    saliency = GradCamModel(new_model=new_model, clone=True)

    saliency_map = saliency.get_saliency_map(score=score, img=img, )
        # smooth_samples=20,  # The number of calculating gradients iterations.
        # smooth_noise=0.20,)  # noise spread level.


    cmap = mpl.cm.jet
    fig, axs = plt.subplots(1,2,figsize=(15,8))
    axs[0].imshow(img.numpy().squeeze())
    axs[1].imshow(img.numpy().squeeze())
    axs[1].imshow(saliency_map.squeeze(), cmap=cmap, alpha=0.5)
    # saliency_fig_path = PROJECT_ROOT / "reports/figures/smoothgrad_saliency.png"
    # plt.savefig(saliency_fig_path)
