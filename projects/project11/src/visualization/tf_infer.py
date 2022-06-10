from xml.sax.saxutils import prepare_input_source
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from scipy.ndimage.filters import gaussian_filter

from projects.project11.src.data.dataloader import load_dataset
from projects.utils import get_project11_root

PROJECT_ROOT = get_project11_root()
model_name = "hotdog_conv_20220604214318"  #'hotdog_conv_20220604190940'
#model_path = PROJECT_ROOT / "models" / model_name
model_path = "/Users/simonyamazaki/Documents/2_M/DTU-DLCV/projects/project1/models/hotdog_conv_20220604214318"


new_model = tf.keras.models.load_model(model_path)

# Check its architecture
new_model.summary()

batch_size = 100
img_size = (64, 64)

test_data = load_dataset(
    train=False,
    normalize=True,
    batch_size=batch_size,
    tune_for_perfomance=False,
    image_size=img_size,
)

(test_img, y) = list(iter(test_data))[2]

logits = new_model(test_img, training=False).numpy()

predicted = K.cast(K.argmax(logits, axis=1), "uint8").numpy()
y_targets = tf.cast(y, tf.uint8).numpy()
probs = tf.nn.softmax(logits, axis=1).numpy()


hotdog_certain_idx = np.argmax( probs[:,0] )
nothotdog_certain_idx = np.argmax( probs[:,1] )

confidences = np.max(probs,1)
uncertain_idx = np.argpartition(np.abs(confidences-0.5), 2)[:2]
#uncertain_idx = np.argmin( np.abs(confidences-0.5) )

idx2plot = K.cast(np.concatenate((hotdog_certain_idx, nothotdog_certain_idx, uncertain_idx), axis=None), tf.int32)

test_img_plot = tf.gather(test_img, idx2plot)

probs_plot = probs[idx2plot.numpy().tolist(),:]
predicted_plot = predicted[idx2plot.numpy().tolist()]


labels = test_data._input_dataset.class_names

fig, axs = plt.subplots(2,2, figsize=(15,15))

for (img, pred, prob, ax) in zip(test_img_plot.numpy(), predicted_plot, probs_plot,axs.ravel()):
    ax.imshow(img)
    pred_prob = np.max(prob)
    pred_label = labels[int(pred)]
    ax.title.set_text(f"Pred: {pred_label}, p={pred_prob:.2f}")

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)

fig_path = PROJECT_ROOT / "reports/figures/test.png"
plt.savefig(fig_path)



# Vanilla Saliency map
img_idx = 0
img = test_img[img_idx, ...]
img = tf.reshape(img, [-1] + img.shape.as_list())


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


class_idx = 0
smap = get_saliency_map(new_model, img, class_idx)

# raw saliency map
fig, axs = plt.subplots(1,2,figsize=(15,8))
axs[0].imshow(img.numpy().squeeze())
pred_prob = np.max(probs[img_idx, ...])
pred_label = labels[int(predicted[img_idx, ...])]
axs[0].title.set_text(f"Pred: {pred_label}, p={pred_prob:.2f}")

axs[1].imshow(smap.squeeze(), cmap="jet")
axs[1].title.set_text(f"Saliency of P(img=class {class_idx})")

saliency_fig_path = PROJECT_ROOT / "reports/figures/raw_saliency.png"
plt.savefig(saliency_fig_path)


# smoothed saliency maps 
fig, axs = plt.subplots(1,2,figsize=(15,8))
axs[0].imshow(img.numpy().squeeze())

pred_prob = np.max(probs[img_idx, ...])
pred_label = labels[int(predicted[img_idx, ...])]

axs[0].title.set_text(f"Pred: {pred_label}, p={pred_prob:.2f}")
axs[1].imshow(img.numpy().squeeze())
blurred = gaussian_filter(smap, sigma=3)
# blurred[blurred<0.4] = np.NaN ### use 0.5 for non blurred
cmap = mpl.cm.jet
cmap.set_bad("white")
axs[1].imshow(blurred.squeeze(), cmap=cmap, alpha=0.5)
saliency_fig_path = PROJECT_ROOT / "reports/figures/raw_saliency_smooth.png"
plt.savefig(saliency_fig_path)



### Smooth-grad Saliency maps 

from keras import backend as K
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.scores import CategoricalScore

logits = new_model(img)
class_pred = predicted[0]

score = CategoricalScore([class_pred])

saliency = Saliency(new_model, clone=True)

saliency_map = saliency(
    score,
    img,
    smooth_samples=20,  # The number of calculating gradients iterations.
    smooth_noise=0.20,
)  # noise spread level.

fig, axs = plt.subplots(1,2,figsize=(15,8))

axs[0].imshow(img.numpy().squeeze())

axs[1].imshow(img.numpy().squeeze())
axs[1].imshow(saliency_map.squeeze(), cmap=cmap, alpha=0.5)
saliency_fig_path = PROJECT_ROOT / "reports/figures/smoothgrad_saliency.png"
plt.savefig(saliency_fig_path)
