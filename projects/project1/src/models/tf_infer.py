import tensorflow as tf
from src.utils import get_project_root
from src.data.dataloader import load_dataset
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from scipy.ndimage.filters import gaussian_filter

PROJECT_ROOT = get_project_root()
model_name = 'hotdog_conv_20220604214318' #'hotdog_conv_20220604190940'
model_path = PROJECT_ROOT / "models" / model_name

new_model = tf.keras.models.load_model(model_path)

# Check its architecture
new_model.summary()

batch_size=4
img_size = (64,64)

test_data = load_dataset(
        train=False,
        normalize=True,
        batch_size=batch_size,
        tune_for_perfomance=False,
        image_size=img_size,
    )

(test_img, y) = next(iter(test_data))

logits = new_model(test_img, training=False).numpy()

predicted = K.cast(K.argmax(logits, axis=1), "uint8").numpy()  # one dimensional
y_targets = tf.cast(y, tf.uint8).numpy()

labels = test_data._input_dataset.class_names

probs = tf.nn.softmax(logits,axis=1)

for n, (img,pred,prob) in enumerate(zip(test_img.numpy(),predicted,probs)):
    plt.subplot(2,2,n+1)
    plt.imshow(img)
    pred_prob = np.max(prob)
    pred_label = labels[int(pred)]
    plt.title(f"Pred: {pred_label}, p={pred_prob:.2f}")
    #plt.show()

plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)

fig_path = PROJECT_ROOT / "reports/figures/test.png" 
plt.savefig(fig_path)



# Vanilla Saliency map
img_idx = 0
img = test_img[img_idx,...]
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

#raw saliency map
plt.subplot(1,2,1)
plt.imshow(img.numpy().squeeze())
pred_prob = np.max(probs[img_idx,...])
pred_label = labels[int(predicted[img_idx,...])]
plt.title(f"Pred: {pred_label}, p={pred_prob:.2f}")
 
plt.subplot(1,2,2)
plt.imshow(smap.squeeze(),cmap='jet')
plt.title(f"Saliency of P(img=class {class_idx})")

saliency_fig_path = PROJECT_ROOT / "reports/figures/raw_saliency.png" 
plt.savefig(saliency_fig_path)



#saliency map on top 
plt.subplot(1,2,1)
plt.imshow(img.numpy().squeeze())
pred_prob = np.max(probs[img_idx,...])
pred_label = labels[int(predicted[img_idx,...])]
plt.title(f"Pred: {pred_label}, p={pred_prob:.2f}")
plt.subplot(1,2,2)
plt.imshow(img.numpy().squeeze())

blurred = gaussian_filter(smap, sigma=3)
#blurred[blurred<0.4] = np.NaN ### use 0.5 for non blurred

cmap = mpl.cm.jet
cmap.set_bad("white")
plt.imshow(blurred.squeeze(),cmap=cmap,alpha=0.5)
saliency_fig_path = PROJECT_ROOT / "reports/figures/raw_saliency_smooth.png" 
plt.savefig(saliency_fig_path)






from keras import backend as K
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.scores import CategoricalScore
#from tf_keras_vis.gradcam import Gradcam


logits = new_model(img)
class_pred = predicted[0]

score = CategoricalScore([class_pred])

saliency = Saliency(new_model, clone=True)

saliency_map = saliency(score,
                        img,
                        smooth_samples=20, # The number of calculating gradients iterations.
                        smooth_noise=0.20) # noise spread level.
plt.subplot(1,2,1)
plt.imshow(img.numpy().squeeze())
plt.subplot(1,2,2)
plt.imshow(img.numpy().squeeze())
plt.imshow(saliency_map.squeeze(),cmap=cmap,alpha=0.5)
saliency_fig_path = PROJECT_ROOT / "reports/figures/smoothgrad_saliency.png" 
plt.savefig(saliency_fig_path)