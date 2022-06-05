import tensorflow as tf
from src.utils import get_project_root
from src.data.dataloader import load_dataset
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

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



# Saliency map
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

#original image
plt.subplot(1,2,1)
plt.imshow(img.numpy().squeeze())
pred_prob = np.max(probs[img_idx,...])
pred_label = labels[int(predicted[img_idx,...])]
plt.title(f"Pred: {pred_label}, p={pred_prob:.2f}")

#saliency map 
plt.subplot(1,2,2)
plt.imshow(smap.squeeze(),cmap='jet')
plt.title(f"Saliency of P(img=class {class_idx})")

saliency_fig_path = PROJECT_ROOT / "reports/figures/test_saliency.png" 
plt.savefig(saliency_fig_path)

#two images on top
plt.subplot(1,2,1)
plt.imshow(img.numpy().squeeze())
pred_prob = np.max(probs[img_idx,...])
pred_label = labels[int(predicted[img_idx,...])]
plt.title(f"Pred: {pred_label}, p={pred_prob:.2f}")
plt.subplot(1,2,2)
plt.imshow(img.numpy().squeeze())
smap[smap<0.4] = np.NaN
cmap = mpl.cm.jet
cmap.set_bad("white")
plt.imshow(smap.squeeze(),cmap=cmap,alpha=0.5)
saliency_fig_path = PROJECT_ROOT / "reports/figures/test_saliency2.png" 
plt.savefig(saliency_fig_path)

"""images = tf.Variable(img, dtype=float)

y_pred = model.predict(img)

with tf.GradientTape() as tape:
    pred = model(images, training=False)
    class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
    loss = pred[0][class_idxs_sorted[0]]
    
grads = tape.gradient(loss, images)

dgrad_abs = tf.math.abs(grads)

dgrad_max_ = np.max(dgrad_abs, axis=3)[0]

arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)

fig, axes = plt.subplots(1,2,figsize=(14,5))
axes[0].imshow(_img)
i = axes[1].imshow(grad_eval,cmap="jet",alpha=0.8)
fig.colorbar(i)"""