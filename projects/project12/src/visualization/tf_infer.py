import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
import matplotlib.patches as patches
from src.data.dataloader import load_dataset
from src.utils import get_project_root

PROJECT_ROOT = get_project_root()
model_name = 'hotdog_conv_20220604214318' #'hotdog_conv_20220604190940'
model_path = PROJECT_ROOT / "models" / model_name

new_model = tf.keras.models.load_model(model_path)

# Check its architecture
new_model.summary()

batch_size=1
img_size = (64,64)

test_data = load_dataset(
        train=False,
        normalize=True,
        batch_size=batch_size,
        tune_for_perfomance=False,
        image_size=img_size,
    )

(test_img, y) = next(iter(test_data))

#proposal BB cropped images 
#yield: prop_imgs, BB
# with the first dimension as the number of proposals 

logits = new_model(prop_imgs, training=False)
probs = tf.nn.softmax(logits,axis=1).numpy()
predicted = K.cast(K.argmax(logits, axis=1), "uint8").numpy()

labels = test_data._input_dataset.class_names


plt.imshow(test_img.numpy().squeeze())

for bb,pred,prob in zip(BB.numpy(),predicted.numpy(),probs.numpy()):
    
    rect = patches.Rectangle((bb[0],bb[1]), bb[2], bb[3], linewidth=1,
                         edgecolor='g', facecolor="none")
    plt.add_patch(rect)

    pred_prob = np.max(prob)
    pred_label = labels[int(pred)]
    object_text = f"{pred_label}, p={pred_prob:.2f}"

    plt.text(bb[0],bb[1], object_text, color='red', 
        bbox=dict(facecolor='None', edgecolor='red'))



fig_path = PROJECT_ROOT / "reports/figures/Objects_detected.png" 
plt.savefig(fig_path)

