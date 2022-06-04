import tensorflow as tf
from src.utils import get_project_root
from src.data.dataloader import load_dataset
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = get_project_root()
model_name = 'hotdog_conv_20220604190940'
model_path = PROJECT_ROOT / "models" / model_name

new_model = tf.keras.models.load_model(model_path)

# Check its architecture
new_model.summary()

batch_size=4
img_size = (32,32)

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

#loss, acc = new_model.evaluate(test_image, test_labels)
