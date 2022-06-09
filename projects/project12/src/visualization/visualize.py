from ctypes.wintypes import PLARGE_INTEGER
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
#model_path = PROJECT_ROOT / "models" / model_name
model_path = "/Users/simonyamazaki/Documents/2_M/DTU-DLCV/projects/project1/models/hotdog_conv_20220604214318"

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

test_img = test_img[0,:,:,:]
test_img = tf.expand_dims(test_img, axis=0)
test_img = tf.concat([test_img,test_img], axis=0)

#proposal BB cropped images 
#yield: prop_imgs, BB
# with the first dimension as the number of proposals 

BB = np.array([[30,30,15,15], [32,32,15,15]])
BB = tf.convert_to_tensor(BB,dtype=tf.float32)


#logits = new_model(prop_imgs, training=False)
logits = new_model(test_img, training=False)
probs = tf.nn.softmax(logits,axis=1).numpy()
predicted = K.cast(K.argmax(logits, axis=1), "uint8").numpy()

#predicted = np.concatenate((predicted,predicted),axis=0)
#probs = np.concatenate((probs,probs),axis=0)

labels = test_data._input_dataset.class_names


# NMS post processing 
iou_threshold = tf.convert_to_tensor(0.5,dtype=tf.float32)
score_threshold = tf.convert_to_tensor(0.2,dtype=tf.float32)
max_output_size = tf.convert_to_tensor(10,dtype=tf.int32)
scores = tf.convert_to_tensor(probs.max(1),dtype=tf.float32)

selected_indices = tf.image.non_max_suppression(
    BB, scores=scores, max_output_size=max_output_size, iou_threshold=iou_threshold, score_threshold=score_threshold)
selected_boxes = tf.gather(BB, selected_indices)



fig, axs = plt.subplots(1,2)

cmap = mpl.cm.get_cmap('Spectral')

BBs = [BB,selected_boxes]
tits = ["before NMS","after NMS"]


for m in range(test_img.shape[0]):
    axs[m].imshow(test_img[m,:,:,:].numpy().squeeze())

    BB_plot = BBs[m]

    for n, (bb,pred,prob) in enumerate(zip(BB_plot.numpy(),predicted,probs)):
        
        rgba = cmap(n/predicted.shape[0])

        rect = patches.Rectangle((bb[0],bb[1]), bb[2], bb[3], linewidth=1,
                            edgecolor=rgba, facecolor="none")
        axs[m].add_patch(rect)

        pred_prob = np.max(prob)
        pred_label = labels[int(pred)]
        object_text = f"{pred_label}, p={pred_prob:.2f}"

        axs[m].text(bb[0],bb[1], object_text, color='red', 
            bbox=dict(facecolor='white', edgecolor='black'))

        axs[m].title.set_text(tits[m])



fig_path = PROJECT_ROOT.parent / "project12/reports/figures/Objects_detected.png" 
plt.savefig(fig_path)




### mAP 

batch_size = 2
img_size = (64,64)

test_data = load_dataset(
        train=False,
        normalize=True,
        batch_size=batch_size,
        tune_for_perfomance=False,
        image_size=img_size,
    )

img_id_now = -1 

#### LOOP ####
(bb_img, y) = next(iter(test_data))

img_id = 0
bb_class = [1,1]

if img_id_now != img_id:
    confidence = []
    GT_bb = GT_bb_batch
    

BB = np.array([[30,30,15,15], [32,32,15,15]])
BB = tf.convert_to_tensor(BB,dtype=tf.float32)

logits = new_model(bb_img, training=False)
probs = tf.nn.softmax(logits,axis=1).numpy()
predicted = K.cast(K.argmax(logits, axis=1), "uint8").numpy()


img_id_now = img_id


# NMS post processing 
iou_threshold = tf.convert_to_tensor(0.5,dtype=tf.float32)
score_threshold = tf.convert_to_tensor(0.2,dtype=tf.float32)
max_output_size = tf.convert_to_tensor(10,dtype=tf.int32)
scores = tf.convert_to_tensor(probs.max(1),dtype=tf.float32)

selected_indices = tf.image.non_max_suppression(
    BB, scores=scores, max_output_size=max_output_size, iou_threshold=iou_threshold, score_threshold=score_threshold)
selected_boxes = tf.gather(BB, selected_indices)


#for c in labels:

#tf.metrics.average_precision_at_k



