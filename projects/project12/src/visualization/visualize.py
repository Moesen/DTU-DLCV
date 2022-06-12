from ctypes.wintypes import PLARGE_INTEGER

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.image as img

import json
import numpy as np
import tensorflow as tf
from keras import backend as K

from projects.project12.src.data.dataloader import load_dataset_rcnn
from projects.project12.src.models.post_processing import NMS
from projects.utils import get_project12_root


PROJECT_ROOT = get_project12_root()
model_name = "trash_conv_20220611230431" #"trash_conv_20220611135130" #'hotdog_conv_20220604214318' #'hotdog_conv_20220604190940'
model_path = PROJECT_ROOT / "models" / model_name
#model_path = "/Users/simonyamazaki/Documents/2_M/DTU-DLCV/projects/project1/models/hotdog_conv_20220604214318"

new_model = tf.keras.models.load_model(model_path)

# Check its architecture
new_model.summary()

batch_size = 100
img_size = (128,128)


val_data = load_dataset_rcnn(
    split="validation",
    normalize=False,
    use_data_augmentation=False,
    batch_size=batch_size,
    tune_for_perfomance=False,
    image_size=img_size,
    validation_mode = "object"
)

#labels = val_data._input_dataset.class_names
path = PROJECT_ROOT / "data/data_wastedetection"
#with open(path / "annotations.json", "r") as f:
#    dataset_json = json.loads(f.read())
#catid2supercat = {i["id"]: i["supercategory"] for i in dataset_json["categories"]}
#all_super_cat = list(set(i["supercategory"] for i in dataset_json["categories"]))
#all_super_cat.append("Background")
#labels = all_super_cat
#b, classes = np.unique(labels, return_inverse=True)
#classes = np.arrange()

with open(path / f"cat2id.json", "r") as f:
        cat2id_json = json.loads(f.read())

labels = list(cat2id_json.keys())
classes = np.arange(0,len(labels))


BB_all_predicted = []
bb_class = []
bb_confidence = []

test_img, tensor_labels, img_path0, BB = list(iter(val_data))[11] #31=cans, 41=cans in fence, 51=redbull, 61=can with leaves, 71=wash bottle, not trash
img_path0 = img_path0[0].numpy().decode("UTF-8")

#### LOOP ####
print("Running all batches to find batches for image of interest")
found_image = False
for n,(bb_img, tensor_labels, img_path, BB) in enumerate(val_data):
    img_path = img_path[0].numpy().decode("UTF-8")

    if img_path0 == img_path:
        #compute classification predictions on the bb cropped images with bounding boxes = BB
        logits = new_model(bb_img, training=False)
        probs = tf.nn.softmax(logits,axis=1)
        predicted = K.cast(K.argmax(logits, axis=1), "uint8")

        BB_all_predicted.append(BB)
        bb_confidence.append( probs.numpy().max(1) )
        bb_class.append( predicted.numpy() )

        found_image = True

        print('Processing image batch: %i' % n)
    else:
        print('Did not find image in batch: %i' % n)

    if found_image and (img_path0 != img_path):
        break

#reformat from shape (n_batches, batch_size, 4) -> (n_batches*batch_size, 4) = (2000,4)
BB_all_predicted = np.array(BB_all_predicted)
BB_all_predicted = BB_all_predicted.reshape((-1,BB_all_predicted.shape[-1]))

#reformat from shape (n_batches, batch_size, 1) -> (n_batches*batch_size, 1) = (2000,1)
bb_class = np.array(bb_class)
bb_class = bb_class.flatten()[:,np.newaxis]

#reformat from shape (n_batches, batch_size, 1) -> (n_batches*batch_size, 1) = (2000,1)
bb_confidence = np.array(bb_confidence)
bb_confidence = bb_confidence.flatten()[:,np.newaxis]

#cast to tensorflow tensors
BB_all_predicted = tf.convert_to_tensor(np.array(BB_all_predicted),dtype=tf.float32)
bb_confidence = tf.convert_to_tensor(np.array(bb_confidence),dtype=tf.float32)
bb_class = tf.convert_to_tensor(np.array(bb_class),dtype=tf.float32)

#remove background objects 
idx = K.cast(np.where(bb_class.numpy()!=28)[0], tf.int32)
bb_class = tf.gather(bb_class, idx)
bb_confidence = tf.gather(bb_confidence, idx)
BB_all_predicted = tf.gather(BB_all_predicted, idx)


# NMS post processing
print("Running NMS post-processing")
all_selected_boxes, all_selected_probs, all_selected_preds = NMS(BB_all_predicted, bb_class, bb_confidence, classes[:-1], iout = 0.5, st = 0.4, max_out = 10)
#all_selected_preds = [labels[int(i)] for i in all_selected_preds.numpy().squeeze().tolist()]


base_img_path = PROJECT_ROOT / "data/data_wastedetection" / img_path0
base_img = img.imread(base_img_path)


fig, axs = plt.subplots(1,2, figsize=(15,8))

cmap = mpl.cm.get_cmap('Spectral')

BBs = [BB_all_predicted,all_selected_boxes]
probss = [bb_confidence,all_selected_probs]
predss = [bb_class,all_selected_preds]

tits = ["Before post-processing", "After post-processing"]

for m in range(2):
    axs[m].imshow(base_img)

    BB_plot = BBs[m]
    prob_plot = probss[m]
    pred_plot = predss[m]

    for n, (bb,pred,prob) in enumerate(zip(BB_plot.numpy(),pred_plot.numpy().squeeze(),prob_plot.numpy())):
        
        #rgba = cmap(n/prob_plot.shape[0])
        rgba = cmap(pred/len(classes[:-1]))

        rect = patches.Rectangle((bb[0],bb[1]), bb[2], bb[3], linewidth=1.5,
                            edgecolor=rgba, facecolor="none")
        axs[m].add_patch(rect)

        pred_prob = np.max(prob)
        pred_label = labels[int(pred)]
        object_text = f"{pred_label}, p={pred_prob:.2f}"

        axs[m].text(bb[0],bb[1], object_text, color='red', 
            bbox=dict(facecolor='white', edgecolor='black'),fontsize=10)

        axs[m].set_title(tits[m],fontsize=15,x=0.5, y=1.1)


fig_path = PROJECT_ROOT.parent / "project12/reports/figures/Objects_detected.png" 
plt.savefig(fig_path)







########### OTHER 
fig, axs = plt.subplots(2,2, figsize=(15,15))
cmap = mpl.cm.get_cmap('Spectral')


BB_all_predicted = []
bb_class = []
bb_confidence = []

good_b = [11, 41, 71, 31]
tits = ["Something 1", "Something 2", "Something 3", "Something 4"]


for gg, (ii, ax) in enumerate(zip(good_b,axs.ravel())):
    _, _, img_path0, BB = list(iter(val_data))[ii] #31=cans, 41=cans in fence, 51=redbull, 61=can with leaves, 71=wash bottle not trash, 
    img_path0 = img_path0[0].numpy().decode("UTF-8")

    print(gg)

    #### LOOP ####
    print("Running all batches to find batches for image of interest")
    found_image = False
    for n,(bb_img, tensor_labels, img_path, BB) in enumerate(val_data):
        img_path = img_path[0].numpy().decode("UTF-8")

        if img_path0 == img_path:
            #compute classification predictions on the bb cropped images with bounding boxes = BB
            logits = new_model(bb_img, training=False)
            probs = tf.nn.softmax(logits,axis=1)
            predicted = K.cast(K.argmax(logits, axis=1), "uint8")
            BB_all_predicted.append(BB)
            bb_confidence.append( probs.numpy().max(1) )
            bb_class.append( predicted.numpy() )

            found_image = True

            print('Processing image batch: %i' % n)
        else:
            print('Did not find image in batch: %i' % n)

        if found_image and (img_path0 != img_path):
            break

    #reformat from shape (n_batches, batch_size, 4) -> (n_batches*batch_size, 4) = (2000,4)
    BB_all_predicted = np.array(BB_all_predicted)
    BB_all_predicted = BB_all_predicted.reshape((-1,BB_all_predicted.shape[-1]))

    #reformat from shape (n_batches, batch_size, 1) -> (n_batches*batch_size, 1) = (2000,1)
    bb_class = np.array(bb_class)
    bb_class = bb_class.flatten()[:,np.newaxis]

    #reformat from shape (n_batches, batch_size, 1) -> (n_batches*batch_size, 1) = (2000,1)
    bb_confidence = np.array(bb_confidence)
    bb_confidence = bb_confidence.flatten()[:,np.newaxis]

    #cast to tensorflow tensors
    BB_all_predicted = tf.convert_to_tensor(np.array(BB_all_predicted),dtype=tf.float32)
    bb_confidence = tf.convert_to_tensor(np.array(bb_confidence),dtype=tf.float32)
    bb_class = tf.convert_to_tensor(np.array(bb_class),dtype=tf.float32)

    #remove background objects 
    idx = K.cast(np.where(bb_class.numpy()!=28)[0], tf.int32)
    bb_class = tf.gather(bb_class, idx)
    bb_confidence = tf.gather(bb_confidence, idx)
    BB_all_predicted = tf.gather(BB_all_predicted, idx)

    # NMS post processing
    print("Running NMS post-processing")
    all_selected_boxes, all_selected_probs, all_selected_preds = NMS(BB_all_predicted, bb_class, bb_confidence, classes[:-1], iout = 0.5, st = 0.4, max_out = 10)

    BB_all_predicted = []
    bb_class = []
    bb_confidence = []

    base_img_path = PROJECT_ROOT / "data/data_wastedetection" / img_path0
    base_img = img.imread(base_img_path)

    BB_plot = all_selected_boxes
    prob_plot = all_selected_probs
    pred_plot = all_selected_preds
    
    ax.set_title(tits[gg],fontsize=15,x=0.5, y=1.1)
    ax.imshow(base_img)

    for m, (bb,pred,prob) in enumerate(zip(BB_plot.numpy(),pred_plot.numpy().squeeze(),prob_plot.numpy())):
        
        rgba = cmap(pred/len(classes[:-1]))

        rect = patches.Rectangle((bb[0],bb[1]), bb[2], bb[3], linewidth=1.5,
                            edgecolor=rgba, facecolor="none")
        ax.add_patch(rect)

        pred_prob = np.max(prob)
        pred_label = labels[int(pred)]
        object_text = f"{pred_label}, p={pred_prob:.2f}"

        ax.text(bb[0],bb[1], object_text, color='red', 
            bbox=dict(facecolor='white', edgecolor='black'),fontsize=10)



fig_path = PROJECT_ROOT.parent / "project12/reports/figures/Objects_detected2.png" 
plt.savefig(fig_path)