import ssl
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import regularizers
from projects.project12.src.data.dataloader import load_dataset
from projects.utils import get_project12_root
from tensorflow import keras
from tqdm import tqdm

ssl._create_default_https_context = ssl._create_unverified_context

img_size_loader = (128,128)
img_size = (128,128,3)
batch_size = 64

# REMEBER TO ADD ONE IF THE BACKGROUND IS NOT INCLUDED 
num_classes = 2

loss_weight = 1


base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=img_size,
    pooling=None,
    classes=None,
    classifier_activation=None,
    include_preprocessing=True
)

train_dataset = load_dataset(
    train=True,
    normalize=False,
    shuffle = True,
    batch_size=batch_size,
    tune_for_perfomance=False,
    use_data_augmentation=True,
    image_size=img_size_loader,
)
test_data = load_dataset(
    train=False,
    normalize=False,
    batch_size=batch_size,
    tune_for_perfomance=False,
    use_data_augmentation=False,
    image_size=img_size_loader,
)


#batch_imgs,labels = next(iter(train_dataset))
#feature_batch = base_model(batch_imgs)

base_model.trainable = False

#inputs to model
inputs = tf.keras.Input(shape=img_size)

#efficientNET feature extractor 
feature_maps = base_model(inputs) #model output is feature maps 
feature_vec = tf.keras.layers.GlobalAveragePooling2D()(feature_maps) 

#classification head
c = tf.keras.layers.Dense(200)(feature_vec) ## 
logits = tf.keras.layers.Dense(num_classes)(c)
class_pred = K.argmax(logits,axis=1)

#regression head
r = tf.keras.layers.Dense(200)(feature_vec) ## 
r = tf.keras.layers.Dense(4)(r)

# combine outputs 
output = K.concatenate((class_pred, r), axis=-1)

# define keras model
model = tf.keras.Model(inputs, output)

model.summary()



#classification loss
loss_c_fun = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

#regression loss
mse_fun = tf.keras.losses.MeanSquaredError()

# Instantiate an optimizer to train the model.
optimizer = keras.optimizers.Adam(learning_rate=1e-3)


out_dict = defaultdict(list)


epochs = 5
# for epoch in range(epochs):
for epoch in tqdm(range(epochs), unit="epoch"):
    print("\nStart of epoch %d" % (epoch,))

    train_loss = []
    val_loss = []

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train, bb_batch_train) in tqdm(
        enumerate(train_dataset), total=len(train_dataset)
    ):
        with tf.GradientTape() as tape:
            output = model(x_batch_train, training=True)
            
            y_class_pred = output[0]
            y_bb = output[1:]

            loss_c = loss_c_fun(y_batch_train, y_class_pred)
            loss_r = mse_fun(y_bb, bb_batch_train)
            loss_combined = loss_r + loss_weight*loss_c

            grads = tape.gradient(loss_combined, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # training loss
        train_loss.append(loss_combined.numpy())

    out_dict["train_loss"].append(np.mean(train_loss))

    print("Training loss over epoch: %.4f" % (float(np.mean(train_loss)),))

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val, bb_batch_val in test_data:
        val_output = model(x_batch_val, training=False)

        y_class_pred = output[0]
        y_bb = output[1:]

        loss_c = loss_c_fun(y_batch_train, y_class_pred)
        loss_r = mse_fun(y_bb, bb_batch_train)
        loss_combined = loss_r + loss_weight*loss_c

        # validation loss
        val_loss.append(loss_combined.numpy())

    out_dict["val_loss"].append(np.mean(val_loss))
    print("Training loss over epoch: %.4f" % (float(np.mean(val_loss)),))


"""#save last model 
if save_model:
    PROJECT_ROOT = get_project_root()
    model_name = 'hotdog_conv_'+datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    model_path = PROJECT_ROOT / "models" / model_name
    model.save(model_path)"""


