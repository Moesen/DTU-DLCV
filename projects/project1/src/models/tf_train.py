from __future__ import annotations

import os
import ssl
import time

import tensorflow as tf
#from tensorflow import keras 
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from src.data.dataloader import load_dataset
from src.models.optuna_model import ConvNet
from tensorflow.python.client import device_lib
from tqdm import tqdm
import numpy as np


ssl._create_default_https_context = ssl._create_unverified_context

# built tensorflow with GPU

print("TENSORFLOW BUILT WITH CUDA: ", tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))


print("TENSORFLOW VISIBLE DEVIES: ", device_lib.list_local_devices())

method = "GPU"

if method == "GPU":

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def recall(y_true, y_pred):
    y_true = tf.cast(y_true,tf.float32)
    y_pred = tf.cast(y_pred,tf.float32)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + K.epsilon())
    return recall_keras

out_dict = {'train_acc': [],
            'train_recall': [],
            'train_loss': []}


if __name__ == "__main__":
    img_size = (32, 32)
    batch_size = 64

    train_dataset = load_dataset(
        train=True,
        normalize=True,
        batch_size=batch_size,
        tune_for_perfomance=False,
        image_size=img_size,
    )

    test_data = load_dataset(
        train=False,
        normalize=True,
        batch_size=batch_size,
        tune_for_perfomance=False,
        image_size=img_size,
    )

    net = ConvNet(32, 3, 2, (*img_size, 3), BN=True, DO=True)
    model = net.build_model()
    model.summary()

    # Instantiate an optimizer to train the model.
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    # Instantiate a loss function.
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Prepare the metrics.
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    epochs = 50
    # for epoch in range(epochs):
    for epoch in tqdm(range(epochs), unit="epoch"):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        train_acc = []
        train_loss = []
        train_recall = []
        train_n_correct_epoch = 0
        dataset_size = 0

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in tqdm(
            enumerate(train_dataset), total=len(train_dataset)
        ):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update training metric.
            # accuracy with tensorflow metric object
            train_acc_metric.update_state(y_batch_train, logits)

            #custom accuracy computation with keras backend 
            predicted = K.cast(K.argmax(logits,axis=1),"uint8") #one dimensional

            #y_targets = K.squeeze(y_batch_train, axis=1) #y_batch_train is 2 dimensional
            y_targets = tf.cast(y_batch_train,tf.uint8)
            train_n_correct_epoch += K.sum(tf.cast(y_targets==predicted, tf.float32))
            dataset_size += len(y_batch_train) 
            #training loss
            train_loss.append( loss_value.numpy() )
            
            #custom computation of recall with keras backend
            train_recall.append( recall(y_targets, predicted).numpy() )

            # Log every 200 batches.
            if step % 200 == 0:
                print("Training loss (for one batch) at batch step %d: %.4f" % (step, float(loss_value)))

        out_dict['train_acc'].append(train_n_correct_epoch / dataset_size )
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['train_recall'].append(np.mean(train_recall))


        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))
        print("Training acc (numpy) over epoch: %.4f" % (float(train_n_correct_epoch / dataset_size),))
        print("Training loss over epoch: %.4f" % (float(np.mean(train_loss)),))
        print("Training recall over epoch: %.4f" % (float(np.mean(train_recall)),))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in test_data:
            val_logits = model(x_batch_val, training=False)
            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))
