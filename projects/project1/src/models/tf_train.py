from __future__ import annotations
import os
import ssl
import time

# import matplotlib.pyplot as plt
# import numpy as np
import tensorflow as tf
from keras import backend as K
# from keras import datasets, layers, models
from tensorflow import keras
from tensorflow.python.client import device_lib
from tqdm import tqdm

from src.models.optuna_model import ConvNet
from src.data.dataloader import load_dataset

ssl._create_default_https_context = ssl._create_unverified_context

# built tensorflow with GPU

print("TENSORFLOW BUILT WITH CUDA: ", tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))


print("TENSORFLOW VISIBLE DEVIES: ", device_lib.list_local_devices())

method = "GPU"

if method == "GPU":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + K.epsilon())
    return recall_keras


if __name__ == "__main__":

    batch_size = 64
    train_dataset  = load_dataset(train=True, normalize=True, batch_size=batch_size, tune_for_perfomance=True)
    test_data = load_dataset(train=False, normalize=True, batch_size=batch_size, tune_for_perfomance=True)

    net = ConvNet()
    model = net.build_model()
    model.summary()

    # Instantiate an optimizer to train the model.
    optimizer = keras.optimizers.Adam(lr=1e-3)

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

        # Iterate over the batches of the dataset.
        # for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        for step, (x_batch_train, y_batch_train) in tqdm(
            enumerate(train_dataset), total=len(train_dataset)
        ):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update training metric.
            train_acc_metric.update_state(y_batch_train, logits)

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                # print("Seen so far: %d samples" % ((step + 1) * batch_size))

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

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
