from __future__ import annotations

import datetime
import os
import ssl
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import backend as K
#from tensorflow import keras 
from tensorflow import keras
from tensorflow.python.client import device_lib
from tqdm import tqdm

from projects.project11.src.data.dataloader import load_dataset
from projects.project11.src.models import optuna_model
from projects.utils import get_project11_root

ssl._create_default_https_context = ssl._create_unverified_context

# built tensorflow with GPU

print("TENSORFLOW BUILT WITH CUDA: ", tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))


print("TENSORFLOW VISIBLE DEVIES: ", device_lib.list_local_devices())

method = "GPU"

if method == "GPU":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def recall(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + K.epsilon())
    return recall_keras


out_dict = defaultdict(list)


if __name__ == "__main__":
    save_model = True
    img_size = (160, 160)
    batch_size = 64

    train_dataset = load_dataset(
        train=True,
        normalize=True,
        shuffle=True,
        batch_size=batch_size,
        tune_for_perfomance=False,
        use_data_augmentation=True,
        augmentation_rotation= 0.1,
        augmentation_contrast= 0.55,
        augmentation_flip = "vertical",
        image_size=img_size,
    )

    test_data = load_dataset(
        train=False,
        normalize=True,
        batch_size=batch_size,
        tune_for_perfomance=False,
        use_data_augmentation=False,
        image_size=img_size,
    )

    model = optuna_model.build_model(46, 5, 2, (*img_size, 3), num_kernels=2, dropout_percentage=0.3, kernel_regularizer_strength=5e-9,do_batchnorm=False, do_dropout=True)
    model.summary()

    # Instantiate an optimizer to train the model.
    optimizer = keras.optimizers.Adam(learning_rate=1.2e-4)

    # Instantiate a loss function.
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Prepare the metrics.
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    epochs = 15
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

            # custom accuracy computation with keras backend
            predicted = K.cast(K.argmax(logits, axis=1), "uint8")  # one dimensional

            # y_targets = K.squeeze(y_batch_train, axis=1) #y_batch_train is 2 dimensional
            y_targets = tf.cast(y_batch_train, tf.uint8)
            train_n_correct_epoch += K.sum(tf.cast(y_targets == predicted, tf.float32))
            dataset_size += len(y_batch_train)
            # training loss
            train_loss.append(loss_value.numpy())

            # custom computation of recall with keras backend
            train_recall.append(recall(y_targets, predicted).numpy())

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at batch step %d: %.4f"
                    % (step, float(loss_value))
                )

        out_dict["train_acc"].append(train_n_correct_epoch / dataset_size)
        out_dict["train_loss"].append(np.mean(train_loss))
        out_dict["train_recall"].append(np.mean(train_recall))

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))
        print(
            "Training acc (numpy) over epoch: %.4f"
            % (float(train_n_correct_epoch / dataset_size),)
        )
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
        out_dict["val_acc"].append(val_acc.numpy())
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))
    
    #save last model 
    if save_model:
        PROJECT_ROOT = get_project11_root()
        model_name = 'hotdog_conv_'+datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        model_path = PROJECT_ROOT / "models" / model_name
        model.save(model_path)
