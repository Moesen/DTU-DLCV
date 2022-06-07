from __future__ import annotations

import os

import numpy as np
import optuna
import tensorflow as tf
import wandb
from keras.models import Model
from tensorflow import keras
from tqdm import tqdm

from src.data.dataloader import load_dataset
from src.models.optuna_model import build_model


def train_and_validate(
    trial,
    model: Model,
    optimizer,
    loss_function,
    train_dataset,
    validation_dataset,
    wandb_run,
    epochs: int,
) -> float:
    """train.
    :param model:
    :type model: Model
    Model created with convnet
    :param optimizer:
    Chosen optimizer from keras.optimizers
    :param loss_function:
    Loss function chosen from keras.losses
    :param x_train:
    dataset taken from get_dataset
    :rtype: float
    """

    # Constructing accuracy objects
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc = None

    for epoch in tqdm(range(epochs), unit="epoch"):
        # Itterating over the batches of the dataset
        for step, (x_batch_train, y_batch_train) in tqdm(
            enumerate(train_dataset), total=len(train_dataset)
        ):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_function(y_batch_train, logits)
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

            train_acc_metric.update_state(y_batch_train, logits)

            with wandb_run:
                wandb_run.log({"train_accuracy": train_acc_metric.result()})

        # 
        for x_batch_val, y_batch_val in validation_dataset:
            val_logits = model(x_batch_val, training=False)

            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()

    return val_acc


def objective(trial):
    # List of things we want parameters for
    #   - model
    #   - optimizer
    #   - loss_function
    #   - train_dataset

    trial_first_layer_channels = trial.suggest_int("first layer channels", 1, 50)
    trial_num_conv_blocks = trial.suggest_int("number convolutional blocks", 2, 10)
    trial_image_size = trial.suggest_int("image size", 16, 256, 16)
    trial_dropout_percentage = trial.suggest_float("dropout_percentage", 0.0, 0.6)
    trial_do_crop = trial.suggest_catagorical("crop images", [True, False])
    trial_learning_rate = trial.suggest_loguniform("learning rate", 1e-6, 1e-2)
    trial_batch_size = trial.suggest_int("batch size", 32, 128, 16)

    img_shape = (trial_image_size, trial_image_size, 3)

    config = {
        "trial_first_layer_channels": trial_first_layer_channels,
        "trial_num_conv_blocks": trial_num_conv_blocks,
        "trial_image_size": trial_image_size,
        "trial_dropout_percentage": trial_dropout_percentage,
        "trial_do_crop": trial_do_crop,
        "trial_learning_rate": trial_learning_rate,
        "trial_batch_size": trial_batch_size,
    }

    wandb_run = wandb.init(
        project="DTU-DLCV",
        name=f"trial_{trial.number}",
        group="sampling",
        config=config,
        reinit=True, # Dunno why this is needed but it is
    )

    model = build_model(
        trial_first_layer_channels,
        trial_num_conv_blocks,
        2,
        img_shape=img_shape,
        dropout_percentage=trial_dropout_percentage,
    )

    train_dateset = load_dataset(
        train=True,
        batch_size=trial_batch_size,
        image_size=img_shape,
        crop_to_aspect_ratio=trial_do_crop,
    )

    test_dataset = load_dataset(
        train=False,
        batch_size=32,
        img_size=img_shape,
        crop_to_aspect_ratio=trial_do_crop,
    )


if __name__ == "__main__":
    method = "GPU"
    if method == "GPU":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    pass
