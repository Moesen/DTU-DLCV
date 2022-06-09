from __future__ import annotations

import os

# Turn off tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path

import joblib
import optuna
import tensorflow as tf
from dotenv import find_dotenv, load_dotenv
from keras.models import Model
from tensorflow import keras
from tqdm import tqdm
import json

import wandb
from wandb.keras import WandbCallback
from src.color_logger import init_logger
from src.data.dataloader import load_dataset
from src.models.optuna_model import build_model
from src.utils import get_project_root

log_path = Path("./log")
logger = init_logger(__name__, True, log_path)

study_name = "fit_test_100"


def objective(trial: optuna.trial.Trial) -> float:
    logger.info(f"starting {trial.number = }")

    # config
    c = dict(
        # Network
        first_layer_channels=trial.suggest_int("first layer channels", 20, 50),
        num_conv_blocks=trial.suggest_int("number convolutional blocks", 1, 5),
        # Kernels
        num_kernels=trial.suggest_int("num kernels", 2, 5),
        kernel_regularizer_strength=trial.suggest_loguniform(
            "kernel regularizer strength", 1e-10, 1e-2
        ),
        kernel_initializer=trial.suggest_categorical(
            "kernel initializer", ["he_normal", "glorot_uniform"]
        ),
        # Learning
        dropout_percentage=trial.suggest_float("dropout_percentage", 0.1, 0.8),
        learning_rate=trial.suggest_loguniform("learning rate", 1e-6, 1e-3),
        batch_size=trial.suggest_int("batch size", 32, 64, 16),
        batchnorm=trial.suggest_categorical("batch norm", [True, False]),
        # Img attributes
        image_size=trial.suggest_int("image size", 64, 256, 16),
        # Augmentations
        #augmentation_flip=trial.suggest_categorical(
        #    "augmentation_flip",
        #    ["horizontal_and_vertical", "horizontal", "vertical", "none"],
        #),
        augmentation_rotation=trial.suggest_float("augmentation_rotation", 0.0, 0.2),
        augmentation_contrast=trial.suggest_float("augmentation_contrast", 0.0, 0.6),
    )

    logger.info(f"config:\n{json.dumps(c, indent=4)}", )

    img_size = (c["image_size"], c["image_size"])
    img_shape = (*img_size, 3)

    train_ds, val_ds = load_dataset(
        train=True,
        batch_size=c["batch_size"],
        image_size=img_size,
        shuffle=True,
        validation_split=0.2,
        augmentation_flip="vertical",
        augmentation_rotation=c["augmentation_rotation"],
        augmentation_contrast=c["augmentation_contrast"],
    )

    run = wandb.init(
        project="project1",
        name=f"trial_{trial.number}",
        group=study_name,
        config=c,
        reinit=True,
    )

    model = build_model(
        c["first_layer_channels"],
        c["num_conv_blocks"],
        2,
        img_shape=img_shape,
        dropout_percentage=c["dropout_percentage"],
        do_batchnorm=c["batchnorm"],
        kernel_regularizer_strength=c["kernel_regularizer_strength"],
        kernel_initializer=c["kernel_initializer"],
        num_kernels=c["num_kernels"],
    )

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam(learning_rate=c["learning_rate"])

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1, mode='min')
    wandb_callback = WandbCallback(monitor="val_sparse_categorical_accuracy", log_evaluation=False, save_model=False, validation_steps = len(val_ds))

    # metrics
    metric = keras.metrics.SparseCategoricalAccuracy()

    # Compiling model with optimizer and loss function
    model.compile(optimizer, loss=loss_fn, metrics=[metric])
    history = model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=[early_stopping, wandb_callback])


    run.log({"best validation accuracy": max(history.history["val_sparse_categorical_accuracy"])}) # type: ignore
    run.finish() # type: ignore

    return max(history.history["val_sparse_categorical_accuracy"])


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    os.environ["WANDB_START_METHOD"] = "thread"

    # Get root of project
    root = get_project_root()

    # Checking to see if gpu is available
    avail = len(tf.config.list_physical_devices("GPU")) > 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(avail - 1)
    logger.info(f"Gpus available: {avail}")

    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.PercentilePruner(
        25.0, n_startup_trials=20, n_warmup_steps=10, interval_steps=1
    )
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    logger.info("Beginning optuna optimization")
    study.optimize(objective, n_trials=300)
