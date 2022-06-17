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
from projects.color_logger import init_logger

import tensorflow as tf
from tensorflow.python.client import device_lib

from glob import glob
import datetime

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from collections import defaultdict

from projects.utils import get_project3_root
#from projects.project3.src.data.simple_dataloader import basic_loader
from projects.project3.src.data.dataloader import IsicDataSet
from projects.project3.src.models.Networks import Pix2Pix_Unet
from projects.project3.src.metrics.losses import *
from projects.project3.src.metrics.eval_metrics import *



log_path = Path("./log")
logger = init_logger(__name__, True, log_path)

study_name = "unet_test_100"

IMG_SIZE = (256,256) #(256,256,3)


def objective(trial: optuna.trial.Trial) -> float:
    logger.info(f"starting {trial.number = }")

    # config
    c = dict(
        # Network
        first_layer_channels=trial.suggest_int("First layer channels", 20, 50),
        depth=trial.suggest_int("Depth", 2, 5),

        # Kernels
        num_kernels=trial.suggest_int("Convolutinal layers", 1, 3),
        #kernel_regularizer_strength=trial.suggest_loguniform(
        #    "kernel regularizer strength", 1e-10, 1e-2
        #),
        #kernel_initializer=trial.suggest_categorical(
        #    "kernel initializer", ["he_normal", "glorot_uniform"]
        #),
        # Learning
        dropout_percentage=trial.suggest_float("dropout_percentage", 0.1, 0.8),
        learning_rate=trial.suggest_loguniform("learning rate", 1e-6, 1e-3),
        batch_size=trial.suggest_int("batch size", 8, 32, 8),
        batchnorm=trial.suggest_categorical("batch norm", [True, False]),
        loss_func=trial.suggest_categorical(
            "Loss function", [tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                                focal_loss(), 
                                dice_loss(),
                                weighted_cross_entropy()]
        )
        # Img attributes
        #image_size=trial.suggest_int("image size", 64, 256, 16),
        # Augmentations
        #augmentation_flip=trial.suggest_categorical(
        #    "augmentation_flip",
        #    ["horizontal_and_vertical", "horizontal", "vertical", "none"],
        #),
        #augmentation_rotation=trial.suggest_float("augmentation_rotation", 0.0, 0.2),
        #augmentation_contrast=trial.suggest_float("augmentation_contrast", 0.0, 0.6),
    )

    logger.info(f"config:\n{json.dumps(c, indent=4)}", )

    """img_size = (c["image_size"], c["image_size"])
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
    )"""

    proot = get_project3_root()
    data_root = proot / "data/isic/train_allstyles"
    image_path = data_root / "Images"
    mask_path = data_root / "Segmentations"

    dataset_loader = IsicDataSet(
        image_folder=image_path,
        mask_folder=mask_path,
        image_size=IMG_SIZE,
        image_channels=3,
        mask_channels=1,
        image_file_extension="jpg",
        mask_file_extension="png",
        do_normalize=True,
    )

    train_dataset = dataset_loader.get_dataset(batch_size=c["batch_size"], shuffle=True)

    run = wandb.init(
        project="project3",
        name=f"trial_{trial.number}",
        group=study_name,
        config=c,
        reinit=True,
    )

    # metrics
    metric = keras.metrics.SparseCategoricalAccuracy()

    unet = Pix2Pix_Unet(loss_f= c["loss_func"], 
                        train_dataset=[], #given in fit below
                        test_data=[], #given in fit below
                        img_size=(*IMG_SIZE, 3),
                        gf=c["first_layer_channels"],
                        num_conv=c["num_kernels"],
                        depth=c["depth"],
                        lr=c["learning_rate"],
                        dropout_percent=c["dropout_percentage"],
                        batchnorm=c["batchnorm"],
                        )

    unet.unet.summary()

    num_epochs = 100
    unet.train(epochs=num_epochs)

    # Callbacks
    #early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1, mode='min')
    #wandb_callback = WandbCallback(monitor="val_sparse_categorical_accuracy", log_evaluation=False, save_model=False, validation_steps = len(val_ds))

    # Compiling model with optimizer and loss function
    #model.compile(optimizer, loss=loss_fn, metrics=[metric])
    
    history = unet.unet.fit(train_dataset, validation_data=val_dataset, epochs=100, callbacks=[early_stopping, wandb_callback])


    run.log({"best validation accuracy": max(history.history["val_sparse_categorical_accuracy"])}) # type: ignore
    run.finish() # type: ignore

    return max(history.history["val_sparse_categorical_accuracy"])


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    os.environ["WANDB_START_METHOD"] = "thread"

    # Get root of project
    root = get_project3_root()

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
    study.optimize(objective, n_trials=50)
