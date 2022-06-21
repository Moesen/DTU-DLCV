from __future__ import annotations

import os

# Turn off tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import tensorflow as tf
import wandb
from dotenv import find_dotenv, load_dotenv
from projects.color_logger import init_logger
from projects.project3.src.data.dataloader import IsicDataSet
from projects.project3.src.metrics.eval_metrics import *
from projects.project3.src.metrics.losses import *
from projects.project3.src.models.Networks import Pix2Pix_Unet
from projects.utils import get_project3_root
from tensorflow import keras
from wandb.keras import WandbCallback

log_path = Path("./log")
logger = init_logger(__name__, True, log_path)

study_name = "unet_test_aug"

NUM_EPOCHS = 100
IMG_SIZE = (256, 256)  # (256,256,3)


class Timer:
    def __init__(self):
        self._last_time = time.time()
        self._timings = []
        self._sections = []
        self._training_time = []
        self._epoch_sections = []

    def add_new(self, name: str):
        self._timings += [time.time() - self._last_time]
        self._sections += [name]
        self._last_time = time.time()

    def add_new_training_time(self, epoch: str | int):
        self._epoch_sections += [epoch]
        self._training_time += [time.time() - self._last_time]
        self._last_time = time.time()

    def log_time(self):
        log_msg = "#" * 10 + " " * 5 + "TIMINGS" + " " * 5 + "#" * 10 + "\n"
        for s, t in zip(self._sections, self._timings):
            log_msg += f"\t{s}: \t{t} seconds \t\n"
        log_msg += "#" * 10 + " " * 5 + "TRAINING" + " " * 5 + "#" * 10 + "\n"
        for s, t in zip(self._epoch_sections, self._training_time):
            log_msg += f"\t{s}: \t{t} seconds \t\n"
        logger.info(log_msg)


def objective(trial: optuna.trial.Trial) -> float:
    logger.info(f"starting {trial.number = }")
    timer = Timer()

    # config
    c = dict(
        # Network
        first_layer_channels=trial.suggest_int("First layer channels", 10, 20),
        depth=trial.suggest_int("Depth", 2, 5),
        # Kernels
        num_kernels=trial.suggest_int("Convolutinal layers", 1, 3),
        # Learning
        dropout_percentage=trial.suggest_float("dropout_percentage", 0.1, 0.4),
        learning_rate=trial.suggest_loguniform("learning rate", 1e-5, 1e-3),
        batch_size=trial.suggest_int("batch size", 4, 16, 4),
        batchnorm=trial.suggest_categorical("batch norm", [True, False]),
        loss_func=trial.suggest_categorical(
            #  
            "Loss function", ["focal_loss", "dice_loss", "weighted_cross_entropy"]
        ),
        augmentation_flip=trial.suggest_categorical(
            "augmentation_flip",
            ["horizontal_and_vertical", "horizontal", "vertical", "none"],
        ),
        augmentation_rotation=trial.suggest_float("augmentation_rotation", 0.0, 0.2),
        augmentation_brightness=trial.suggest_float(
            "augmentation_brightness", 0.0, 1.0
        ),
        augmentation_contrast=trial.suggest_float("augmentation_contrast", 0.0, 0.6),
        augmentation_saturation=trial.suggest_float(
            "augmentation_saturation", 0.0, 1.0
        ),
        augmentation_hue=trial.suggest_float("augmentation_hue", 0.0, 0.5),
    )

    #fmt: off
    timer.add_new("creating config")
    switch = {
        "focal_loss"             : focal_loss,
        "dice_loss"              : dice_loss,
        "weighted_cross_entropy" : weighted_cross_entropy,
    }
    # fmt: on

    loss_func = switch[c["loss_func"]]()

    logger.info(
        f"config:\n{json.dumps(c, indent=4)}",
    )

    data_root = Path("/dtu/datasets1/02514/isic/train_allstyles")
    image_path = data_root / "Images"
    mask_path = data_root / "Segmentations"
    
    # fmt: off
    dataset_loader = IsicDataSet(
        image_folder          = image_path,
        mask_folder           = mask_path,
        image_size            = IMG_SIZE,
        image_channels        = 3,
        mask_channels         = 1,
        image_file_extension  = "jpg",
        mask_file_extension   = "png",
        do_normalize          = True,
        validation_percentage = 0.2,
        seed                  = 69,
        flipping              = c["augmentation_flip"],       # type: ignore
        rotation              = c["augmentation_rotation"],   # type: ignore
        brightness            = c["augmentation_brightness"], # type: ignore
        contrast              = c["augmentation_contrast"],   # type: ignore
        saturation            = c["augmentation_saturation"], # type: ignore
        hue                   = c["augmentation_hue"],        # type: ignore
    )
    # fmt: on

    timer.add_new("Dataset class")
    train_dataset, val_dataset = dataset_loader.get_dataset(
        batch_size=c["batch_size"], shuffle=True
    )

    timer.add_new("Dataset Loader")

    run = wandb.init(
        project="project3",
        name=f"trial_{trial.number}",
        group=study_name,
        config=c,
        reinit=True,
    )

    timer.add_new("Wandb")

    unet = Pix2Pix_Unet(
        loss_f=loss_func,
        train_dataset=[],  # given in fit below
        test_data=[],  # given in fit below
        img_size=(*IMG_SIZE, 3),
        gf=c["first_layer_channels"],
        num_conv=c["num_kernels"],
        depth=c["depth"],
        lr=c["learning_rate"],
        dropout_percent=c["dropout_percentage"],
        batchnorm=c["batchnorm"],
    )

    timer.add_new("Unet creation")

    # Callback functions
    def log_image(epoch, logs):
        (x_batch_val, y_batch_val) = next(iter(val_dataset))
        val_logits = unet.unet(x_batch_val, training=False)
        val_probs = tf.keras.activations.sigmoid(val_logits)
        val_probs = tf.math.round(val_probs)
        
        num_plots = 4
        _, axs = plt.subplots(3, num_plots)

        for i in range(num_plots):
            ax = axs[0, i]
            ax.imshow(x_batch_val[i, :, :, :], cmap="gray")
            ax.set_title("Input")
            ax.axis("off")

            ax = axs[1, i]
            ax.imshow(y_batch_val[i, :, :, :], cmap="gray")
            ax.set_title("GT")
            ax.axis("off")

            ax = axs[2, i]
            ax.imshow(val_probs[i, :, :, :], cmap="gray")
            ax.set_title("Pred")
            ax.axis("off")
        wandb.log({"Validation:": plt}, step=epoch)

    def log_time(epoch, logs):
        timer.add_new_training_time(epoch)

    # Callbacks
    image_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_image)
    wandb_callback = WandbCallback(
        monitor="val_loss",
        log_evaluation=False,
        save_model=False,
        validation_steps=len(val_dataset),
    )
    image_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_image)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        verbose=1,
        mode="min",
        restore_best_weights=True,
    )

    timer.add_new("Callback init")
    history = unet.unet.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=NUM_EPOCHS,
        callbacks=[early_stopping, image_callback, wandb_callback],
    )
    # fmt: on

    # Compute IoU for the best model
    total_iou = []
    print("Computing final metrics...")
    for (x_batch_val, true_mask) in val_dataset:
        for (val_img, val_GT_mask) in zip(x_batch_val, true_mask):
            val_logits = unet.unet(tf.expand_dims(val_img, 0), training=False)
            val_probs = tf.keras.activations.sigmoid(val_logits)
            pred_mask = tf.squeeze(tf.math.round(val_probs))

            compute_IoU = (
                tf.keras.metrics.BinaryIoU()
            )  # tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])
            batch_iou = compute_IoU(pred_mask, val_GT_mask)

            total_iou.append(batch_iou)

    best_iou = np.array(total_iou).mean()
    
    timer.log_time()
    run.log({"best model IoU": best_iou})  # type: ignore
    run.finish()  # type: ignore

    return best_iou 


if __name__ == "__main__":
    # Loading environment from .env file
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
    study.optimize(objective, n_trials=100)
