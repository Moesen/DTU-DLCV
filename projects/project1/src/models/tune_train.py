from __future__ import annotations

import os

import joblib
import optuna
import tensorflow as tf
import wandb
from tqdm import tqdm
from dotenv import find_dotenv, load_dotenv
from keras.models import Model
from src.data.dataloader import load_dataset
from src.models.optuna_model import build_model
from tensorflow import keras
from src.utils import get_project_root
from src.color_logger import init_logger
from pathlib import Path

log_path = Path("./log")
logger = init_logger(__name__, True, log_path)

# Get name of study
study_name = "250OptunaStudy_2"


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

        for x_batch_val, y_batch_val in validation_dataset:
            val_logits = model(x_batch_val, training=False)
            val_acc_metric.update_state(y_batch_val, val_logits)

        train_acc = train_acc_metric.result()
        val_acc = val_acc_metric.result()

        wandb_run.log({"validation accuracy": val_acc, "training accuracy": train_acc})

        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        train_acc_metric.reset_states()
        val_acc_metric.reset_states()
        logger.info(f"epoch: {epoch}/{epochs} finished")
    return val_acc


def objective(trial) -> float:
    # List of things we want parameters for
    #   - model
    #   - optimizer
    #   - loss_function
    #   - train_dataset

    trial_first_layer_channels = trial.suggest_int("first layer channels", 30, 100)
    trial_num_conv_blocks = trial.suggest_int("number convolutional blocks", 1, 5)
    trial_num_kernels = trial.suggest_int("num kernels", 1, 5)
    trial_image_size = trial.suggest_int("image size", 64, 400, 16)
    trial_dropout_percentage = trial.suggest_float("dropout_percentage", 0.1, 0.4)
    # trial_do_crop = trial.suggest_categorical("crop images", [True, False])
    trial_learning_rate = trial.suggest_loguniform("learning rate", 1e-9, 1e-3)
    trial_batch_size = trial.suggest_int("batch size", 32, 64, 16)
    trial_augmentation_flip = trial.suggest_categorical(
        "augmentation_flip",
        ["horizontal_and_vertical", "horizontal", "vertical", "none"],
    )
    trial_augmentation_rotation = trial.suggest_float("augmentation_rotation", 0.0, 0.3)
    trial_augmentation_contrast = trial.suggest_float("augmentation_contrast", 0.0, 0.8)
    trial_batchnorm = trial.suggest_categorical("batch norm", [True, False])
    trial_kernel_regularizer_strength = trial.suggest_loguniform(
        "kernel regularizer strength", 1e-25, 1e-1
    )
    trial_kernel_initializer = trial.suggest_categorical(
        "kernel initializer", ["he_normal", "he_uniform", "glorot_uniform"]
    )

    img_size = (trial_image_size, trial_image_size)
    img_shape = (*img_size, 3)

    config = {
        "first_layer_channels ": trial_first_layer_channels,
        "num_conv_blocks ": trial_num_conv_blocks,
        "image_size ": trial_image_size,
        "dropout_percentage ": trial_dropout_percentage,
        #l_do_crop ": trial_do_crop,
        "learning_rate ": trial_learning_rate,
        "batch_size ": trial_batch_size,
        "augmentation_flip ": trial_augmentation_flip,
        "augmentation_rotation ": trial_augmentation_rotation,
        "augmentation_contrast ": trial_augmentation_contrast,
        "batchnorm ": trial_batchnorm,
        "kernel_regularizer_strength ": trial_kernel_regularizer_strength,
        "kernel_initializer ": trial_kernel_initializer,
        "num_kernels": trial_num_kernels
    }

    wandb_run = wandb.init(
        project="project1",
        name=f"trial_{trial.number}",
        group=study_name,
        config=config,
        reinit=True  
    )

    model = build_model(
        trial_first_layer_channels,
        trial_num_conv_blocks,
        2,
        img_shape=img_shape,
        dropout_percentage=trial_dropout_percentage,
        do_batchnorm=trial_batchnorm,
        kernel_regularizer_strength=trial_kernel_regularizer_strength,
        kernel_initializer=trial_kernel_initializer,
        num_kernels = trial_num_kernels
    )

    train_dateset = load_dataset(
        train=True,
        batch_size=trial_batch_size,
        image_size=img_size,
        # crop_to_aspect_ratio=trial_do_crop,
        augmentation_flip=trial_augmentation_flip,
        augmentation_rotation=trial_augmentation_rotation,
        augmentation_contrast=trial_augmentation_contrast,
    )

    test_dataset = load_dataset(
        train=False,
        batch_size=32,
        image_size=img_size,
        # crop_to_aspect_ratio=trial_do_crop,
        use_data_augmentation=False,
    )

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam(learning_rate=trial_learning_rate)

    logger.info(f"Beginning {trial.number = }")
    num_epochs = 50
    try:
        acc = train_and_validate(
            trial,
            model,
            optimizer,
            loss_fn,
            train_dateset,
            test_dataset,
            wandb_run,
            num_epochs,
        )
    except:
        logger.warning(f"Encountered error when trying to run train_validate with parameters: ", config)
        raise optuna.TrialPruned()
    else:
        logger.info(f"Finished {trial.number = }")
        return acc


if __name__ == "__main__":
    # Load environment file. Should contain wandb key
    load_dotenv(find_dotenv())
    os.environ["WANDB_START_METHOD"] = "thread"
    # breakpoint()
    # Get root of project
    root = get_project_root()
    model_folder = root / "models"
    method = "GPU"
    if method == "GPU":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    logger.info("Instantiating study")
    if not (model_folder / (study_name + ".pkl")).is_file():
        logger.debug("Study not found in model folder, creating new one")
        sampler = optuna.samplers.TPESampler()
        pruner = optuna.pruners.PercentilePruner(
            20.0, n_startup_trials=10, n_warmup_steps=10, interval_steps=1
        )
        study = optuna.create_study(
            direction="maximize", sampler=sampler, pruner=pruner
        )
        logger.debug(f'Study path set to {(model_folder / (study_name + ".pkl"))}')
    else:
        logger.debug(f"Study with name {study_name} found in models folder")
        study = joblib.load(model_folder / (study_name + ".pkl"))

    logger.info("Beginning optuna optimization")
    study.optimize(objective, n_trials=250)

    joblib.dump(study, model_folder / (study_name + ".pkl"))
