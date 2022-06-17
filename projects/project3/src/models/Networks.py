import tensorflow as tf

from keras.layers import Conv2D,Conv2DTranspose, Dropout, Input, Activation, BatchNormalization, concatenate, add
#from keras import backend as K, regularizers
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from collections import defaultdict

from projects.utils import get_project3_root
from projects.project3.src.metrics.losses import *
from projects.project3.src.metrics.eval_metrics import *


class Pix2Pix_Unet():
    def __init__(self,loss_f, 
                train_dataset=[],test_data=[],
                img_size=(256,256,3),
                gf=32, 
                num_conv=2,
                depth=4,
                lr=1e-4,
                dropout_percent=0.1, 
                batchnorm=True,
                metrics=[],
    ):

        #Data
        self.train_dataset = train_dataset
        self.test_data = test_data

        #Network
        self.img_shape = img_size
        self.gf = gf  # Number of filters in the first layer of G
        self.num_conv = num_conv
        self.depth = depth
        self.batchnorm = batchnorm

        #training
        self.optimizer = Adam(learning_rate=lr)
        self.loss_func = loss_f 
        self.dropout_percent = dropout_percent
        self.metrics = metrics

        #initiate network
        self.unet = self.build_unet()
        self.out_dict = defaultdict(list)
        self.unet.compile(loss=self.loss_func,
                          optimizer=self.optimizer,
                          metrics=self.metrics)


    def build_unet(self):
        """U-Net Generator"""

        def conv2d(layer, filters, f_size=3, dropout=self.dropout_percent, downsample=True):
            #shortcut = layer
            for _ in range(self.num_conv):
                layer = Conv2D(filters, kernel_size=f_size, padding='same', strides=1)(layer)
                layer = BatchNormalization()(layer)
                layer = Activation('relu')(layer)
                layer = Dropout(dropout)(layer)
            if downsample:
                downsample = Conv2D(filters*2, kernel_size=f_size, padding='same', strides=2)(layer)
                if self.batchnorm:
                    downsample = BatchNormalization()(downsample)
                downsample = Activation('relu')(downsample)
            return layer, downsample

        #def convt_block(layer, concat, filters)
        def deconv2d(layer, concat, filters, f_size=3):
            layer = Conv2DTranspose(filters, kernel_size=f_size, padding='same', strides=2)(layer)
            if self.batchnorm:
                layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            layer = concatenate([layer, concat], axis=-1)
            return layer


        ## BUILT THE UNET STRUCTURE 

        #Input
        d0 = Input(shape=self.img_shape)
        d1 = Conv2D(16, kernel_size=7, padding='same', strides=1)(d0)

        enc_blocks = []
        #ENCODER
        for i in range(self.depth):
            block, d1 = conv2d(d1, self.gf*2**(i+1), dropout=.1)
            enc_blocks.append( block )

        #bottleneck
        d2, _ = conv2d(d1,self.gf*2**(self.depth), dropout=self.dropout_percent, downsample=False)

        #DECODER
        for i in range(self.depth):
            d2 = deconv2d(d2, enc_blocks[-(i+1)], self.gf*2**(self.depth-i-1)) 
            d2, _ = conv2d(d2, self.gf*2**(self.depth-i-1), dropout=self.dropout_percent, downsample=False)

        output = Conv2D(1,kernel_size=3, padding='same', strides=1)(d2) #, activation='relu'

        return Model(d0, output)
        


    def train(self, epochs, STEPS_PER_EPOCH=None, sample_interval_epoch=None):

        # for epoch in range(epochs):
        for epoch in tqdm(range(epochs), unit="epoch"):
            print("\nStart of epoch %d" % (epoch,))
            #start_time = time.time()

            train_acc = []
            train_loss = []
            train_recall = []
            train_n_correct_epoch = 0
            dataset_size = 0

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in tqdm(
                enumerate(self.train_dataset), total=len(self.train_dataset), #total=STEPS_PER_EPOCH,#)
            ):
                if step==STEPS_PER_EPOCH:
                    break

                with tf.GradientTape() as tape:
                    logits = self.unet(x_batch_train, training=True)
                    loss_value = self.loss_func(y_batch_train, logits)
                    grads = tape.gradient(loss_value, self.unet.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads, self.unet.trainable_weights))

                # Update training metric.
                # accuracy with tensorflow metric object
                #train_acc_metric.update_state(y_batch_train, logits)

                # custom accuracy computation with keras backend
                #predicted = K.cast(K.argmax(logits, axis=1), "uint8")  # one dimensional

                # y_targets = K.squeeze(y_batch_train, axis=1) #y_batch_train is 2 dimensional
                #y_targets = tf.cast(y_batch_train, tf.uint8)
                #train_n_correct_epoch += K.sum(tf.cast(y_targets == predicted, tf.float32))
                #dataset_size += len(y_batch_train)

                # training loss
                train_loss.append(loss_value.numpy())

                # custom computation of recall with keras backend
                #train_recall.append(recall(y_targets, predicted).numpy())

            # If at save interval => save generated image samples
            if sample_interval_epoch and epoch % sample_interval_epoch == 0:
                print("Saving test image of epoch:",epoch)

                (x_batch_val, y_batch_val) = next(iter(self.train_dataset))
                val_logits = self.unet(x_batch_val, training=False)
                val_probs = tf.keras.activations.sigmoid(val_logits)

                val_probs = tf.math.round(val_probs)

                for k in range(6):
                    plt.subplot(3, 6, k+1)
                    plt.imshow(x_batch_val[k,:,:,:], cmap='gray')
                    plt.title('Input')
                    plt.axis('off')

                    plt.subplot(3, 6, k+7)
                    plt.imshow(y_batch_val[k,:,:,:], cmap='gray')
                    plt.title('GT')
                    plt.axis('off')

                    plt.subplot(3, 6, k+13)
                    plt.imshow(val_probs[k,:,:,:], cmap='gray')
                    plt.title('Pred')
                    plt.axis('off')

                PROJECT_ROOT = get_project3_root()
                fig_path = PROJECT_ROOT / f"reports/figures/seg_predictions_epoch{epoch}.png"
                plt.savefig(fig_path)


            #out_dict["train_acc"].append(train_n_correct_epoch / dataset_size)
            self.out_dict["train_loss"].append(np.mean(train_loss))
            #out_dict["train_recall"].append(np.mean(train_recall))

            # Display metrics at the end of each epoch.
            #train_acc = train_acc_metric.result()

            print("Training loss over epoch: %.4f" % (float(np.mean(train_loss)),))
            #print("Training recall over epoch: %.4f" % (float(np.mean(train_recall)),))

            # Reset training metrics at the end of each epoch
            #train_acc_metric.reset_states()