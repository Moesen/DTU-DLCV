import scipy

#from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D

from keras.layers import Conv3D, Conv3DTranspose, Dropout, Input, Activation, BatchNormalization, concatenate, add
from keras import backend as K, regularizers
from glob import glob
import tensorflow as tf


from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys


import numpy as np
import os

from tqdm import tqdm

from projects.utils import get_project3_root
from projects.project3.src.data.dataloader import load_dataset


class Pix2Pix_Unet():
    def __init__(self,img_size=(128,128,3),batch_size=16, gf=32):
        
        self.img_size = img_size
        self.batch_size = batch_size
        self.gf = gf  # Number of filters in the first layer of G

        # Configure data loader
        self.train_dataset = load_dataset(train=True,
                                        normalize=True,
                                        shuffle=True,
                                        batch_size=self.batch_size,
                                        use_data_augmentation=False,
                                        image_size=img_size,
        )

        self.test_data = load_dataset(train=False,
                                    normalize=True,
                                    batch_size=self.batch_size,
                                    use_data_augmentation=False,
        )


        self.optimizer = Adam(1e-4)
        self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.unet = self.build_unet()

        #self.unet.compile(loss=self.loss_func,
        #                       optimizer=optimizer)


    def build_unet(self):
        """U-Net Generator"""

        def conv3d(layer,filters,f_size=3,dropout=0.1, downsample=True):
            shortcut = layer
            for i in range(1,3):
                layer = Conv3D(filters, kernel_size=f_size, kernel_regularizer=regularizers.l2(1e-1), kernel_initializer='glorot_uniform', padding='same',strides=(1,1,1))(layer)
                layer = BatchNormalization()(layer)
                layer = Activation('relu')(layer)
                layer = Dropout(dropout)(layer)
            if downsample:
                downsample = Conv3D(filters*2, kernel_size=f_size, kernel_regularizer=regularizers.l2(1e-1), kernel_initializer='he_normal', padding='same', strides=(2,2,2))(layer)
                downsample = BatchNormalization()(downsample)
                downsample = Activation('relu')(downsample)
            return layer, downsample

        #def convt_block(layer, concat, filters)
        def deconv3d(layer, concat, filters, f_size=3, stridevals=(2,2,2)):
            #layer = Conv3DTranspose(filters, (3,3,3), kernel_regularizer=regularizers.l2(1e-1), kernel_initializer='he_normal', padding='same', strides=(2,2,2))(layer)
            layer = Conv3DTranspose(filters, kernel_size=f_size, kernel_regularizer=regularizers.l2(1e-1), kernel_initializer='he_normal', padding='same', strides=stridevals)(layer)
            layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            layer = concatenate([layer, concat], axis=-1)
            return layer


        d0 = Input(shape=self.img_shape)
        block1, dblock1 = conv3d(d0,self.gf,dropout=.1)
        block2, dblock2 = conv3d(dblock1,self.gf*2,dropout=.1)
        block3, dblock3 = conv3d(dblock2,self.gf*4,dropout=.2)
        block4, dblock4 = conv3d(dblock3,self.gf*8,dropout=.2)
        block5, _ = conv3d(dblock4,self.gf*16,dropout=.3,downsample=False)
        
        
        # DECODING
        block7 = deconv3d(block5,block4,self.gf*8) 
        block8, _ = conv3d(block7,self.gf*8,dropout=.3,downsample=False)

        block9 = deconv3d(block8,block3,self.gf*4) 
        block10, _ = conv3d(block9,self.gf*4,dropout=.2,downsample=False)

        block11 = deconv3d(block10,block2,self.gf*2)
        block12, _ = conv3d(block11,self.gf*2,dropout=.1,downsample=False)

        block13 = deconv3d(block12,block1,self.gf)
        block14, _ = conv3d(block13,self.gf,dropout=.1,downsample=False)

        output = Conv3D(1,kernel_size=3, kernel_regularizer=regularizers.l2(1e-1), kernel_initializer='glorot_uniform', padding='same',strides=(1,1,1), activation='relu')(block14)        
        return Model(d0, output)
        



    def train(self, epochs, batch_size=1, sample_interval_epoch=30, save_interval_epoch=500):

        epochs = 15
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
                enumerate(self.train_dataset), total=len(self.train_dataset)
            ):
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
                #train_loss.append(loss_value.numpy())

                # custom computation of recall with keras backend
                #train_recall.append(recall(y_targets, predicted).numpy())

                # If at save interval => save generated image samples
                #if epoch % sample_interval_epoch == 0:
                #    self.sample_images(epoch, batch_i)

                # If at save interval => save generated model
                #if epoch % save_interval_epoch == 0:
                #    self.sample_images(epoch, batch_i)
                #    self.generator.save('%s/%s_e%d.h5' % (self.model_name, self.model_name, epoch))
                
                # Log every 200 batches.
                if step % 200 == 0:
                    print(
                        "Training loss (for one batch) at batch step %d: %.4f"
                        % (step, float(loss_value))
                    )

            #out_dict["train_acc"].append(train_n_correct_epoch / dataset_size)
            #out_dict["train_loss"].append(np.mean(train_loss))
            #out_dict["train_recall"].append(np.mean(train_recall))

            # Display metrics at the end of each epoch.
            #train_acc = train_acc_metric.result()
            #print("Training acc over epoch: %.4f" % (float(train_acc),))
            #print(
            #    "Training acc (numpy) over epoch: %.4f"
            #    % (float(train_n_correct_epoch / dataset_size),)
            #)
            #print("Training loss over epoch: %.4f" % (float(np.mean(train_loss)),))
            #print("Training recall over epoch: %.4f" % (float(np.mean(train_recall)),))

            # Reset training metrics at the end of each epoch
            #train_acc_metric.reset_states()

            # Run a validation loop at the end of each epoch.
            #for x_batch_val, y_batch_val in test_data:
            #    val_logits = model(x_batch_val, training=False)

                # Update val metrics
                #val_acc_metric.update_state(y_batch_val, val_logits)
            #val_acc = val_acc_metric.result()
            #out_dict["val_acc"].append(val_acc.numpy())
            #val_acc_metric.reset_states()
            #print("Validation acc: %.4f" % (float(val_acc),))
            #print("Time taken: %.2fs" % (time.time() - start_time))



if __name__ == '__main__':

    save_model = False

    unet = Pix2Pix_Unet()

    unet.train(epochs=5000, batch_size=16, sample_interval_epoch=10, save_interval_epoch=2)
    
    model_name = 'unet_'+datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    if save_model:
        PROJECT_ROOT = get_project3_root()
        model_path = PROJECT_ROOT / "models" / model_name
        unet.save(model_path)

    #unet.generator.save(unet.model_name+'/'+unet.model_name)
    #with open(unet.model_name+'/'+unet.model_name+'.json', "w") as json_file:
    #    json_file.write(unet.generator.to_json())