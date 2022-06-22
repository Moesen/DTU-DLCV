import numpy as np
import tensorflow as tf

def discriminator_loss(fake: np.ndarray, real: np.ndarray):
    def loss_func(logits_in, labels_in):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits = logits_in, labels = labels_in))
    real_tf = tf.convert_to_tensor(real, dtype=tf.float32) 
    fake_tf = tf.convert_to_tensor(fake, dtype=tf.float32) 
    D_real_loss = loss_func(real_tf, tf.ones_like(real, dtype=tf.float32))
    D_fake_loss = loss_func(fake_tf, tf.zeros_like(real, dtype=tf.float32))
    print((D_real_loss + D_fake_loss).numpy().round(3))

if __name__ == "__main__":
    discriminator_loss(np.array([0.1, 0.5]), np.array([0.5, 0.9]))
