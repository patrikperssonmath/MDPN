from Graphics.Graphics3 import Graphics3
from tensorflow.keras.optimizers import Adamax
import tensorflow as tf


class DepthOptimizer:
    def __init__(self, config):
        self.optimizer = Adamax(lr=1e-3)
        self.g = Graphics3()

    def get_graphics(self):
        return self.g

    def updateLearningRate(self, lr):
        self.optimizer.learning_rate.assign(lr)

    def getLearningRate(self):
        return self.optimizer.learning_rate

    def getCheckPointVariables(self):
        return {"depth_optimizer": self.optimizer}

    def predict(self, I, D, mask, z, alpha, calib, network):

        I_batch = tf.stack(I)
        D_batch = tf.stack(D)
        mask_batch = tf.stack(mask)
        calib_batch = tf.stack(calib)

        R = self.g.normalized_points(I_batch, calib_batch)

        IR = tf.concat((I_batch, R), axis=-1)

        q, mu, logvar = network.encode(IR)
        trainable_variables = z

        rel_error = tf.constant(0.0, dtype=tf.float32)

        for i in range(200):

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(trainable_variables)

                z_batch = tf.stack(z)

                alpha_batch = tf.stack(alpha)

                loss_next = self.predict_update(D_batch, mask_batch, q,
                                                z_batch, alpha_batch, network)

                loss_next += self.g.log_normal_loss(z_batch, mu, logvar)

            gradients = tape.gradient(loss_next, trainable_variables)

            self.optimizer.apply_gradients(
                zip(gradients, trainable_variables))

            if i > 0:

                rel_error = 0.8*(tf.abs(loss_prev-loss_next) /
                                 loss_next) + 0.2 * rel_error

            if i > 0 and rel_error < 1e-3:
                break

            loss_prev = loss_next

        return loss_next

    @tf.function(experimental_compile=False)
    def predict_update(self, D_batch, mask_batch, q, z_batch, alpha_batch, network):

        shape = tf.shape(D_batch)

        D_p = network.decode(q, z_batch)

        loss_val = self.loss(D_batch, mask_batch, D_p, alpha_batch)

        return loss_val

    def train(self, I, D, mask, z, alpha, calib, network):

        I_batch = tf.stack(I)
        D_batch = tf.stack(D)
        mask_batch = tf.stack(mask)
        alpha_batch = tf.stack(alpha)

        z_batch = tf.stack(z)

        recon_loss, loss_val = self.train_loss(
            I_batch, D_batch, mask_batch, z_batch, alpha_batch,
            calib, network)

        return recon_loss, loss_val

    @tf.function(experimental_compile=False)
    def train_loss(self, I_batch, D_batch, mask_batch, z_batch, alpha_batch, calib, network):

        shape = tf.shape(I_batch)

        calib_batch = tf.stack(calib)

        R = self.g.normalized_points(I_batch, calib_batch)

        IR = tf.concat((I_batch, R), axis=-1)

        P, mu, logvar = network(IR, z_batch)

        D_p = network.mapToDepth(alpha_batch, P)

        recon_loss = self.loss(D_batch, mask_batch, D_p, alpha_batch)
        reg_loss = network.sum_losses()
        loss_val = recon_loss + reg_loss
        loss_val += self.g.log_normal_loss(z_batch, mu, logvar)

        return recon_loss, loss_val

    def mapToProximal(self, A, Dinv):

        shape = tf.shape(Dinv)

        A = tf.reshape(A, shape=[shape[0], 1, 1, 1])

        return tf.math.divide_no_nan(A*tf.ones_like(Dinv), Dinv + A*tf.ones_like(Dinv))

    @tf.function(experimental_compile=False)
    def loss(self, y_true, mask, y_pred, alpha_batch):
        return tf.reduce_sum(tf.boolean_mask(tf.math.square(y_true-y_pred), mask))/0.01

    @tf.function(experimental_compile=False)
    def loss2(self, y_true, mask, y_pred, alpha_batch):

        Diff = y_true-y_pred

        shape = tf.shape(Diff)

        Diff = Diff/tf.reshape(alpha_batch, shape=[shape[0], 1, 1, 1])

        loss = tf.boolean_mask(self.g.Huber(Diff, 0.1), mask)

        return tf.reduce_sum(loss)/0.01
