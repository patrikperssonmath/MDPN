# MIT License

# Copyright (c) 2020 Patrik Persson and Linn Öström

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from Graphics.Graphics3 import Graphics3
from tensorflow.keras.optimizers import Adam, Adamax
import tensorflow as tf
import numpy as np
import time
from Trainer.Timer import Timer
from PIL import Image
from tensorflow_addons.image import interpolate_bilinear


class PhotometricOptimizer2:
    def __init__(self, config):
        self.max_iterations = config['PhotometricOptimizer']['max_iterations']
        self.termination_crit = config['PhotometricOptimizer']['termination_crit']
        self.image_height = config['dataset']['image_height']
        self.image_width = config['dataset']['image_width']
        self.g = Graphics3()
        self.optimizer = Adamax(lr=1e-3)
        self.timer = Timer(config)

        self.angle_th = np.cos(
            (config['PhotometricOptimizer']['angle_th']/180.0) * np.pi)

        self.angle_th = tf.constant(self.angle_th, dtype=tf.float32)

    def updateLearningRate(self, lr):
        self.optimizer.learning_rate.assign(lr)

    def getLearningRate(self):
        return self.optimizer.learning_rate

    def getCheckPointVariables(self):
        return {"photometric_optimizer": self.optimizer}

    def predict_sparse(self, I, z, alpha, s_depth, calib, network):
        t1 = time.perf_counter()*1000

        self.timer.start()

        I_batch = tf.stack(I)
        calib_batch = tf.stack(calib)
        alpha_batch = tf.stack(alpha)
        s_depth = tf.stack(s_depth)

        self.timer.log("stack")

        R = self.g.normalized_points(I_batch, calib_batch)

        IR = tf.concat((I_batch, R), axis=-1)

        q, mu, logvar = network.encode(IR)

        count = tf.constant(0.0, dtype=tf.float32)

        mu_unstacked = tf.unstack(mu)

        trainable_variables = [*z, *alpha]

        for i, e in enumerate(mu_unstacked):

            if tf.reduce_sum(tf.abs(z[i])) == 0.0:
                z[i].assign(mu_unstacked[i])

        prev_loss = tf.constant(0.0, dtype=tf.float32)

        for i in range(self.max_iterations):

            z_batch = tf.stack(z)

            self.timer.log("gradient")

            loss_val = tf.constant(0.0, dtype=tf.float32)

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch([z_batch, alpha_batch])

                P = network.decode(q, z_batch)

                D_batch = network.mapToDepth(
                    alpha_batch, P)

                D_int = interpolate_bilinear(D_batch, s_depth, indexing='xy')

                loss_val += self.g.log_normal_loss(z_batch, mu, logvar)
                loss_val += tf.reduce_sum(tf.square(tf.reduce_mean(network.mapToDepth(
                    tf.ones_like(alpha_batch), P), axis=[1, 2, 3]) - 1.0))

            gradients = tape.gradient(loss_val, [z_batch, alpha_batch])

            rel_e = tf.abs(prev_loss - loss_val)/loss_val

            s, count, stop = self.evaluate_rel_error(rel_e, s, count)

            self.timer.log("update")

            g = [tf.unstack(gradient) for gradient in gradients]

            flat_list = [item for sublist in g for item in sublist]

            self.optimizer.apply_gradients(
                zip(flat_list, trainable_variables))

            self.timer.log("end")

            if i > 0:

                if stop:
                    break

            # print("loss, s", loss_val.numpy(), s.numpy())

        self.timer.print()

        diff = time.perf_counter()*1000 - t1

        print('\nItr: {0} of {1}. Time {2} ms, per itr: {3}: loss {4}\n'.format(
            str(i), str(self.max_iterations), str(diff), str(diff/(i+1)), str(loss_val.numpy())))

        return loss_val

    def predict(self, I, z, alpha, T, Tinv, calib, network):

        t1 = time.perf_counter()*1000

        self.timer.start()

        I_batch = tf.stack(I)
        T_batch = tf.stack(T)
        Tinv_batch = tf.stack(Tinv)
        calib_batch = tf.stack(calib)
        alpha_batch = tf.stack(alpha)

        self.timer.log("stack")

        R = self.g.normalized_points(I_batch, calib_batch)

        IR = tf.concat((I_batch, R), axis=-1)

        q, mu, logvar = network.encode(IR)

        self.timer.log("setup")

        s = tf.constant(1.0, dtype=tf.float32)

        count = tf.constant(0.0, dtype=tf.float32)

        loss_val = tf.constant(-10.0, dtype=tf.float32)

        mu_unstacked = tf.unstack(mu)
        sig_unstacked = tf.unstack(tf.math.exp(logvar))

        trainable_variables = [*z, *alpha]

        for i, e in enumerate(mu_unstacked):

            if tf.reduce_sum(tf.abs(z[i])) == 0.0:
                z[i].assign(mu_unstacked[i])

        for i in range(self.max_iterations):

            z_batch = tf.stack(z)

            self.timer.log("gradient")

            gradients, loss_val, s, count, stop = self.calculate_gradients(s, T_batch, Tinv_batch,
                                                                           calib_batch, I_batch,
                                                                           alpha_batch, z_batch,
                                                                           q, mu, logvar, network, loss_val, count)

            self.timer.log("update")

            g = [tf.unstack(gradient) for gradient in gradients]

            flat_list = [item for sublist in g for item in sublist]

            self.optimizer.apply_gradients(
                zip(flat_list, trainable_variables))

            self.timer.log("end")

            if i > 0:

                if stop:
                    break

            # print("loss, s", loss_val.numpy(), s.numpy())

        self.timer.print()

        diff = time.perf_counter()*1000 - t1

        print('\nItr: {0} of {1}. Time {2} ms, per itr: {3}: loss {4}\n'.format(
            str(i), str(self.max_iterations), str(diff), str(diff/(i+1)), str(loss_val.numpy())))

        return loss_val

    @tf.function
    def calculate_gradients(self, s, T_batch, Tinv_batch, calib_batch, I_batch, alpha_batch, z_batch, q, mu, logvar, network, prev_loss, count):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch([z_batch, alpha_batch])

            loss_val = self.prediction_error(
                s, T_batch, Tinv_batch, calib_batch, I_batch, alpha_batch, z_batch, q, network)

            loss_val += self.g.log_normal_loss(z_batch, mu, logvar)

        gradients = tape.gradient(loss_val, [z_batch, alpha_batch])

        rel_e = tf.abs(prev_loss - loss_val)/loss_val

        s, count, stop = self.evaluate_rel_error(rel_e, s, count)

        return gradients, loss_val, s, count, stop

    @tf.function
    def prediction_error(self, s, T_batch, Tinv_batch, calib_batch, I_batch, alpha_batch, z_batch, q, network):

        P = network.decode(q, z_batch)

        D_batch = network.mapToDepth(
            alpha_batch, P)

        self.timer.log("prediction")

        error_photometric, error_depth = self.g.evaluate_photogeometric_error(
            I_batch, D_batch, T_batch, Tinv_batch,  calib_batch, self.angle_th, alpha_batch)

        self.timer.log("error")

        loss_val = error_photometric + error_depth
        loss_val += tf.reduce_sum(tf.square(tf.reduce_mean(network.mapToDepth(
            tf.ones_like(alpha_batch), P), axis=[1, 2, 3]) - 1.0))

        return loss_val

    @tf.function
    def prediction_error_pyramid(self, s, T_batch, Tinv_batch, calib_batch, I_batch, alpha_batch, z_batch, q, network):

        D_batch = network.mapToDepth(
            alpha_batch, network.decode(q, z_batch))

        shape = tf.shape(I_batch)

        h = tf.cast(shape[1], dtype=tf.float32)
        w = tf.cast(shape[2], dtype=tf.float32)

        s_dinv = tf.reshape(tf.linalg.diag(
            [s, s, 1.0, 1.0]), shape=[1, 4, 4])

        s_d = tf.reshape(tf.linalg.diag(
            [1.0/s, 1.0/s, 1.0, 1.0]), shape=[1, 4, 4])

        T_batch_s = tf.matmul(s_d, T_batch)
        Tinv_batch_s = tf.matmul(Tinv_batch, s_dinv)

        h_s = tf.cast(tf.math.floordiv(
            h, s), dtype=tf.float32)
        w_s = tf.cast(tf.math.floordiv(
            w, s), dtype=tf.float32)

        I_batch_s = tf.image.resize(I_batch, [h_s, w_s])
        D_batch_s = tf.image.resize(D_batch, [h_s, w_s])

        self.timer.log("prediction")

        error_photometric, error_depth = self.g.evaluate_photogeometric_error(
            I_batch_s, D_batch_s, T_batch_s, Tinv_batch_s,  calib_batch/s, self.angle_th)

        self.timer.log("error")

        loss_val = error_photometric + error_depth

        return loss_val

    @ tf.function(experimental_compile=True)
    def evaluate_rel_error(self, rel_e, s, count):

        if rel_e < self.termination_crit/s:

            count += 1.0
        else:
            count = 0.0

        flag = tf.greater_equal(count, 5.0)

        if flag:
            s /= 2.0

            count = tf.zeros_like(count)

            flag = tf.logical_not(flag)

            if s < 1.0:
                s = tf.ones_like(s)
                flag = tf.logical_not(flag)

        return s, count, flag

    def initialize(self, I, z, alpha, T, Tinv, calib, network):

        I_batch = tf.stack(I)
        calib_batch = tf.stack(calib)

        R = self.g.normalized_points(I_batch, calib_batch)

        IR = tf.concat((I_batch, R), axis=-1)

        _, mu, logvar = network.encode(IR)

        mu_unstacked = tf.unstack(mu)

        for i, e in enumerate(mu_unstacked):

            if tf.reduce_sum(tf.abs(z[i])) == 0.0:
                z[i].assign(mu_unstacked[i])

    def train(self, I, z, alpha, T, Tinv, calib, network):

        I_batch = tf.stack(I)
        T_batch = tf.stack(T)
        Tinv_batch = tf.stack(Tinv)
        calib_batch = tf.stack(calib)
        alpha_batch = tf.stack(alpha)

        R = self.g.normalized_points(I_batch, calib_batch)

        IR = tf.concat((I_batch, R), axis=-1)

        z_batch = tf.stack(z)

        P, mu, logvar = network(IR, z_batch)

        D_batch = network.mapToDepth(alpha_batch, P)

        error_photometric, error_depth = self.g.evaluate_photogeometric_error(
            I_batch, D_batch, T_batch, Tinv_batch, calib_batch, self.angle_th, alpha_batch)

        recon_loss = error_photometric + error_depth

        loss_val = recon_loss
        loss_val += self.g.log_normal_loss(z_batch, mu, logvar)
        loss_val += network.sum_losses()
        loss_val += tf.reduce_sum(tf.square(tf.reduce_mean(network.mapToDepth(
            tf.ones_like(alpha_batch), P), axis=[1, 2, 3]) - 1.0))

        return recon_loss, loss_val

    def get_graphics(self):

        return self.g
