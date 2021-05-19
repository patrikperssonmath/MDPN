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
from .InfereSparse import InfereSparse
from .InferePhotometric import InferePhotometric


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

        self.infer_sparse = InfereSparse(config)
        self.infer_photo = InferePhotometric(config)

    def updateLearningRate(self, lr):
        self.optimizer.learning_rate.assign(lr)

    def getLearningRate(self):
        return self.optimizer.learning_rate

    def getCheckPointVariables(self):
        return {"photometric_optimizer": self.optimizer}

    def predict_sparse(self, I, z, alpha, s_depths, calib, network):

        self.timer.start()

        I_batch = tf.stack(I)
        calib_batch = tf.stack(calib)

        z_batch = tf.stack(z)
        alpha_batch = tf.stack(alpha)

        max_len = 20000

        mask_depths = [tf.range(0, max_len) < s_depth.shape[1]
                       for s_depth in s_depths]

        mask_depths = tf.stack(mask_depths)

        s_depths = [tf.pad(tf.constant(s_depth, dtype=tf.float32), [[0, 0], [0, max_len-s_depth.shape[1]]])
                    for s_depth in s_depths]

        s_depths = tf.stack(s_depths)

        t1 = time.perf_counter()*1000

        z_res, alpha_res, loss_val, iterations = self.infer_sparse.infere(I_batch, calib_batch, z_batch, alpha_batch,
                                                                  s_depths, mask_depths, network)

        for i, e in enumerate(tf.unstack(z_res)):
            z[i].assign(e)

        for i, e in enumerate(tf.unstack(alpha_res)):
            alpha[i].assign(e)

        diff = time.perf_counter()*1000 - t1

        print('\nItr: {0} of {1}. Time {2} ms, per itr: {3}: loss {4}\n'.format(
            str(iterations.numpy()), str(self.max_iterations), str(diff), str(diff/(iterations.numpy())), str(loss_val.numpy())))

        return loss_val

    def predict(self, I, z, alpha, T, Tinv, calib, network):

        I_batch = tf.stack(I)
        T_batch = tf.stack(T)
        Tinv_batch = tf.stack(Tinv)
        calib_batch = tf.stack(calib)
        alpha_batch = tf.stack(alpha)
        z_batch = tf.stack(z)

        t1 = time.perf_counter()*1000

        z_res, alpha_res, loss_val, iterations = self.infer_photo.infere(I_batch, T_batch, Tinv_batch,
                                                                          calib_batch, z_batch, alpha_batch,
                                                                          network)

        for i, e in enumerate(tf.unstack(z_res)):
            z[i].assign(e)

        for i, e in enumerate(tf.unstack(alpha_res)):
            alpha[i].assign(e)

        diff = time.perf_counter()*1000 - t1

        print('\nItr: {0} of {1}. Time {2} ms, per itr: {3}: loss {4}\n'.format(
            str(iterations.numpy()), str(self.max_iterations), str(diff), str(diff/(iterations.numpy())), str(loss_val.numpy())))

        return loss_val

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
