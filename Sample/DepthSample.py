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

import tensorflow as tf
import numpy as np


class DepthSample:
    def __init__(self, K, I, D, mask, z, alpha, ids, depth_optimizer):
        self.K = K
        self.I = I
        self.D = D
        self.mask = mask
        self.z = z
        self.alpha = alpha
        self.ids = ids
        self.depth_optimizer = depth_optimizer

        self.calib = []
        self.Kinv = []

        for k in self.K:

            self.calib = [
                *self.calib, tf.convert_to_tensor(self.getCalibVec(k), dtype=tf.float32)]

            self.Kinv = [*self.Kinv, self.getKinv(k)]

        self.calib = tf.stack(self.calib)
        self.Kinv = tf.stack(self.Kinv)

        for i in range(len(self.D)):
            if self.alpha[i] < 0:
                self.alpha[i].assign(tf.reduce_mean(self.D[i]))

    def getAlpha(self):
        return []

    def getZ(self):
        return self.z

    def getCalibVec(self, K):
        return [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]

    def getKinv(self, K):
        return tf.cast(np.array([[1.0/K[0, 0], 0, -K[0, 2]/K[0, 0], 0.0],
                                 [0.0, 1.0/K[1, 1], -K[1, 2]/K[1, 1], 0.0],
                                 [0.0, 0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0]]), dtype=tf.float32)

    def predict(self, network):
        # optimizes over z and alpha
        recon_loss = self.depth_optimizer.predict(
            self.I, self.D, self.mask, self.z, self.alpha, self.calib, network)

        return recon_loss

    def train(self, network):
        recon_loss, loss = self.depth_optimizer.train(
            self.I, self.D, self.mask, self.z, self.alpha, self.calib, network)

        return recon_loss, loss

    def writeTensorboard(self, network, step):

        D_batch_gt = tf.stack(self.D)

        g = self.depth_optimizer.get_graphics()

        calib_batch = tf.stack(self.calib)
        I_batch = tf.stack(self.I)
        z_batch = tf.stack(self.z)
        alpha_batch = tf.stack(self.alpha)

        R = g.normalized_points(I_batch, calib_batch)

        IR = tf.concat((I_batch, R), axis=-1)

        P, mu, logvar = network(IR, z_batch)

        D_batch = network.mapToDepth(
            alpha_batch, P)

        d_max = tf.reduce_max(D_batch)

        P, mu, logvar = network(IR, tf.zeros_like(z_batch))

        D_batch_z0 = network.mapToDepth(
            alpha_batch, P)

        d_max_z0 = tf.reduce_max(D_batch_z0)

        d_max_gt = tf.reduce_max(D_batch_gt)

        tf.summary.image('input_image', I_batch, step=step)
        tf.summary.image('D', D_batch/d_max, step=step)
        tf.summary.image('D_batch_gt', D_batch_gt/d_max_gt, step=step)
        tf.summary.image('D_z0', D_batch_z0/d_max_z0, step=step)
        tf.summary.histogram('hist_z', z_batch, step=step)

    def write(self, path, network, max_samples, angle_th):

        g = self.depth_optimizer.get_graphics()

        calib_batch = tf.stack(self.calib)
        I_batch = tf.stack(self.I)
        z_batch = tf.stack(self.z)
        alpha_batch = tf.stack(self.alpha)

        R = g.normalized_points(I_batch, calib_batch)

        IR = tf.concat((I_batch, R), axis=-1)

        P, mu, logvar = network(IR, z_batch)

        D_batch = network.mapToDepth(
            alpha_batch, P)

        P, mu, logvar = network(IR,  tf.zeros_like(z_batch))

        D_batch_z0 = network.mapToDepth(
            alpha_batch, P)

        D_batch_gt = tf.stack(self.D)

        shape = tf.shape(I_batch)

        I_flat = tf.reshape(
            I_batch, shape=[shape[0], shape[1]*shape[2], shape[3]])

        D_batch = g.unproject(D_batch, self.Kinv)

        D_batch = tf.transpose(D_batch[:, 0:3, :], perm=[0, 2, 1])

        D_batch_flat = tf.reshape(
            D_batch, shape=[shape[0], shape[1]*shape[2], 3])

        D_batch_z0 = g.unproject(D_batch_z0, self.Kinv)

        D_batch_z0 = tf.transpose(D_batch_z0[:, 0:3, :], perm=[0, 2, 1])

        D_batch_z0_flat = tf.reshape(
            D_batch_z0, shape=[shape[0], shape[1]*shape[2], 3])

        D_batch_gt = g.unproject(D_batch_gt, self.Kinv)

        D_batch_gt = tf.transpose(D_batch_gt[:, 0:3, :], perm=[0, 2, 1])

        D_batch_gt_flat = tf.reshape(
            D_batch_gt, shape=[shape[0], shape[1]*shape[2], 3])

        for i in range(len(self.I)):

            g.write_ply(I_flat[i], D_batch_gt_flat[i],
                        self.ids[i] + "_gt", path)

            g.write_ply(I_flat[i], D_batch_flat[i],
                        self.ids[i], path)

            g.write_ply(I_flat[i], D_batch_z0_flat[i],
                        self.ids[i] + "_z0", path)

        return len(self.I)
