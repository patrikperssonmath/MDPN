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
        self.sparse_test = config['Sparse']['test']
        self.g = Graphics3()
        self.optimizer = Adamax(lr=1e-3)
        self.timer = Timer(config)

        self.angle_th = np.cos(
            (config['PhotometricOptimizer']['angle_th']/180.0) * np.pi)

        self.angle_th = tf.constant(self.angle_th, dtype=tf.float32)

        self.infer_sparse = InfereSparse(config)
        self.infer_photo = InferePhotometric(config)

        self.results_sparse = {}

    def store_results(self):

        if not self.store_results:
            return

        accumulated = {}
        count = {}

        for key, v in self.results_sparse.items():

            for val in v:

                if val[0] not in accumulated:
                    accumulated[val[0]] = val[1]
                    count[val[0]] = 1

                else:
                    accumulated[val[0]] += val[1]
                    count[val[0]] += 1

        k = list(accumulated.keys())

        k.sort()

        result = []

        for key in k:
            result = [*result, accumulated[key]/count[key]]

        f = open("./data_sparse/data.cvs", "w+")

        p = np.expand_dims(np.array(k), axis=0)
        err = np.expand_dims(np.array(result), axis=0)

        np.savetxt(f, p, delimiter=',')
        np.savetxt(f, err, delimiter=',')

        f.close()

    def updateLearningRate(self, lr):
        self.optimizer.learning_rate.assign(lr)

    def getLearningRate(self):
        return self.optimizer.learning_rate

    def getCheckPointVariables(self):
        return {"photometric_optimizer": self.optimizer}

    def predict_sparse(self, I, z, alpha, s_depths, calib, network, names):

        self.timer.start()

        iterations_test = 1
        percentages = [0.8]

        if self.sparse_test:
            iterations_test = 20

            percentages = [0.01, 0.02, 0.05, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7]

            for i, e in enumerate(z):
                z[i].assign(tf.zeros_like(e))

        errors_p = []

        I_batch = tf.stack(I)
        calib_batch = tf.stack(calib)

        z_batch = tf.stack(z)
        alpha_batch = tf.stack(alpha)

        for percentage in percentages:

            error_validation_array = []

            for i in range(iterations_test):

                max_len = 20000

                validation = 0.2

                mask_examples = [tf.range(0, s_depth.shape[1]) < int(
                    s_depth.shape[1] * percentage) for s_depth in s_depths]

                mask_examples = [tf.pad(m, [[0, max_len-m.get_shape()[0]]])
                                 for m in mask_examples]

                mask_examples = tf.stack(mask_examples)

                mask_examples_validation = [tf.range(0, s_depth.shape[1]) > int(
                    s_depth.shape[1] * (1-validation)) for s_depth in s_depths]

                mask_examples_validation = [tf.pad(m, [[0, max_len-m.get_shape()[0]]])
                                            for m in mask_examples_validation]

                mask_examples_validation = tf.stack(mask_examples_validation)

                mask_depths = [tf.range(0, max_len) < s_depth.shape[1]
                               for s_depth in s_depths]

                mask_depths = tf.stack(mask_depths)

                depth_permute = [tf.transpose(tf.random.shuffle(tf.transpose(tf.constant(s_depth, dtype=tf.float32), perm=[1, 0])), perm=[1, 0])
                                 for s_depth in s_depths]

                s_depths_extend = [tf.pad(s_depth, [[0, 0], [0, max_len-s_depth.shape[1]]])
                                   for s_depth in depth_permute]

                s_depths_extend = tf.stack(s_depths_extend)

                t1 = time.perf_counter()

                z_res, alpha_res, loss_val, iterations, error_validation = self.infer_sparse.infere(I_batch, calib_batch, z_batch, alpha_batch,
                                                                                                    s_depths_extend, mask_depths, mask_examples,
                                                                                                    mask_examples_validation, network)

                error_validation_array = [
                    *error_validation_array, error_validation.numpy()]

            avg_error = np.mean(error_validation_array, axis=0)

            print(avg_error)

            errors_p = [*errors_p, avg_error]

        for i, e in enumerate(tf.unstack(z_res)):
            z[i].assign(e)

        for i, e in enumerate(tf.unstack(alpha_res)):
            alpha[i].assign(e)

        diff = time.perf_counter() - t1

        print('\nItr: {0} of {1}. Time {2} ms, per itr: {3}: loss {4}\n'.format(
            str(iterations.numpy()), str(self.max_iterations), str(diff*1000), str(diff*1000/(iterations.numpy())), str(loss_val.numpy())))

        """
        f = open("./data_sparse/data.cvs", "w+")

        p = np.expand_dims(np.array(percentages), axis=0)
        err = np.expand_dims(np.array(errors_p), axis=0)

        np.savetxt(f, p, delimiter=',')
        np.savetxt(f, err, delimiter=',')

        f.close()
        """

        for i, name in enumerate(names):
            self.results_sparse[name] = [
                (percentages[j], e[i]) for j, e in enumerate(errors_p)]

        return loss_val

    def predict(self, I, z, alpha, T, Tinv, calib, network):

        I_batch = tf.stack(I)
        T_batch = tf.stack(T)
        Tinv_batch = tf.stack(Tinv)
        calib_batch = tf.stack(calib)
        alpha_batch = tf.stack(alpha)
        z_batch = tf.stack(z)

        t1 = time.perf_counter()

        z_res, alpha_res, loss_val, iterations = self.infer_photo.infere(I_batch, T_batch, Tinv_batch,
                                                                         calib_batch, z_batch, alpha_batch,
                                                                         network)

        for i, e in enumerate(tf.unstack(z_res)):
            z[i].assign(e)

        for i, e in enumerate(tf.unstack(alpha_res)):
            alpha[i].assign(e)

        diff = time.perf_counter() - t1

        print('\nItr: {0} of {1}. Time {2} ms, per itr: {3}: loss {4}\n'.format(
            str(iterations.numpy()), str(self.max_iterations), str(diff*1000), str(diff*1000/(iterations.numpy())), str(loss_val.numpy())))

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
