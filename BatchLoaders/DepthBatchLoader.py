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
from DataLoaderTask.DataLoader import DataLoader
from Sample.DepthSample import DepthSample
import numpy as np
import shutil
import os


class DepthBatchLoader:
    def __init__(self, config, optimizer):
        self.shape_size = config['model']['shape_size']
        self.batch_size = config['model']['batch_size']
        self.logging_dir = config['model']['log_dir']
        self.dataLoader = DataLoader(config)
        self.optimizer = optimizer
        self.i = 0

        depth_log_dir = os.path.join(self.logging_dir, "depth")

        if os.path.exists(depth_log_dir):
            shutil.rmtree(depth_log_dir)

        os.makedirs(depth_log_dir)

        self.depth_summary_writer = tf.summary.create_file_writer(
            depth_log_dir)

    def getSummaryWriter(self):
        return self.depth_summary_writer

    def getOpimizer(self):
        return self.optimizer

    def terminate(self):
        self.dataLoader.setExit()

    def setup(self, z_variables, alpha_variables):

        ids = self.dataLoader.getIds()

        for key in ids:

            # if key not in z_variables:
            #    z_variables.update({key: tf.Variable(np.random.normal(
            #        0.0, 1e-4, (self.shape_size)), dtype=tf.float32)})

            if key not in z_variables:
                z_variables.update({key: tf.Variable(np.zeros(
                    (self.shape_size), dtype=np.float32),  dtype=tf.float32)})

            if key not in alpha_variables:
                alpha_variables.update(
                    {key+"alpha": tf.Variable(-1.0,  dtype=tf.float32)})

        self.dataLoader.start()

    def getNbrBatches(self):
        return int(np.floor(self.dataLoader.getNbrSamples()/self.batch_size))

    def reset(self):
        self.i = 0

    def getName(self):
        return "supervised"

    def getNext(self, z_variables, alpha_variables):

        epoch_done = False
        batch, succ = self.dataLoader.next()

        if succ:
            self.i = self.i + 1
        else:
            return None, epoch_done

        if self.i >= self.dataLoader.getNbrSamples()/self.batch_size:
            self.i = 0
            epoch_done = True

        z = []
        alpha = []
        I = []
        D = []
        mask = []
        ids = []
        K = []

        for i in range(len(batch)):
            z.append(z_variables[batch[i].id])
            alpha.append(alpha_variables[batch[i].id+"alpha"])
            I.append(tf.convert_to_tensor(batch[i].image, dtype=tf.float32))
            D.append(tf.expand_dims(tf.convert_to_tensor(
                batch[i].depth, dtype=tf.float32), axis=-1))
            mask.append(tf.convert_to_tensor(batch[i].mask, dtype=tf.float32))
            ids.append(batch[i].id)
            K.append(batch[i].K)

        return DepthSample(K, I, D, mask, z, alpha, ids, self.optimizer), epoch_done
