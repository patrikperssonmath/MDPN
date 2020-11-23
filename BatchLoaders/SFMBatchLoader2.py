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
from Sfm.sfm_loader import sfm_loader
from Sample.PhotometricSample import PhotometricSample
import numpy as np
import random
import shutil
import os
import copy
from BatchLoaders.Sfm_loader_thread import Sfm_loader_thread


class SFMBatchLoader2:
    def __init__(self, config, optimizer):
        self.shape_size = config['model']['shape_size']
        self.min_covisibility = config['SFM']['min_covisibility']
        self.nbr_covisible_cameras = config['SFM']['nbr_covisible_cameras']
        self.logging_dir = config['model']['log_dir']
        self.optimizer = optimizer

        self.sfm_dataset = sfm_loader(config)
        self.sfm_dataset.load(loadImages=False)

        sfm_log_dir = os.path.join(self.logging_dir, "sfm")

        if os.path.exists(sfm_log_dir):
            shutil.rmtree(sfm_log_dir)

        os.makedirs(sfm_log_dir)

        self.sfm_summary_writer = tf.summary.create_file_writer(sfm_log_dir)

        self.loader_thread = Sfm_loader_thread(config)

    def getSummaryWriter(self):
        return self.sfm_summary_writer

    def getOpimizer(self):
        return self.optimizer

    def terminate(self):
        self.loader_thread.terminate()

    def setup(self, z_variables, alpha_variables):

        for sfm_image in self.sfm_dataset.getSFMDataset().values():

            key = sfm_image.getId()

            if key not in z_variables:
                z_variables.update({key: tf.Variable(np.zeros(
                    (self.shape_size), dtype=np.float32),  dtype=tf.float32)})

            if key not in alpha_variables:
                alpha_variables.update(
                    {key+"alpha": tf.Variable(-1.0,  dtype=tf.float32)})

        self.loader_thread.start()

    def getNbrBatches(self):
        return self.loader_thread.getNbrBatches()

    def reset(self):
        # self.loader_thread.reset()
        i = 0

    def getName(self):
        return "unsupervised"

    def getNext(self, z_variables, alpha_variables):

        covisible, new_epoch = self.loader_thread.fetch_next()

        if any(x is None for x in covisible) or len(covisible) == 0:
            return None, False

        if len(covisible) != self.nbr_covisible_cameras:
            return None, False

        z_batch, images, alphas, T, Tinv, U, ids = self.gather_batch(
            z_variables, alpha_variables, covisible)

        return PhotometricSample(images, z_batch, alphas, T, Tinv, U, ids, covisible,
                                 self.sfm_dataset.getDatasetName(ids[0]),
                                 self.optimizer), new_epoch

    def gather_batch(self, z_variables, alpha_variables, covisible_images):

        z_batch = []
        images = []
        alphas = []
        T = []
        Tinv = []
        U = []
        ids = []

        for j, image in enumerate(covisible_images):

            z_batch = [*z_batch, z_variables[image.getId()]]

            alphas = [*alphas, alpha_variables[image.getId()+"alpha"]]

            images = [
                *images, tf.convert_to_tensor(image.getImage(), dtype=tf.float32)]

            T1, Tinv1 = image.getTransformations()

            T = [*T, tf.convert_to_tensor(T1, dtype=tf.float32)]
            Tinv = [*Tinv, tf.convert_to_tensor(Tinv1, dtype=tf.float32)]

            U = [*U, image.getMeanDepth()]
            ids = [*ids, image.uuid]

        return z_batch, images, alphas, T, Tinv, U, ids
