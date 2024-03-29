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

import time
from Graphics.Graphics2 import Graphics2
import numpy as np

import tensorflow as tf
from Sfm.sfm_loader import sfm_loader

import yaml
import argparse

from datetime import datetime

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

parser = argparse.ArgumentParser()

parser.add_argument('config', type=str)

args = parser.parse_args()

with open(args.config, 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

sfm_dataset = sfm_loader(config)
sfm_dataset.load()

sfm_ids = []

for id in sfm_dataset.getSFMDataset().keys():
    sfm_ids.append(id)

sfm_iteratior = iter(sfm_ids)


images = []
depths = []

batch_size = 5
T = []
Tinv = []

image_id = next(sfm_iteratior)

image = sfm_dataset.getImage(image_id)

covisibles = image.getCovisibleCameras()

scale = []

for i, im in enumerate(covisibles):

    if i >= batch_size:
        break

    image = sfm_dataset.getImage(im[0])

    images = [*images, tf.convert_to_tensor(image.getImage())]
    depths = [*depths, tf.ones([240, 320, 1])]

    T1, Tinv1 = image.getTransformations()

    T = [*T, T1]
    Tinv = [*Tinv, Tinv1]

    scale = [*scale, tf.constant(5.0, dtype=tf.float32)]


I = tf.stack(images)
D = tf.stack(depths)
alpha = tf.stack(scale)

g = Graphics2()

grid = g.generate_homogeneous_points(I)

M = g.create_transformation_matrix(T, Tinv)

g.transformBatch(D, M, grid, alpha)

print("testing")

t1 = time.perf_counter()

# Sets up a timestamped log directory.
logdir = "/data/logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# Creates a file writer for the log directory.
file_writer = tf.summary.create_file_writer(logdir)

# Using the file writer, log the reshaped image.
for i in range(50):
    mask, transformed_points, depths = g.transformBatch(D, M, grid, alpha)

    loss = g.photometric_loss(transformed_points, mask, I)

    loss += g.geometric_loss(transformed_points, depths, mask, D, alpha)

    images = g.warp_images(transformed_points, mask, I)

t2 = time.perf_counter()

with file_writer.as_default():
    tf.summary.image("Training data", images[0], step=0)

print((t2-t1)/1000.0)
