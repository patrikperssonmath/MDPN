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

import numpy as np
from Graphics.Graphics3 import Graphics3
import unittest
import tensorflow as tf
import json
from PIL import Image
# Own packages
from Sfm.sfm_image import sfm_image
import OpenEXR

import Imath
import uuid

import argparse
import yaml
import time

from GTAReader import GTAReader

config = []


def get_exr_rgb(path):
    I = OpenEXR.InputFile(path)
    dw = I.header()['displayWindow']
    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    data = [np.frombuffer(c, np.float16).reshape(size)
            for c in I.channels('Y')]
    return (np.array(data[0])).astype(np.float32)


class TestStringMethods(unittest.TestCase):

    def test_gta_gt_test(self):

        g = Graphics3()
        path = "/database/gta_depth/GTAV_720/0000"
        reader = GTAReader("/database/gta_depth/GTAV_720/0000")

        batchSize = 5

        images = []

        sfm_image, depth = reader.next()

        for i in range(batchSize):
            sfm_image, depth = reader.next()

            depth = tf.convert_to_tensor(depth)

            depth = tf.where(tf.math.is_finite(depth),
                             depth, tf.zeros_like(depth))

            sfm_image.depth = tf.expand_dims(depth, axis=-1)

            images = [*images, sfm_image]

        T_batch = []
        Tinv_batch = []
        D_batch = []
        I_batch = []

        calib_batch = []

        for im in images:

            T, Tinv = im.getTransformations()

            D = im.getDepth()
            image = im.getImage()
            fx, fy, x0, y0 = im.getCalib()

            T_batch = [*T_batch, tf.constant(T, dtype=tf.float32)]
            Tinv_batch = [*Tinv_batch, tf.constant(Tinv, dtype=tf.float32)]
            D_batch = [*D_batch, tf.constant(D, dtype=tf.float32)]
            I_batch = [*I_batch,  tf.constant(image, dtype=tf.float32)]
            calib_batch = [*calib_batch,
                           tf.constant([fx, fy, x0, y0], dtype=tf.float32)]

        T_batch = tf.stack(T_batch)
        Tinv_batch = tf.stack(Tinv_batch)
        D_batch = tf.stack(D_batch)
        I_batch = tf.stack(I_batch)
        calib_batch = tf.stack(calib_batch)

        shape = I_batch.get_shape()

        w = shape[2]  # tf.constant(self.shape[1], dtype=tf.float32)
        h = shape[1]  # tf.constant(self.shape[0], dtype=tf.float32)

        shape_i = tf.shape(I_batch)

        X = g.unproject(D_batch, Tinv_batch)

        XX = tf.transpose(X[:, 0:3, :], perm=[0, 2, 1])

        II = tf.reshape(
            I_batch, shape=[shape_i[0], shape_i[1]*shape_i[2], shape_i[3]])

        # for i in range(shape_i[0]):

        #    g1.write_ply(II[i], XX[i], "im_"+str(i),
        #                 "/data/unit_test_results/")

        I_warp, D_warp, d_projected_warp, mask = g.warp(I_batch,
                                                        D_batch,
                                                        T_batch,
                                                        Tinv_batch)

        relative_depth_error = g.calculate_relative_depth_error(
            D_warp, d_projected_warp)

        error_image = g.calculate_error_image(I_warp, I_batch)

        angle = np.cos(85.0*np.pi/180.0).astype(np.float32)

        mask = g.calculate_masks(D_batch, calib_batch, mask, angle)

        shape_d = D_batch.get_shape()

        mask_I = g.calculate_Identity_mask(
            shape_d[2], shape_d[1], shape_d[0])

        mask = tf.logical_and(mask, mask_I)

        mask_g = g.calculate_occlusion_mask(relative_depth_error, mask)

        mask = tf.logical_and(mask, mask_g)

        I_warp_masked = tf.where(mask, I_warp, tf.zeros_like(I_warp))
        error_image_masked = tf.where(
            mask, error_image, tf.zeros_like(error_image))
        D_warp_masked = tf.where(mask, D_warp, tf.zeros_like(D_warp))

        

        for i in range(shape_i[0]):
            for j in range(shape_i[0]):

                I_ij = I_warp_masked[i, j].numpy()

                im = Image.fromarray((I_ij*255.0).astype(np.uint8))

                im.save("/data/unit_test_results/im_"+str(i)+"_"+str(j)+".png")

                I_ij = I_batch[j].numpy()

                im = Image.fromarray((I_ij*255.0).astype(np.uint8))

                im.save("/data/unit_test_results/im_ref_" +
                        str(i)+"_"+str(j)+".png")

                Ie_ij = tf.abs(error_image_masked[i, j]).numpy()

                im = Image.fromarray((Ie_ij*255.0).astype(np.uint8))

                im.save("/data/unit_test_results/im_e_" +
                        str(i)+"_"+str(j)+".png")

                d_max = tf.reduce_max(D_warp_masked[i, j])

                D_s = D_warp_masked[i, j]/d_max

                d_ij = D_s.numpy()

                im = Image.fromarray((d_ij[:, :, 0]*255.0).astype(np.uint8))

                im.save("/data/unit_test_results/depth_" +
                        str(i)+"_"+str(j)+".png")
        

        mask_s = tf.squeeze(mask)

        error_image = tf.where(
            mask, error_image, tf.zeros_like(error_image))

        relative_depth_error = tf.where(
            mask, relative_depth_error, tf.zeros_like(relative_depth_error))

        error_photometric = tf.reduce_mean(g.Huber(error_image, 0.1))

        error_depth = tf.reduce_mean(g.Huber(relative_depth_error, 0.1))

        relative_depth_error, error_image, mask = g.calculate_error_image_depth(I_batch,
                                                                                D_batch,
                                                                                T_batch,
                                                                                Tinv_batch,
                                                                                calib_batch,
                                                                                angle,
                                                                                w, h)

        error_photometric, error_depth = g.evaluate_photogeometric_error(I_batch,
                                                                         D_batch,
                                                                         T_batch,
                                                                         Tinv_batch,
                                                                         calib_batch,
                                                                         angle)

        t1 = time.perf_counter()

        N = 10

        for i in range(N):

            error_photometric, error_depth = g.evaluate_photogeometric_error(I_batch,
                                                                             D_batch,
                                                                             T_batch,
                                                                             Tinv_batch,
                                                                             calib_batch,
                                                                             angle)

            """

            relative_depth_error, error_image, mask = g.calculate_error_image_depth(I_batch,
                                                                                    D_batch,
                                                                                    T_batch,
                                                                                    Tinv_batch,
                                                                                    calib_batch,
                                                                                    angle,
                                                                                    w, h)

            error_image = tf.where(
                mask, error_image, tf.zeros_like(error_image))

            relative_depth_error = tf.where(
                mask, relative_depth_error, tf.zeros_like(relative_depth_error))

            error_photometric = tf.reduce_mean(g.Huber(error_image, 0.1))

            error_depth = tf.reduce_mean(g.Huber(relative_depth_error, 0.1))
            """

            """

            
            mask_s = tf.squeeze(mask)

            error_photometric = tf.reduce_mean(g.Huber(
                tf.boolean_mask(error_image, mask_s), 0.1))

            error_depth = tf.reduce_mean(g.Huber(tf.boolean_mask(
                relative_depth_error, mask_s), 0.1))

            """

        t2 = time.perf_counter()

        print("test time: "+str(1000.0*(t2-t1)/N)+" ms")

        self.assertAlmostEqual(error_photometric.numpy(), 0.0, 2)
        self.assertAlmostEqual(error_depth.numpy(), 0.0, 2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('config', type=str)

    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    unittest.main(argv=['first-arg-is-ignored'], exit=False)
