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
import os
import numpy as np
from PIL import Image, ImageColor
import shutil
import time
from colour import Color
import matplotlib as mpl


class PhotometricSample:

    def __init__(self, images, z, alpha, T, Tinv, U_mean, ids, sfm_images, dataset_name, photometric_optimizer):
        self.I = images
        self.z = z

        self.alpha = alpha

        for i, a in enumerate(self.alpha):
            if a < 0:
                a.assign(tf.constant(U_mean[i], dtype=tf.float32))

        self.T = T
        self.Tinv = Tinv
        self.ids = ids
        self.sfm_images = sfm_images
        self.dataset_name = dataset_name
        self.photometric_optimizer = photometric_optimizer
        self.U_mean = U_mean
        self.calib = []

        for image in self.sfm_images:

            self.calib = [*self.calib, image.getCalibVec()]

    def predict(self, network):

        # optimizes over z and alpha
        recon_loss = self.photometric_optimizer.predict(
            self.I, self.z, self.alpha, self.T, self.Tinv, self.calib, network)

        return recon_loss

    def predict_sparse(self, network):

        s_depth = []

        for image in self.sfm_images:

            s_depth = [*s_depth, image.getSparseLocalDepth()]

        # optimizes over z and alpha
        recon_loss = self.photometric_optimizer.predict_sparse(
            self.I, self.z, self.alpha, s_depth, self.calib, network)

        return recon_loss

    def train(self, network):

        recon_loss, loss = self.photometric_optimizer.train(
            self.I, self.z, self.alpha, self.T, self.Tinv, self.calib, network)

        return recon_loss, loss

    def initialize(self, network):
        self.photometric_optimizer.initialize(
            self.I, self.z, self.alpha, self.T, self.Tinv, self.calib, network)

    def getZ(self):
        return self.z

    def getAlpha(self):
        return self.alpha

    def writeTensorboard(self, network, step):

        g = self.photometric_optimizer.get_graphics()

        calib_batch = tf.stack(self.calib)
        I_batch = tf.stack(self.I)
        z_batch = tf.stack(self.z)
        alpha_batch = tf.stack(self.alpha)

        R = g.normalized_points(I_batch, calib_batch)

        IR = tf.concat((I_batch, R), axis=-1)

        P,  mu, logvar = network(IR, z_batch)

        D_batch = network.mapToDepth(
            alpha_batch, P)

        d_max = tf.reduce_max(D_batch)

        P,  mu, logvar = network(IR, tf.zeros_like(z_batch))

        D_batch_z0 = network.mapToDepth(
            alpha_batch, P)

        d_max_z0 = tf.reduce_max(D_batch_z0)

        tf.summary.image('input_image', I_batch, step=step)
        tf.summary.image('D', D_batch/d_max, step=step)
        tf.summary.image('D_z0', D_batch_z0/d_max_z0, step=step)
        tf.summary.histogram('hist_z', z_batch, step=step)

    def write(self, path, network, max_samples, angle_th):

        folder_dataset = os.path.join(
            path, self.dataset_name)

        if not os.path.exists(folder_dataset):
            os.makedirs(folder_dataset)

        g = self.photometric_optimizer.get_graphics()

        angle_th = tf.constant(angle_th, dtype=tf.float32)

        calib_batch = tf.stack(self.calib)
        Tinv_batch = tf.stack(self.Tinv)
        I_batch = tf.stack(self.I)
        z_batch = tf.stack(self.z)
        alpha_batch = tf.stack(self.alpha)

        P = []
        names = []

        for image in self.sfm_images:
            P = [*P, image.getP_Homogeneous()]
            names = [*names, image.getId()]

        P_batch = tf.stack(P)

        R = g.normalized_points(I_batch, calib_batch)

        IR = tf.concat((I_batch, R), axis=-1)

        P,  mu, logvar = network(IR, z_batch)

        D_batch = network.mapToDepth(
            alpha_batch, P)

        names_pred = [e+"_pred_filtered" for e in names]

        g.to_ply(D_batch, I_batch, Tinv_batch,
                 P_batch, calib_batch,  angle_th, names_pred, folder_dataset)

        names_pred = [e+"_pred" for e in names]

        g.to_ply(D_batch, I_batch, Tinv_batch,
                 P_batch, calib_batch,  angle_th, names_pred, folder_dataset, False)

        P,  mu, logvar = network(IR, tf.zeros_like(z_batch))

        D_batch = network.mapToDepth(
            alpha_batch, P)

        names_z0 = [e+"_initial" for e in names]

        g.to_ply(D_batch, I_batch, Tinv_batch,
                 P_batch, calib_batch,  angle_th, names_z0, folder_dataset, False)

        D_gt = []

        for im in self.sfm_images:

            depth = im.getDepth()

            if depth is not None:

                depth = tf.convert_to_tensor(depth, dtype=tf.float32)

                depth = tf.expand_dims(depth, axis=-1)

                D_gt.append(depth)

        if len(D_gt) != 0:

            D_gt = tf.stack(D_gt)

            names_z0 = [e+"_ground_truth" for e in names]

            g.to_ply(D_gt, I_batch, Tinv_batch,
                     P_batch, calib_batch,  angle_th, names_z0, folder_dataset, False)

        for im in self.sfm_images:

            name = im.getId()

            sp_depth = im.getSparseDepth()

            if sp_depth is None:
                continue

            red_pt = np.zeros(sp_depth.shape, dtype=np.float32)

            red_pt[0, :] = 1

            sp_depth = tf.transpose(
                tf.convert_to_tensor(sp_depth, dtype=tf.float32))

            red_pt = tf.transpose(tf.convert_to_tensor(red_pt))

            g.write_ply(
                red_pt, sp_depth, name+"_ground_truth", folder_dataset)

        return len(self.sfm_images)

    def write_depth(self, path, network):

        sfm_image = self.sfm_images[0]

        folder_dataset = os.path.join(
            path, self.dataset_name)

        pred_path = os.path.join(folder_dataset, "pred")

        if not os.path.exists(pred_path):
            os.makedirs(pred_path)

        gt_path = os.path.join(folder_dataset, "gt")

        if not os.path.exists(gt_path):
            os.makedirs(gt_path)

        g = self.photometric_optimizer.get_graphics()

        calib_batch = tf.stack(self.calib)
        I_batch = tf.stack(self.I)
        z_batch = tf.stack(self.z)
        alpha_batch = tf.stack(self.alpha)

        R = g.normalized_points(I_batch, calib_batch)

        IR = tf.concat((I_batch, R), axis=-1)

        P,  mu, logvar = network(IR, z_batch)

        D_batch = network.mapToDepth(
            alpha_batch, P)

        depth_prediction = D_batch[0, :, :, 0]

        ar = depth_prediction.numpy()

        factor = 20000.0/np.amax(ar)

        ar = ar * factor

        ar = ar.astype(np.uint32)

        im = Image.fromarray(ar)

        path = os.path.join(pred_path, sfm_image.getId() +
                            "_"+str(factor)+"_.png")

        im.save(path)

        depth_gt = sfm_image.getDepth()

        if depth_gt is not None:

            ar = depth_gt

            factor = 20000.0/np.amax(ar)

            ar = ar * factor

            ar = ar.astype(np.uint32)

            im = Image.fromarray(ar)

            path = os.path.join(
                gt_path, sfm_image.getId() + "_"+str(factor)+"_.png")

            im.save(path)

    def write_z_heatmap(self, path, network):

        sfm_image = self.sfm_images[0]

        folder_dataset = os.path.join(
            path, self.dataset_name)

        if not os.path.exists(folder_dataset):
            os.makedirs(folder_dataset)

        depth_pertb = os.path.join(folder_dataset, "pertb")

        if not os.path.exists(depth_pertb):
            os.makedirs(depth_pertb)

        pertb_img = os.path.join(folder_dataset, "pertb_img")

        if not os.path.exists(pertb_img):
            os.makedirs(pertb_img)

        g = self.photometric_optimizer.get_graphics()

        angle_th = tf.constant(85.0*3.14/180.0, dtype=tf.float32)

        Tinv_batch = tf.stack(self.Tinv)

        P = []
        names = []

        for image in self.sfm_images:
            P = [*P, image.getP_Homogeneous()]
            names = [*names, image.getId()]

        P_batch = tf.stack(P)

        calib_batch = tf.stack(self.calib)
        I_batch = tf.stack(self.I)
        alpha_batch = tf.stack(self.alpha)

        R = g.normalized_points(I_batch, calib_batch)

        IR = tf.concat((I_batch, R), axis=-1)

        z = tf.Variable(self.z[0])

        z_batch = tf.stack([z])

        P,  mu, logvar = network(IR[0:1], z_batch)

        D_batch = network.mapToDepth(tf.ones_like(alpha_batch[0:1]), P)

        depth_optimal = D_batch[0, :, :, 0:1]

        red = Color("#FF0000")
        colors = list(red.range_to(Color("#0000FF"), 192))

        image = np.zeros((192, 256, 3))

        Tinv_batch = Tinv_batch[0:1]
        P_batch = P_batch[0:1]
        calib_batch = calib_batch[0:1]
        alpha_batch = alpha_batch[0:1]
        I_batch = I_batch[0:1]

        for i in range(192):

            pertb = np.zeros((192))

            pertb[i] = 1

            pertb = tf.convert_to_tensor(pertb, dtype=tf.float32)

            z = tf.Variable(self.z[0])

            z.assign(z + 5e-3*pertb)

            z_batch = tf.stack([z])

            P,  mu, logvar = network(IR[0:1], z_batch)

            D_batch = network.mapToDepth(tf.ones_like(alpha_batch), P)

            depth_prediction = D_batch[0, :, :, 0:1]

            diff = tf.abs(depth_optimal - depth_prediction)

            ar = diff.numpy()

            max_val = np.amax(ar)

            ar = ar / max_val

            col = colors[i].get_rgb()

            image = image + (1.0/192.0) * ar * np.array(col).reshape((1, 1, 3))

            D_batch = tf.reshape(alpha_batch[0:1], shape=[
                1, 1, 1, 1]) * D_batch

            alpha = tf.expand_dims(
                tf.convert_to_tensor(ar, dtype=tf.float32), 0)

            color = tf.expand_dims(tf.convert_to_tensor(
                np.array(col).reshape((1, 1, 3)), dtype=tf.float32), 0)

            im_tmp = I_batch * (1.0 - alpha) + alpha*color

            g.to_ply(D_batch, im_tmp, Tinv_batch,
                     P_batch, calib_batch,  angle_th, [name+"_"+str(i) for name in names], depth_pertb, False)

            im_tmp = im_tmp[0].numpy()

            max_val = np.amax(im_tmp)

            im_tmp = im_tmp/max_val

            im = Image.fromarray((im_tmp*255.0).astype(np.uint8))

            path = os.path.join(
                pertb_img, sfm_image.getId() + str(i)+"_.png")

            im.save(path)

        max_val = np.amax(image)

        image = image/max_val

        im = Image.fromarray((image*255.0).astype(np.uint8))

        path = os.path.join(folder_dataset, sfm_image.getId() + "_.png")

        im.save(path)

        depth_optimal = tf.reshape(alpha_batch[0:1], shape=[
                                   1, 1, 1, 1]) * tf.expand_dims(depth_optimal, 0)

        g.to_ply(depth_optimal, tf.expand_dims(tf.convert_to_tensor(image), 0), Tinv_batch,
                 P_batch, calib_batch,  angle_th, names, folder_dataset, False)
