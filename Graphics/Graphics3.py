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
from tensorflow_addons.image import interpolate_bilinear, dense_image_warp
from tensorflow.python.ops import array_ops
import tensorflow_addons as tfa
# import tensorflow_probability as tfp
import os


class Graphics3:

    @tf.function(experimental_compile=True)
    def generate_homogeneous_points(self, I):

        shape = tf.shape(I)

        y, x = tf.range(0, shape[1]), tf.range(0, shape[2])

        grid = tf.meshgrid(x, y)

        grid = tf.concat(
            [tf.cast(tf.expand_dims(grid[0], 2), dtype=tf.float32),
             tf.cast(tf.expand_dims(grid[1], 2), dtype=tf.float32),
             tf.ones([shape[1], shape[2], 1], dtype=tf.float32)], 2)

        grid = tf.expand_dims(grid, 0)

        return grid

    @tf.function(experimental_compile=True)
    def normalized_points(self, D, calib):

        fx, fy, x0, y0 = tf.split(calib, 4, axis=-1)

        shape = tf.shape(D)

        X = self.generate_homogeneous_points(D)

        CC = tf.concat((x0, y0, tf.zeros_like(x0)),
                       axis=-1)

        C = tf.reshape(CC, [shape[0], 1, 1, 3])

        X = X - C

        CC = tf.concat((fx, fy, tf.ones_like(x0)), axis=-1)

        C = tf.reshape(CC, [shape[0], 1, 1, 3])

        X = tf.math.divide(X, C)

        # X, _ = tf.linalg.normalize(X, axis=-1)

        return X

    @tf.function
    def parallax(self, X, C1, C2, mask):

        # X dim: (w*h,3)
        # C1 dim (1,3)

        d1 = C1 - X
        d2 = C2 - X

        d1, _ = tf.linalg.normalize(d1, axis=-1)
        d2, _ = tf.linalg.normalize(d2, axis=-1)

        cos_angle = tf.math.acos(tf.reduce_sum(d1*d2, axis=-1))*180.0/3.14

        mask_nan = tf.math.is_nan(cos_angle)

        maks = tf.logical_and(tf.squeeze(mask), tf.logical_not(mask_nan))

        masked_angles = tf.boolean_mask(cos_angle, maks)

        mean = tf.reduce_mean(masked_angles)

        return mean

    @tf.function(experimental_compile=True)
    def voxelize(self, X, width, primes):

        X = tf.transpose(X, perm=[0, 2, 1])

        X2 = tf.math.floordiv(X[:, :, 0:3], width)

        index = tf.reduce_sum(X2 * primes, axis=-1, keepdims=True)

        return tf.cast(tf.concat((X2, index), axis=-1), dtype=tf.int32)

    @tf.function  # (experimental_compile=True)
    def voxelize2(self, X, width, primes):

        X = tf.transpose(X, perm=[0, 2, 1])

        X2 = tf.cast(tf.math.floordiv(X[:, :, 0:3], width), dtype=tf.int32)

        min_val_x = tf.reduce_min(X2[:, :, 0:1], axis=1)
        min_val_y = tf.reduce_min(X2[:, :, 1:2], axis=1)
        min_val_z = tf.reduce_min(X2[:, :, 2:], axis=1)

        max_val_x = tf.reduce_max(X2[:, :, 0:1], axis=1)
        max_val_y = tf.reduce_max(X2[:, :, 1:2], axis=1)
        max_val_z = tf.reduce_max(X2[:, :, 2:], axis=1)

        min_point = tf.concat((min_val_x,
                               min_val_y,
                               min_val_z), axis=-1)

        max_point = tf.concat((max_val_x,
                               max_val_y,
                               max_val_z), axis=-1)

        min_point = tf.reshape(min_point, shape=[1, 1, 3])
        max_point = tf.reshape(max_point, shape=[1, 1, 3])

        X2_n = X2 - min_point

        coeffs = max_point - min_point + 1

        coeffs = tf.concat((coeffs[:, :, 1:2]*coeffs[:, :, 2:3],
                            coeffs[:, :, 2:3],
                            tf.ones_like(coeffs[:, :, 0:1])), axis=-1)

        index = tf.reduce_sum(X2_n * coeffs, axis=-1, keepdims=True)
        index = tf.squeeze(index)
        coeffs = tf.squeeze(coeffs)

        # cannot do this batchwise!
        index_uniqe, _ = tf.unique(index)

        # index_uniqe = index

        x = tf.math.floordiv(index_uniqe, coeffs[0:1])
        y = tf.math.floordiv(index_uniqe-x*coeffs[0:1], coeffs[1:2])
        z = index_uniqe-x*coeffs[0:1] - y*coeffs[1:2]

        x = tf.expand_dims(x, axis=-1)
        y = tf.expand_dims(y, axis=-1)
        z = tf.expand_dims(z, axis=-1)

        X2_n_u = tf.reshape(tf.concat((x, y, z), axis=-1),
                            shape=[1, -1, 3]) + min_point

        primes = tf.cast(primes, dtype=tf.int32)

        index = tf.reduce_sum(X2_n_u * primes, axis=-1, keepdims=True)

        #index_test = tf.reduce_sum(X2 * primes, axis=-1, keepdims=True)

        #test = tf.reduce_sum(tf.abs(X2_n_u-X2))

        """
        tf.print(X2_n_u, summarize=-1, output_stream="file://x2_n_u")
        tf.print(X2_n, summarize=-1, output_stream="file://x2")
        """

        return tf.cast(tf.concat((X2_n_u, index), axis=-1), dtype=tf.int32)

    @ tf.function(experimental_compile=True)
    def unproject(self, D, Tinv):

        shape = tf.shape(D)

        X = self.generate_homogeneous_points(D)*D

        X = tf.concat((X, tf.ones(shape)), axis=-1)

        X = tf.reshape(X, shape=[shape[0], shape[1]*shape[2], 4])

        X = tf.transpose(X, perm=[0, 2, 1])

        X = tf.matmul(Tinv, X)

        return X

    @ tf.function(experimental_compile=True)
    def unprojectNormals(self, N, P):

        shape = tf.shape(N)

        X = tf.concat(
            (N, tf.zeros((shape[0], shape[1], shape[2], 1))), axis=-1)

        X = tf.reshape(X, shape=[shape[0], shape[1]*shape[2], 4])

        X = tf.transpose(X, perm=[0, 2, 1])

        X = tf.matmul(tf.linalg.matrix_transpose(P), X)

        X = tf.transpose(X, perm=[0, 2, 1])

        X, _ = tf.linalg.normalize(X[:, :, 0:3], axis=-1)

        return X

    @ tf.function(experimental_compile=True)
    def project(self, X, T):

        shape = tf.shape(X)

        X = tf.expand_dims(X, axis=0)  # (1,batch,4,w*h)
        T = tf.expand_dims(T, axis=1)  # (batch,1,4,4)

        # batch i
        # [Ti*T_1*D1,
        # Ti*T_2*D2,
        # ...
        # Ti*T_N*DN]

        x = tf.matmul(T, X)  # (batch,batch,4,w*h)

        d = x[:, :, 2:3, :]

        x = tf.math.divide_no_nan(x, d)

        return x[:, :, 0:2, :], d

    @ tf.function(experimental_compile=True)
    def interpolate(self, I, x):

        y = tf.transpose(x, perm=[0, 1, 3, 2])

        shape = tf.shape(y)
        shape_I = tf.shape(I)

        y = tf.reshape(y, shape=[shape[0], shape[1]*shape[2], shape[3]])

        I_int = interpolate_bilinear(I, y, indexing='xy')

        I_int = tf.reshape(
            I_int, shape=[shape[0], shape[1], shape[2], shape_I[-1]])

        return I_int

    @ tf.function(experimental_compile=True)
    def transform_to_image(self, I_vec, w, h):
        shape = tf.shape(I_vec)

        I = tf.reshape(I_vec, shape=[shape[0], shape[1], h, w, shape[-1]])

        return I

    @ tf.function(experimental_compile=True)
    def mask_projections(self, x, d, w, h):

        w = tf.cast(w, dtype=tf.float32)
        h = tf.cast(h, dtype=tf.float32)

        mask = tf.greater(d, 0.0)

        xx = x[:, :, 0:1, :]
        yy = x[:, :, 1:2, :]

        mask = tf.logical_and(mask, tf.math.greater_equal(xx, 0.0))
        mask = tf.logical_and(mask, tf.math.greater_equal(yy, 0.0))

        mask = tf.logical_and(mask, tf.math.less_equal(xx, w-1))
        mask = tf.logical_and(mask, tf.math.less_equal(yy, h-1))

        return mask

    @ tf.function(experimental_compile=True)
    def calculate_error_image(self, I, Ir):

        Ie = I - tf.expand_dims(Ir, axis=0)

        return Ie

    @ tf.function(experimental_compile=True)
    def mask_normals(self, D, calib, cos_theta):

        fx, fy, x0, y0 = tf.split(calib, 4, axis=-1)

        D = tfa.image.gaussian_filter2d(D)

        shape = tf.shape(D)

        X = self.generate_homogeneous_points(D)

        CC = tf.concat((x0, y0, tf.zeros_like(x0)),
                       axis=-1)

        C = tf.reshape(CC, [shape[0], 1, 1, 3])

        X = X - C

        CC = tf.concat((fx, fy, tf.ones_like(x0)), axis=-1)

        C = tf.reshape(CC, [shape[0], 1, 1, 3])

        X = tf.math.divide(X, C)

        dy, dx = self.image_gradients(D)

        dx = dx * tf.reshape(fx, [shape[0], 1, 1, 1])

        dy = dy * tf.reshape(fy, [shape[0], 1, 1, 1])

        n = D*tf.concat((-dx, -dy, X[:, :, :, 0:1]*dx +
                         X[:, :, :, 1:2]*dy + D), axis=-1)

        n, _ = tf.linalg.normalize(n, axis=-1)
        X, _ = tf.linalg.normalize(X, axis=-1)

        prod = tf.reduce_sum(n*X, axis=-1, keepdims=True)

        return tf.math.greater(prod, cos_theta), -n

    @ tf.function(experimental_compile=True)
    def warp(self, I, D, T, Tinv):

        shape = tf.shape(I)

        h = shape[1]
        w = shape[2]

        # calculate warpings

        X = self.unproject(D, Tinv)

        x, d = self.project(X, T)

        # calculate masks

        mask = self.mask_projections(x, d, w, h)

        mask = tf.transpose(mask, perm=[0, 1, 3, 2])

        mask = self.transform_to_image(mask, w, h)

        # warp images and depths

        I_int = self.interpolate(I, x)

        I_warp = self.transform_to_image(I_int, w, h)

        D_int = self.interpolate(D, x)

        D_warp = self.transform_to_image(D_int, w, h)

        # transformed depth
        d_projected_warp = tf.transpose(d, perm=[0, 1, 3, 2])

        d_projected_warp = self.transform_to_image(d_projected_warp, w, h)

        return I_warp, D_warp, d_projected_warp, mask

    @ tf.function(experimental_compile=True)
    def calculate_Identity_mask(self, w, h, N):

        mask = tf.linalg.diag(tf.ones((1, N), dtype=tf.float32))

        return tf.reshape(tf.less(mask, 1.0), shape=[N, N, 1, 1, 1])

    @ tf.function(experimental_compile=True)
    def calculate_relative_depth_error(self, D_warp, d_projected_warp, alpha_batch):
        deph_error = D_warp - d_projected_warp

        shape = tf.shape(alpha_batch)

        alpha_batch = tf.reshape(alpha_batch, shape=[shape[0], 1, 1, 1, 1])

        relative_error = tf.math.divide_no_nan(deph_error, alpha_batch)

        return relative_error

    @ tf.function(experimental_compile=True)
    def calculate_occlusion_mask(self, relative_error, mask):

        relative_error = tf.where(
            mask, relative_error, np.inf * tf.ones_like(relative_error))

        median_error = self.calculate_mad(relative_error, mask)

        shape = tf.shape(median_error)

        median_error = tf.reshape(
            median_error, shape=[shape[0], shape[1], 1, 1, 1])

        e_rel = relative_error - median_error

        median_variance = self.calculate_mad(tf.square(e_rel), mask)

        median_variance = tf.reshape(
            median_variance, shape=[shape[0], shape[1], 1, 1, 1])

        mask_g = tf.greater(e_rel, -4.44 * tf.sqrt(median_variance))

        return mask_g

    @ tf.function(experimental_compile=True)
    def calculate_mad(self, V, mask_count):

        mask_count = tf.cast(mask_count, tf.int32)
        mask_count = tf.reduce_sum(mask_count, axis=[2, 3, 4])

        shape = tf.shape(V)

        V = tf.reshape(V, [shape[0], shape[1], shape[2]*shape[3]*shape[4]])

        el = shape[2]*shape[3]*shape[4]

        mid = tf.expand_dims(el - tf.math.floordiv(mask_count, 2) - 1, axis=-1)

        values = tf.nn.top_k(V, el).values

        values_median = tf.gather(values, mid, batch_dims=2)

        return values_median

    @ tf.function(experimental_compile=True)
    def calculate_mad2(self, V):

        shape = tf.shape(V)

        V = tf.reshape(V, [shape[0], shape[1], shape[2]*shape[3]*shape[4]])

        el = shape[2]*shape[3]*shape[4]

        N = tf.math.floordiv(el, 3)  # 10000

        idxs = tf.range(el)
        ridxs = tf.random.shuffle(idxs)[:N]
        rinput = tf.gather(V, ridxs, axis=2)

        no_nan_mask = tf.math.is_finite(rinput)

        mask_count = tf.cast(no_nan_mask, tf.int32)
        mask_count = tf.reduce_sum(mask_count, axis=-1)

        mid = tf.expand_dims(N - tf.math.floordiv(mask_count, 2) - 1, axis=-1)

        values = tf.nn.top_k(rinput, N).values

        values_median = tf.gather(values, mid, batch_dims=2)

        return values_median

    @ tf.function(experimental_compile=True)
    def calculate_masks(self, D, calib, mask, angle):

        mask_d = tf.expand_dims(tf.greater(D, 0.0), axis=0)

        mask_normal, _ = self.mask_normals(D, calib, angle)

        mask_normal = tf.expand_dims(mask_normal, axis=0)

        mask = tf.logical_and(mask, mask_d)
        mask = tf.logical_and(mask, mask_normal)

        return mask

    @ tf.function(experimental_compile=True)
    def Huber(self, error, delta):

        # return tf.square(error)

        abs_error = tf.reduce_sum(tf.math.abs(error), axis=-1)
        quadratic = tf.math.minimum(abs_error, delta)
        linear = tf.math.subtract(abs_error, quadratic)
        return tf.math.add(
            tf.math.multiply(
                tf.convert_to_tensor(0.5, dtype=quadratic.dtype),
                tf.math.multiply(quadratic, quadratic)),
            tf.math.multiply(delta, linear))

    def image_gradients(self, image):

        dy = image[:, 1:, :, :] - image[:, :-1, :, :]
        dx = image[:, :, 1:, :] - image[:, :, :-1, :]

        dy = tf.concat([dy, dy[:, -1:, :, :]], axis=1)

        dx = tf.concat([dx, dx[:, :, -1:, :]], axis=2)

        return dy, dx

    @ tf.function(experimental_compile=True)
    def calculate_image_and_depth_error(self, I_batch, D_batch, T_batch, Tinv_batch, calib_batch, angle, alpha_batch):

        I_warp, D_warp, d_projected_warp, mask = self.warp(I_batch,
                                                           D_batch,
                                                           T_batch,
                                                           Tinv_batch)

        relative_depth_error = self.calculate_relative_depth_error(
            D_warp, d_projected_warp, alpha_batch)

        mask = self.calculate_masks(D_batch, calib_batch, mask, angle)

        shape_d = tf.shape(D_batch)

        mask_I = self.calculate_Identity_mask(
            shape_d[2], shape_d[1], shape_d[0])

        mask = tf.logical_and(mask, mask_I)

        mask_g = self.calculate_occlusion_mask(relative_depth_error, mask)

        mask = tf.logical_and(mask, mask_g)

        relative_depth_error = tf.where(
            mask, relative_depth_error, tf.zeros_like(relative_depth_error))

        mask_count = tf.cast(mask, tf.float32)
        mask_count = tf.reduce_sum(mask_count)

        """
        error_depth = tf.reduce_sum(self.Huber(
            relative_depth_error, 0.1))/mask_count

        """

        error_depth = tf.reduce_sum(self.Huber(
            relative_depth_error, 0.1))/0.1

        mask = tf.logical_or(mask, tf.logical_not(mask_I))

        I_warp = tf.where(
            mask, I_warp, tf.zeros_like(I_warp))

        return error_depth, I_warp, mask

    @ tf.function(experimental_compile=True)
    def calculate_overlap(self, I_batch, D_batch, T_batch, Tinv_batch, calib_batch, angle, alpha_batch):

        I_warp, D_warp, d_projected_warp, mask = self.warp(I_batch,
                                                           D_batch,
                                                           T_batch,
                                                           Tinv_batch)

        relative_depth_error = self.calculate_relative_depth_error(
            D_warp, d_projected_warp, alpha_batch)

        mask = self.calculate_masks(D_batch, calib_batch, mask, angle)

        shape_d = tf.shape(D_batch)

        mask_I = self.calculate_Identity_mask(
            shape_d[2], shape_d[1], shape_d[0])

        mask = tf.logical_and(mask, mask_I)

        mask_g = self.calculate_occlusion_mask(relative_depth_error, mask)

        mask = tf.logical_and(mask, mask_g)

        return mask

    @ tf.function(experimental_compile=True)
    def calculate_error_image_depth(self, I_batch, D_batch, T_batch, Tinv_batch, calib_batch, angle, alpha_batch):

        I_warp, D_warp, d_projected_warp, mask = self.warp(I_batch,
                                                           D_batch,
                                                           T_batch,
                                                           Tinv_batch)

        relative_depth_error = self.calculate_relative_depth_error(
            D_warp, d_projected_warp, alpha_batch)

        error_image = self.calculate_error_image(I_warp, I_batch)

        mask = self.calculate_masks(D_batch, calib_batch, mask, angle)

        shape_d = tf.shape(D_batch)

        mask_I = self.calculate_Identity_mask(
            shape_d[2], shape_d[1], shape_d[0])

        mask = tf.logical_and(mask, mask_I)

        mask_g = self.calculate_occlusion_mask(relative_depth_error, mask)

        mask = tf.logical_and(mask, mask_g)

        return relative_depth_error, error_image, mask

    @ tf.function(experimental_compile=True)
    def evaluate_photogeometric_error(self, I_batch, D_batch, T_batch, Tinv_batch, calib_batch, angle, alpha_batch):

        relative_depth_error, error_image, mask = self.calculate_error_image_depth(I_batch,
                                                                                   D_batch,
                                                                                   T_batch,
                                                                                   Tinv_batch,
                                                                                   calib_batch,
                                                                                   angle, alpha_batch)

        error_image = tf.where(
            mask, error_image, tf.zeros_like(error_image))

        relative_depth_error = tf.where(
            mask, relative_depth_error, tf.zeros_like(relative_depth_error))

        """
        mask_count = tf.cast(mask, tf.float32)
        mask_count = tf.reduce_sum(mask_count)

        error_photometric = tf.reduce_sum(
            self.Huber(error_image, 0.1))/mask_count

        error_depth = tf.reduce_sum(self.Huber(
            relative_depth_error, 0.1))/mask_count
        """

        error_photometric = tf.reduce_sum(
            self.Huber(error_image, 0.1))/0.01

        error_depth = tf.reduce_sum(self.Huber(
            relative_depth_error, 0.1))/0.1

        #error_depth = tf.constant(0.0, dtype=tf.float32)

        return error_photometric, error_depth

    @tf.function(experimental_compile=True)
    def log_normal_loss(self, x, mu, logvar):
        return tf.reduce_sum(0.5*(tf.square(x-mu)*tf.math.exp(-logvar) + logvar)+tf.math.log(2*np.pi))

    @tf.function(experimental_compile=True)
    def log_normal_loss2(self, x, mu, logvar):
        return 0.5*(tf.square(x-mu)*tf.math.exp(-logvar) + logvar)+tf.math.log(2*np.pi)

    @tf.function(experimental_compile=True)
    def D_kl(self, mu, logvar):
        return 0.5*tf.reduce_sum(tf.math.exp(logvar) + tf.square(mu) - 1.0 - logvar)

    def to_ply(self, D, I, Tinv, P, calib, angle, names, path, mask_normals=True):

        shape = tf.shape(D)

        X = self.unproject(D, Tinv)

        X = tf.transpose(X[:, 0:3, :], perm=[0, 2, 1])

        mask, N = self.mask_normals(D, calib, angle)

        mask = tf.reshape(mask, shape=[shape[0], shape[1]*shape[2]])

        if not mask_normals:
            mask = tf.ones_like(mask)

        I = tf.reshape(I, shape=[shape[0], shape[1]*shape[2], 3])

        N = self.unprojectNormals(N, P)

        for i in range(shape[0]):

            X_i = tf.boolean_mask(X[i], mask[i])
            I_i = tf.boolean_mask(I[i], mask[i])
            N_i = tf.boolean_mask(N[i], mask[i])

            self.write_ply_with_normals(I_i, X_i, N_i, names[i], path)

    def write_ply(self, img, points_3d, name, path):
        """
        Function writes a ply file, based on the inputs.

        Input: img (RGB)
            depth (1 channel depth map)
            i (index for ply-file)
            flag: indicator is prediction or gt (bool)
        Output: None
        """

        points_3d_np = points_3d.numpy()
        img_np = (img*255.0).numpy().astype(np.uint8)

        all_ = np.concatenate((points_3d_np, img_np), axis=-1)

        n_elements, _ = all_.shape

        file_added = open(os.path.join(path, name + ".ply"), "w")

        str_first = "ply \nformat ascii 1.0"
        str_add_el = "\nelement vertex "+str(n_elements)
        str_pos = "\nproperty float32 x \nproperty float32 y \nproperty float32 z"
        str_col = "\nproperty uchar red \nproperty uchar green \nproperty uchar blue"
        str_last = "\nend_header\n"

        STR_FULL = str_first+str_add_el+str_pos+str_col+str_last
        file_added.write(STR_FULL)

        np.savetxt(file_added, all_, fmt="%8.5g",
                   delimiter=' ')   # X is an array

        file_added.close()

    def write_ply_with_normals(self, img, points_3d, normals, name, path):
        """
        Function writes a ply file, based on the inputs.

        Input: img (RGB)
            depth (1 channel depth map)
            i (index for ply-file)
            flag: indicator is prediction or gt (bool)
        Output: None
        """

        points_3d_np = points_3d.numpy()
        normals_np = normals.numpy()
        img_np = (img*255.0).numpy().astype(np.uint8)

        all_ = np.concatenate((points_3d_np, normals_np, img_np), axis=-1)

        n_elements, _ = all_.shape

        file_added = open(os.path.join(path, name + ".ply"), "w")

        str_first = "ply \nformat ascii 1.0"
        str_add_el = "\nelement vertex "+str(n_elements)
        str_pos = "\nproperty float32 x \nproperty float32 y \nproperty float32 z"
        str_pos_norm = "\nproperty float32 nx \nproperty float32 ny \nproperty float32 nz"
        str_col = "\nproperty uchar red \nproperty uchar green \nproperty uchar blue"
        str_last = "\nend_header\n"

        STR_FULL = str_first+str_add_el+str_pos + str_pos_norm+str_col+str_last
        file_added.write(STR_FULL)

        np.savetxt(file_added, all_, fmt="%8.5g",
                   delimiter=' ')   # X is an array

        file_added.close()
