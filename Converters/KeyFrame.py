from Sfm.sfm_image import sfm_image
import tensorflow as tf
import numpy as np


class KeyFrame:
    def __init__(self, sfm_image):
        self.image = sfm_image

        self.tf_image_i, self.tf_image_depth_i, self.T_i, self.Tinv_i, self.calib_i, self.C_i = self.convert_to_tensor(
            sfm_image)

    def getId(self):
        return self.image.getId()

    def rel_distance(self, sfm_image_j, factor):

        dist = self.image.distance(sfm_image_j)

        d_mean = self.image.getMeanDepth()

        return dist > factor*d_mean

    def compare(self, sfm_image_j, g, angle):
        tf_image_j, tf_image_depth_j, T_j, Tinv_j, calib_j, C_j = self.convert_to_tensor(
            sfm_image_j)

        I_batch = tf.stack([self.tf_image_i, tf_image_j])
        D_batch = tf.stack([self.tf_image_depth_i, tf_image_depth_j])
        T_batch = tf.stack([self.T_i, T_j])
        Tinv_batch = tf.stack([self.Tinv_i, Tinv_j])
        calib_batch = tf.stack([self.calib_i, calib_j])

        alpha_batch =  tf.reduce_mean(D_batch, axis=[1, 2, 3])

        mask = g.calculate_overlap(
            I_batch, D_batch, T_batch, Tinv_batch, calib_batch, angle, alpha_batch)

        shape = tf.shape(D_batch)

        X = g.unproject(D_batch, Tinv_batch)

        X = tf.transpose(X, perm=[0, 2, 1])

        X = tf.reshape(X[:, :, 0:3], shape=[shape[0], shape[1], shape[2], 3])

        parallax_ij = g.parallax(X[0], self.C_i, C_j, mask[1, 0])

        parallax_ji = g.parallax(X[1], self.C_i, C_j, mask[0, 1])

        mask = tf.cast(mask, dtype=tf.float32)

        overlap_ji = tf.reduce_mean(mask[0, 1])
        overlap_ij = tf.reduce_mean(mask[1, 0])

        return np.min([overlap_ji.numpy(), overlap_ij.numpy()]), np.min([parallax_ji.numpy(), parallax_ij.numpy()])

    def convert_to_tensor(self, image):

        tf_image = tf.convert_to_tensor(
            image.getImage(), dtype=tf.float32)
        tf_image_depth = tf.convert_to_tensor(
            image.getDepth(), dtype=tf.float32)

        tf_image_depth = tf.expand_dims(tf_image_depth, axis=-1)

        T, Tinv = image.getTransformations()

        T = tf.convert_to_tensor(T, dtype=tf.float32)
        Tinv = tf.convert_to_tensor(Tinv, dtype=tf.float32)

        calib = tf.convert_to_tensor(
            image.getCalibVec(), dtype=tf.float32)

        C = tf.convert_to_tensor(
            np.transpose(image.getCameraCenter()), dtype=tf.float32)

        return tf_image, tf_image_depth, T, Tinv, calib, C

    def unproject(self, g):

        X = g.unproject(tf.expand_dims(self.tf_image_depth_i, axis=0),
                        tf.expand_dims(self.Tinv_i, axis=0))

        return X, [self.image.getId()]
