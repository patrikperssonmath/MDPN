from tensorflow.keras.optimizers import Adam, Adamax
import tensorflow as tf
import numpy as np
from tensorflow_addons.image import interpolate_bilinear
from Graphics.Graphics3 import Graphics3


class InferePhotometric:
    def __init__(self, config):

        self.z = None
        self.alpha = None

        self.g = Graphics3()

        self.angle_th = np.cos(
            (config['PhotometricOptimizer']['angle_th']/180.0) * np.pi)

        self.angle_th = tf.constant(self.angle_th, dtype=tf.float32)

        self.optimizer = Adamax(lr=1e-3)
        self.iterations = 100  # config["InferePhotometric"]["iterations"]
        self.termination_crit = config["PhotometricOptimizer"]["termination_crit"]

    @tf.function
    def infere(self, I, T, Tinv, calibration, z_in, alpha_in, network):

        R = self.g.normalized_points(I, calibration)

        IR = tf.concat((I, R), axis=-1)

        shape = tf.shape(I)

        if self.z is None:
            self.z = tf.Variable(z_in, dtype=tf.float32)

            self.alpha = tf.Variable(alpha_in, dtype=tf.float32)
        else:
            self.z.assign(z_in)

            self.alpha.assign(alpha_in)

        trainable_variables = [self.z,
                               self.alpha]

        error = tf.constant(0.0, dtype=tf.float32)
        error_prev = tf.constant(0.0, dtype=tf.float32)

        q, _, _ = network.encode(IR)

        for i in tf.range(self.iterations):

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(trainable_variables)

                P = network.decode(q, self.z)

                D = network.mapToDepth(self.alpha, P)

                error_photometric, error_depth = self.g.evaluate_photogeometric_error(
                    I,
                    D,
                    T,
                    Tinv,
                    calibration,
                    self.angle_th,
                    self.alpha)

                error = error_photometric + error_depth

                error += tf.reduce_sum(tf.square(tf.reduce_mean(network.mapToDepth(
                    tf.ones_like(self.alpha), P), axis=[1, 2, 3]) - 1.0))

                error += tf.reduce_sum(tf.square(self.z))

                #tf.print(i, error)

            gradients = tape.gradient(error, trainable_variables)

            self.optimizer.apply_gradients(
                zip(gradients, trainable_variables))

            if i > 0 and (tf.math.abs(error-error_prev) / tf.math.abs(error)) < self.termination_crit:
                # tf.print("breaking")
                break

            error_prev = error

        return self.z, self.alpha, error, i+1