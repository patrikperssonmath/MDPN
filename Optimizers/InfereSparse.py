from tensorflow.keras.optimizers import Adam, Adamax
import tensorflow as tf
import numpy as np
from tensorflow_addons.image import interpolate_bilinear
from Graphics.Graphics3 import Graphics3


class InfereSparse:
    def __init__(self, config):

        self.z = None
        self.alpha = None

        self.g = Graphics3()

        self.optimizer = Adamax(lr=1e-3)
        self.iterations = 100  # config["Inference"]["iterations"]
        self.termination_crit = config["PhotometricOptimizer"]["termination_crit"]

    @tf.function
    def infere(self, I, calibration, z_in, alpha_in, u, u_mask, network):

        R = self.g.normalized_points(I, calibration)

        IR = tf.concat((I, R), axis=-1)

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

                xy = u[:, 0:2]

                D_int = interpolate_bilinear(
                    D, tf.transpose(xy, perm=[0, 2, 1]), indexing='xy')

                d_sparse = tf.transpose(u[:, 2:3], perm=[0, 2, 1])

                e = self.g.Huber(D_int - d_sparse, 0.1)

                e = tf.boolean_mask(e, u_mask)

                error = tf.reduce_sum(e)

                error += tf.reduce_sum(tf.square(tf.reduce_mean(network.mapToDepth(
                    tf.ones_like(self.alpha), P), axis=[1, 2, 3]) - 1.0))

                error += tf.reduce_sum(tf.square(self.z))

                # tf.print(i, error)

            gradients = tape.gradient(error, trainable_variables)

            self.optimizer.apply_gradients(
                zip(gradients, trainable_variables))

            if i > 0 and (tf.math.abs(error-error_prev) / tf.math.abs(error)) < self.termination_crit:
                    # tf.print("breaking")
                    break

            error_prev = error

        return self.z, self.alpha, error, i+1
