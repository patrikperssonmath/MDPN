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

from tensorflow.keras.optimizers import Adam, Adamax
import tensorflow as tf


class Trainer:
    def __init__(self, config):
        self.predict_only = config['Trainer']['predict_only']
        self.nbr_samples_per_update = config['Trainer']['nbr_samples_per_update']
        self.optimizer = Adamax(lr=1e-3)

        self.buffered_samples = []

    def run(self, sample, network):

        loss = -tf.constant(float('Inf'), dtype=tf.float32)
        recon_loss = -tf.constant(float('Inf'), dtype=tf.float32)

        if sample is not None:

            if self.predict_only:

                loss = sample.predict(network)

                return loss, recon_loss

            else:

                self.buffered_samples.append(sample)

                if len(self.buffered_samples) >= self.nbr_samples_per_update:

                    trainable_variables = network.getTrainableVariables()

                    for sample in self.buffered_samples:

                        alpha = sample.getAlpha()

                        trainable_variables = [
                            *trainable_variables, *alpha]

                        z = sample.getZ()

                        trainable_variables = [
                            *trainable_variables, *z]

                    with tf.GradientTape(watch_accessed_variables=False) as tape:
                        tape.watch(trainable_variables)

                        loss = tf.constant(0.0, dtype=tf.float32)
                        recon_loss = tf.constant(0.0, dtype=tf.float32)

                        for sample in self.buffered_samples:

                            recon_loss_i, loss_i = sample.train(
                                network)

                            loss += loss_i
                            recon_loss += recon_loss_i

                        gradients = tape.gradient(
                            loss, trainable_variables)

                    self.optimizer.apply_gradients(
                        zip(gradients, trainable_variables))

                    loss /= len(self.buffered_samples)
                    recon_loss /= len(self.buffered_samples)

                    self.buffered_samples.clear()

        return loss, recon_loss
