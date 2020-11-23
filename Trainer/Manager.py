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

from Optimizers.DepthOptimizer import DepthOptimizer
from Optimizers.PhotometricOptimizer2 import PhotometricOptimizer2 as PhotometricOptimizer
from BatchLoaders.DepthBatchLoader import DepthBatchLoader
from BatchLoaders.SFMBatchLoader2 import SFMBatchLoader2 as SFMBatchLoader
import signal
import sys
import importlib
import tensorflow as tf
import os
import shutil
from Trainer.Trainer import Trainer
from tensorboard import program
import numpy as np
from tensorflow.keras.optimizers import Adam, Adamax
import time
from datetime import datetime
import sys


class Manager:
    def __init__(self, config):
        self.config = config

        self.log_dir = config["model"]["log_dir"]

        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)

        os.makedirs(self.log_dir)

        self.ckpt_freq = config["Trainer"]["ckpt_freq"]
        self.write_dataset_freq = config["Trainer"]["write_dataset_freq"]
        self.max_dataset_write = config["Trainer"]["max_dataset_write"]
        self.write_tb_freq = config["Trainer"]["write_tb_freq"]

        self.network_root = config['model']['root_dir']
        self.data_root = config['dataset']['root_dir']

        self.epochs = config['Manager']['epochs']
        self.train_photometric_or_depth = config['Manager']['train_photometric_or_depth']
        self.write_predictions_only = config['Manager']['write_predictions_only']

        self.network_name = config['model']['network']

        my_module = importlib.import_module("Networks."+self.network_name)

        MyClass = getattr(my_module, self.network_name)

        self.network = MyClass(config)

        self.depth_optimizer = DepthOptimizer(config)
        self.depth_batch_loader = DepthBatchLoader(
            config, self.depth_optimizer)

        self.photo_optimizer = PhotometricOptimizer(config)
        self.photometric_batch_loader = SFMBatchLoader(
            config, self.photo_optimizer)

        self.z_variables = {}
        self.alpha_variables = {}

        signal.signal(signal.SIGINT, lambda signal,
                      frame: self._signal_handler())

        self.terminated = False
        self.trainer = Trainer(self.config)

        self.predict_only = config['Trainer']['predict_only']

        self.write_result = config["Manager"]["write_result"]
        self.write_heatmap_flag = config["Manager"]["write_heatmap_flag"]

        self.terminate_hard = False

        self.angle_th = np.cos(
            (config['PhotometricOptimizer']['angle_th']/180.0) * np.pi)

        self.angle_th_display = np.cos(
            (config['PhotometricOptimizer']['angle_th_display']/180.0) * np.pi)

    def _signal_handler(self):
        self.terminate()

    def terminate(self):

        if self.terminated:
            self.terminate_hard = True

        self.terminated = True
        # self.depth_batch_loader.terminate()
        # self.photometric_batch_loader.terminate()

    def initialize_z_variables(self):

        if self.train_photometric_or_depth:

            self.photometric_batch_loader.setup(
                self.z_variables, self.alpha_variables)

        else:

            self.depth_batch_loader.setup(
                self.z_variables, self.alpha_variables)

    def setup_checkpoint(self):
        self.root = tf.train.Checkpoint(**self.depth_optimizer.getCheckPointVariables(),
                                        **self.photo_optimizer.getCheckPointVariables(),
                                        **self.z_variables,
                                        **self.alpha_variables,
                                        encoder=self.network.getEncoder(),
                                        decoder=self.network.getDecoder())

        model_dir = os.path.join(
            self.network_root, "models", self.network.getName())

        if self.train_photometric_or_depth:
            model_dir = os.path.join(model_dir, "photometric")
        else:
            model_dir = os.path.join(model_dir, "depth")

        checkpoint_dir = model_dir

        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

        self.model_folder = os.path.join(model_dir, "networks")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        else:
            self.root.restore(tf.train.latest_checkpoint(checkpoint_dir))

        if not os.path.exists(self.model_folder):

            if self.predict_only:
                print("network could not be loaded!")
                return False
            else:
                os.makedirs(self.model_folder)
        else:
            if self.predict_only:

                if not self.network.load(self.model_folder):
                    print("network could not be loaded!")
                    return False

        return True

    def startTensorboard(self):

        self.prof_log_dir = os.path.join(self.log_dir, "profiler")

        if os.path.exists(self.prof_log_dir):
            shutil.rmtree(self.prof_log_dir)

        self.tb = program.TensorBoard()
        self.tb.configure(argv=[None, '--logdir', self.log_dir])
        self.url = self.tb.launch()

        print('\nStarted tensorboard at: {0}'.format(self.url))

    def run(self):

        self.initialize_z_variables()

        if not self.setup_checkpoint():
            self.depth_batch_loader.terminate()
            self.photometric_batch_loader.terminate()
            self.terminate()
            return False

        self.startTensorboard()

        if self.write_predictions_only:
            #self.writeDataset(self.photometric_batch_loader, 0)

            if self.write_heatmap_flag:
                self.write_heatmap()

            print("\nshutting down!")

            self.depth_batch_loader.terminate()
            self.photometric_batch_loader.terminate()
            self.terminate()

            return

        for epoch in range(self.epochs):

            recon_loss_avg = {self.photometric_batch_loader.getName(): [],
                              self.depth_batch_loader.getName(): []}

            loss_avg = {self.photometric_batch_loader.getName(): [],
                        self.depth_batch_loader.getName(): []}

            nbr_samples = self.calculate_epoch_samples(epoch)

            progbar = tf.keras.utils.Progbar(nbr_samples)

            if self.terminated:
                break

            for j in range(nbr_samples):

                if self.terminated:
                    break

                batch_loader = self.drawBatchLoader(epoch)

                sample, on_epoch = batch_loader.getNext(
                    self.z_variables, self.alpha_variables)

                loss, recon_loss = self.trainer.run(
                    sample, self.network)

                prog_list = []

                if recon_loss > -float('Inf'):
                    recon_loss_avg[batch_loader.getName()].append(
                        recon_loss.numpy())

                    prog_list.append(
                        (batch_loader.getName()+"_recon_loss", recon_loss))

                if loss > -float('Inf'):
                    loss_avg[batch_loader.getName()].append(loss.numpy())

                    prog_list.append((batch_loader.getName()+"_loss", loss))

                progbar.update(
                    j+1, prog_list)

            if self.train_photometric_or_depth:
                self.writeLoss(loss_avg, recon_loss_avg,
                               self.photometric_batch_loader, epoch)

            else:
                self.writeLoss(loss_avg, recon_loss_avg,
                               self.depth_batch_loader, epoch)

            print('\nEpoch: {0} of {1}'.format(
                str(epoch), str(self.epochs)))

            if(epoch % self.ckpt_freq == 0) or self.terminated:
                print("storing network")

                self.root.save(self.checkpoint_prefix)
                self.network.save(self.model_folder)

            if(epoch % self.write_dataset_freq == 0) or self.terminated:

                if self.train_photometric_or_depth:
                    self.writeDataset(self.photometric_batch_loader, epoch)
                else:
                    self.writeDataset(self.depth_batch_loader, epoch)

            if(epoch % self.write_tb_freq == 0) or self.terminated:

                if self.train_photometric_or_depth:

                    self.writeTensorboard(self.photometric_batch_loader, epoch)

                else:

                    self.writeTensorboard(self.depth_batch_loader, epoch)

            if self.predict_only:
                self.terminated = True

        if self.write_result:
            self.write_results()

        if self.write_heatmap_flag:
            self.write_heatmap()

        print("\nshutting down!")

        self.depth_batch_loader.terminate()
        self.photometric_batch_loader.terminate()
        self.terminate()

        return True

    def write_results(self):

        path = "/data/results"

        timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

        path = os.path.join(path, timestamp)

        self.photometric_batch_loader.reset()

        N = self.photometric_batch_loader.getNbrBatches()

        if os.path.exists(path):
            shutil.rmtree(path)

        for i in range(N):

            sample, on_epoch = self.photometric_batch_loader.getNext(
                self.z_variables, self.alpha_variables)

            if sample is not None:
                sample.write_depth(path, self.network)

            print('\n sample: {0} of {1}'.format(
                str(i), str(N)))

            if self.terminate_hard:
                break

    def write_heatmap(self):

        path = "/data/heatmap"

        timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

        path = os.path.join(path, timestamp)

        self.photometric_batch_loader.reset()

        N = self.photometric_batch_loader.getNbrBatches()

        if os.path.exists(path):
            shutil.rmtree(path)

        for i in range(N):

            sample, on_epoch = self.photometric_batch_loader.getNext(
                self.z_variables, self.alpha_variables)

            if sample is not None:
                sample.write_z_heatmap(path, self.network)

            print('\n sample: {0} of {1}'.format(
                str(i), str(N)))

            if self.terminate_hard:
                break

    def calculate_epoch_samples(self, epoch):

        if self.train_photometric_or_depth:

            nbr_samples = self.photometric_batch_loader.getNbrBatches()

        else:

            nbr_samples = self.depth_batch_loader.getNbrBatches()

        return nbr_samples

    def writeLoss(self, loss_avg, recon_loss_avg, batch_loader, epoch):

        writer = batch_loader.getSummaryWriter()

        with writer.as_default():

            if len(loss_avg[batch_loader.getName()]) > 0:
                tf.summary.scalar('loss', np.mean(
                    loss_avg[batch_loader.getName()]), step=epoch)

            if len(recon_loss_avg[batch_loader.getName()]) > 0:
                tf.summary.scalar('recon_loss', np.mean(
                    recon_loss_avg[batch_loader.getName()]), step=epoch)

    def writeTensorboard(self, batch_loader, epoch):
        print("writing to tensorboard")

        writer = batch_loader.getSummaryWriter()

        for t in range(1):
            sample, on_epoch = batch_loader.getNext(
                self.z_variables, self.alpha_variables)

            if sample:

                with writer.as_default():

                    sample.writeTensorboard(self.network, epoch)

        batch_loader.reset()

    def drawBatchLoader(self, epoch):

        if self.train_photometric_or_depth:

            batch_loader = self.photometric_batch_loader

        else:

            batch_loader = self.depth_batch_loader

        return batch_loader

    def writeDataset(self, batch_loader, epoch):

        print("writing data")

        folder = os.path.join(
            self.data_root, "predictions", batch_loader.getName(), str(epoch))

        if os.path.exists(folder):
            shutil.rmtree(folder)

        os.makedirs(folder)

        written_nbr_samples = 0
        for t in range(batch_loader.getNbrBatches()):
            sample, on_epoch = batch_loader.getNext(
                self.z_variables, self.alpha_variables)

            if sample:
                written_nbr_samples += sample.write(
                    folder, self.network, self.max_dataset_write - written_nbr_samples, self.angle_th_display)

            if written_nbr_samples >= self.max_dataset_write:
                break

            if self.terminate_hard:
                break

        batch_loader.reset()
