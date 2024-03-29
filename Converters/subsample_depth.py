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
import pickle
import argparse
import yaml
import csv


class subsample_depth:
    def __init__(self, config, output_folder, N_samples):
        self.datafolder = config['rgbd_dataset']['root_dir']
        self.dimensions = (config['dataset']['image_width'],
                           config['dataset']['image_height'])

        self.N_samples = N_samples

        subfolders = [name for name in os.listdir(
            self.datafolder) if os.path.isdir(os.path.join(self.datafolder, name))]

        self.x_data = []
        self.y_data = []
        self.mask = []
        self.calibration = []

        self.output_folder = output_folder

        for folder in subfolders:
            path = os.path.join(self.datafolder, folder)

            subsubfolders = [name for name in os.listdir(
                path) if os.path.isdir(os.path.join(path, name))]

            for dataset in subsubfolders:

                path_dataset = os.path.join(path, dataset, "data.csv")

                with open(path_dataset, newline='') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')
                    for row in reader:

                        self.x_data.append(row[0])
                        self.y_data.append(row[1])
                        self.mask.append(row[2])
                        self.calibration.append(row[3])

        self.indexes = [i for i in range(0, len(self.x_data))]

        random.shuffle(self.indexes)

    def process(self):

        path_dataset = os.path.join(self.output_folder, "data.csv")

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        f = open(path_dataset, "w")

        for i in range(0, min([len(self.x_data), self.N_samples])):

            f.write("{0},{1},{2},{3}\n".format(
                self.x_data[self.indexes[i]],
                self.y_data[self.indexes[i]],
                self.mask[self.indexes[i]],
                self.calibration[self.indexes[i]]))

        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('config', type=str)
    parser.add_argument('--output', '--o', default="/database/depth_subsample/subsample/subsample",
                        type=str, required=False)
    parser.add_argument('--samples', '--s', default=150000,
                        type=str, required=False)

    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)

            subsample_depth(config, args.output, int(args.samples)).process()

        except yaml.YAMLError as exc:
            print(exc)
