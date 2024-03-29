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

import os
import pickle
import numpy as np
from Sfm.sfm_image import sfm_image
from Sfm.sfm_image import create_sfm_image


class sfm_loader:
    def __init__(self, config, dataset_name=None):
        self.dataset_dir = config["sfm_dataset"]["root_dir"]

        if dataset_name is None:
            dataset_name = config["sfm_dataset"]["datasets"]

        self.dataset_name = dataset_name
        self.shape = (config['dataset']['image_height'],
                      config['dataset']['image_width'], 3)
        self.processed_dir = os.path.join(self.dataset_dir, "processed")
        self.sfm_images = {}
        self.image_dataset_map = {}

    def load(self, loadImages=True):

        datasets = []

        for dataset in self.dataset_name:

            parts = dataset.split("/")

            if parts[-1] == "*":
                path = os.path.join(self.processed_dir, "/".join(parts[0:-1]))

                subfolders = [name for name in os.listdir(
                    path) if os.path.isdir(os.path.join(path, name))]

                for folder in subfolders:
                    datasets.append(os.path.join(
                        "/".join(parts[0:-1]), folder))

            else:

                datasets.append(dataset)

        for dataset in datasets:
            path = os.path.join(self.processed_dir,
                                dataset, "sfm_images.pickle")

            with open(path, 'rb') as stream:
                sfm_image_pickle = pickle.load(stream)

                for key, value in sfm_image_pickle.items():

                    image = create_sfm_image(value)

                    if loadImages:
                        image.load()

                    image.setRoot(self.processed_dir)

                    self.sfm_images.update({key: image})

                    self.image_dataset_map.update({key: dataset})

    def getSFMDataset(self):
        return self.sfm_images

    def getImage(self, im_id):

        if im_id not in self.sfm_images:
            return None

        return self.sfm_images[im_id]

    def getDatasetName(self, im_id):
        return self.image_dataset_map[im_id]
