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

import threading
import time
import queue
from PIL import Image
from DataLoaderTask.DataExample import DataExample
from skimage.transform import resize
import csv
import numpy as np
import os


class DataLoaderTask (threading.Thread):
    def __init__(self, config, queue):
        threading.Thread.__init__(self)
        self.q = queue
        self.datafolder = config['rgbd_dataset']['root_dir']
        self.dimensions = (config['dataset']['image_width'],
                           config['dataset']['image_height'])

        subfolders = [name for name in os.listdir(
            self.datafolder) if os.path.isdir(os.path.join(self.datafolder, name))]

        self.x_data = []
        self.y_data = []
        self.mask = []
        self.calibration = []

        for folder in subfolders:
            path = os.path.join(self.datafolder, folder)

            subsubfolders = [name for name in os.listdir(
                path) if os.path.isdir(os.path.join(path, name))]

            for dataset in subsubfolders:

                path_dataset = os.path.join(path, dataset, "data.csv")

                with open(path_dataset, newline='') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')
                    for row in reader:
                        """
                        self.x_data.append(self.datafolder+'/'+row[0])
                        self.y_data.append(self.datafolder+'/'+row[1])
                        self.mask.append(self.datafolder+'/'+row[2])
                        self.calibration.append(self.datafolder+'/'+row[3])
                        """

                        self.x_data.append(row[0])
                        self.y_data.append(row[1])
                        self.mask.append(row[2])
                        self.calibration.append(row[3])

        self.exit_flag = False
        self.queueLock = threading.Lock()

    def getIds(self):

        ids = []

        for path in self.x_data:
            ids.append(path.replace("/", "_"))

        return ids

    def getNbrSamples(self):
        return len(self.x_data)

    def setExit(self):
        self.queueLock.acquire()
        self.exit_flag = True
        self.queueLock.release()

    def run(self):

        if len(self.x_data) == 0:
            return

        while True:

            indices = np.random.randint(0, len(self.x_data))

            x_path = self.x_data[indices].replace("/", "_")

            im = Image.open(self.x_data[indices])

            sx = self.dimensions[0]/im.width
            sy = self.dimensions[1]/im.height

            image = np.array(im.resize(self.dimensions)
                             ).astype(np.float32)
            image /= 255.

            depth = np.array(Image.open(self.y_data[indices])).astype(
                np.float32)/1000.0

            mask = (depth != 0).astype(np.float32)

            depth_r = resize(depth, (self.dimensions[1], self.dimensions[0]))
            mask_r = resize(mask, (self.dimensions[1], self.dimensions[0]))

            mask_r = mask_r == 1

            mask_r = mask_r.astype(np.float32)

            depth_r = depth_r*mask_r

            depth = depth_r

            mask = mask_r.astype(np.bool)

            with open(self.calibration[indices], newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    # division by two due to rescaling
                    K = np.array([[float(row[0])*sx, 0.0, float(row[2])*sx],
                                  [0.0, float(row[1])*sy, float(row[3])*sy],
                                  [0.0, 0.0, 1.0]], dtype=np.float32)

            while True:

                self.queueLock.acquire()
                if self.exit_flag:
                    self.queueLock.release()
                    return True
                self.queueLock.release()

                try:
                    self.q.put(DataExample(x_path, image, depth, mask, K),
                               block=True, timeout=1)
                    break

                except queue.Full:
                    # check exit flag
                    i = 0
