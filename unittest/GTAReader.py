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

import numpy as np
import unittest
import tensorflow as tf
import json
from PIL import Image
# Own packages
from Sfm.sfm_image import sfm_image
import OpenEXR

import Imath
import uuid

import argparse
import yaml


def get_exr_rgb(path):
    I = OpenEXR.InputFile(path)
    dw = I.header()['displayWindow']
    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    data = [np.frombuffer(c, np.float16).reshape(size)
            for c in I.channels('Y')]
    return (np.array(data[0])).astype(np.float32)


class GTAReader:

    def __init__(self, path):
        self.path = path
        self.i = 0
        self.origo = []

    def next(self):
        param_path = self.path + "/poses/"+str(self.i).rjust(4, '0')+".json"
        image_path = self.path + "/images/"+str(self.i).rjust(4, '0')+".png"
        depth_path = self.path + "/depths/"+str(self.i).rjust(4, '0')+".exr"

        with open(param_path, 'r') as f:
            distros_dict = json.load(f)

        image = Image.open(image_path)
        image = np.array(image)

        # get_exr_rgb(depth_path)  # Image.open(depth_path)

        shape = image.shape
        # depth = np.ones((shape[0], shape[1]), dtype=np.float32)
        depth = get_exr_rgb(depth_path)
        depth = np.array(depth)

        K = np.array([[distros_dict.get("f_x"), 0.0, distros_dict.get("c_x")],
                      [0.0, distros_dict.get(
                          "f_y"), distros_dict.get("c_y")],
                      [0.0, 0.0, 1.0]]).astype(np.float32)

        P = np.array(distros_dict.get("extrinsic"), dtype=np.float32)
        # P = np.identity(4, dtype=np.float32)

        if self.i == 0:
            self.origo = np.linalg.inv(P)

        P = np.dot(P, self.origo)

        id = str(uuid.uuid4())

        im_s = sfm_image(image_path, [], P[0:3, 0:4], K, id, np.array([0]))
        im_s.load()

        self.i = self.i+1

        return im_s, depth
