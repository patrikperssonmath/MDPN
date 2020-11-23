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
from skimage.io import imread, imsave
from skimage.transform import resize
import os
import tensorflow as tf
from PIL import Image


def create_sfm_image(dict):
    return sfm_image(dict["im_name"], dict["covisible"], dict["P"], dict["K"], dict["uuid"], dict["sparse_depth"])


class sfm_image:

    def __init__(self, im_name, covisible, P, K, uuid, sparse_depth):
        self.im_name = im_name
        self.covisible = covisible
        self.P = P.astype(np.float32)
        self.K = K.astype(np.float32)
        self.uuid = uuid
        self.sparse_depth = None
        self.depth_mean = None

        if sparse_depth is not None:
            self.sparse_depth = sparse_depth.astype(np.float32)

        self.image = None
        self.depth = None

    def setRoot(self, root):
        tmp = self.im_name.split("processed")

        if len(tmp) > 1:
            self.im_name = os.path.join(root, tmp[1].strip("/"))
        else:
            self.im_name = os.path.join(root, self.im_name)

    def getDepth(self):
        return self.depth

    def getCalib(self):
        return self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]

    def getCalibVec(self):
        return [self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]]

    def getFileName(self):
        return self.im_name

    def resize(self, dim):
        shape = self.image.shape

        h_new = dim[0]
        w_new = dim[1]

        h = shape[0]
        w = shape[1]

        sx = w_new/w
        sy = h_new/h

        if h_new != h and w_new != w:

            self.image = resize(self.image, dim)

            if self.depth is not None:
                mask = self.depth == 0.0
                mask = mask.astype(np.float32)

                self.depth = resize(self.depth, dim)
                mask = resize(self.mask, dim)
                mask = mask == 1.0
                mask = mask.astype(np.float32)
                self.depth = self.depth*mask

            self.rescale(sx, sy)

    def rescale(self, sx, sy):
        self.K = np.array([[sx, 0, 0],
                           [0, sy, 0],
                           [0, 0, 1]], dtype=np.float32).dot(self.K)

    def setFileName(self, name):
        self.im_name = name

    def setSparseDepth(self, sp_depth):
        self.sparse_depth = sp_depth

    def distance(self, sfm_image):
        return np.linalg.norm(self.getCameraCenter() - sfm_image.getCameraCenter())

    def getMeanDepth(self):

        if self.depth_mean is None:

            if self.depth is not None:

                self.depth_mean = tf.reduce_mean(self.depth).numpy()

            else:

                shape = self.sparse_depth.shape

                u = np.dot(self.P, np.concatenate(
                    (self.sparse_depth, np.ones((1, shape[-1])))))

                self.depth_mean = np.mean(u[2, :])

        return self.depth_mean

    def getId(self):
        return self.uuid

    def getP(self):
        return self.P

    def getK(self):
        return self.K

    def getKinv_Homogeneous(self):
        K_tilde_inv = np.array([[1.0/self.K[0, 0], 0.0, -self.K[0, 2]/self.K[0, 0], 0.0],
                                [0.0, 1.0/self.K[1, 1], -
                                    self.K[1, 2]/self.K[1, 1], 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

        return K_tilde_inv

    def getP_Homogeneous(self):
        T = np.identity(4, dtype=np.float64)
        T[0:3, 0:4] = self.P

        return T.astype(np.float32)

    def getSparseDepth(self):

        return self.sparse_depth

    def getCovisibleCameras(self):
        return self.covisible

    def load(self):
        self.image = imread(self.getFileName()).astype(np.float32)/255.0

        path = self.getFileName().split("/")

        name = path[-1].split(".")
        name = name[0]

        path = path[0:-1]

        path = "/".join(path)

        depth_path = os.path.join(path, "dense", name + ".png")

        if os.path.exists(depth_path):

            self.depth = imread(depth_path).astype(np.float32)/1000.0

    def save(self, path_out):

        path = self.getFileName().split("processed")
        path = path[-1].split("/")

        name = path[-1].split(".")
        name_image = "_".join(path[0:-1])+"_"+name[0]

        depth_path = os.path.join(path_out, "dense", name_image + ".png")
        image_path = os.path.join(path_out, name_image + ".jpg")

        imsave(image_path, (self.image*255.0).astype(np.uint8))

        self.im_name = image_path

        if self.depth is not None:

            im_depth = Image.fromarray((self.depth*1000.0).astype(np.uint32))

            im_depth.save(depth_path)

    def getImage(self):
        if self.image is None:
            self.load()

        return self.image

    def clearImage(self):
        self.image = None
        self.depth = None

    def getCameraCenter(self):

        return -np.dot(np.transpose(self.P[0:3, 0:3]), self.P[:, 3:])

    def getTransformations(self):
        T = np.identity(4, dtype=np.float64)
        T[0:3, 0:4] = self.P

        T_inv = np.identity(4, dtype=np.float64)

        T_inv[0:3, 0:3] = np.transpose(self.P[0:3, 0:3])
        T_inv[0:3, 3] = -np.dot(np.transpose(self.P[0:3, 0:3]), self.P[0:3, 3])

        K_tilde = np.identity(4, dtype=np.float64)
        K_tilde[0:3, 0:3] = self.K

        K_tilde_inv = np.array([[1.0/self.K[0, 0], 0.0, -self.K[0, 2]/self.K[0, 0], 0.0],
                                [0.0, 1.0/self.K[1, 1], -
                                    self.K[1, 2]/self.K[1, 1], 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

        return np.dot(K_tilde, T).astype(np.float32), np.dot(T_inv, K_tilde_inv).astype(np.float32)
