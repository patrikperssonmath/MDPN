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

from __future__ import division
import numpy as np
from glob import glob
import os
import imageio
import scipy.io
import csv
import uuid
from skimage.io import imread, imsave
from skimage.transform import resize
import numpy as np
import pickle
from PIL import Image
import shutil
import argparse
import yaml

# Own packages
from Sfm.sfm_image import sfm_image


class sfm_data_converter:
    def __init__(self, config, input_folder, datasets):
        self.dataset_dir = config["sfm_dataset"]["root_dir"]
        self.dataset_name = datasets.split(" ")
        self.dataset_carl_sfm = input_folder
        self.shape = (config['dataset']['image_height'],
                      config['dataset']['image_width'], 3)
        self.output_folder = os.path.join(self.dataset_dir, "processed")

    def create_camera_files(self, mat_file, dir_):
        U, P = mat_file['U'], mat_file['P']
        U_vis = np.zeros((U.shape[1], P.shape[1]))

        indicies = mat_file['u_uncalib']['index'][0, 0]

        for i, row in enumerate(indicies):
            ind = row[0][0]
            for j in ind:
                U_vis[j-1, i] = 1  # Due to allocation MATLAB/python (-1)

        P_covis = np.zeros((P.shape[1], P.shape[1]))

        for i in range(P.shape[1]):
            ind = mat_file['u_uncalib']['index'][0, 0][i][0]

            for j in range(ind.shape[1]):
                covisible = U_vis[ind[0][j]-1, :].nonzero()[0]
                P_covis[i, covisible] = P_covis[i, covisible] + 1

        uuids = [str(uuid.uuid4()) for i in range(P.shape[1])]

        sfm_images = []

        im_names = mat_file["imnames"]

        for i in range(P.shape[1]):
            R, Q = self.rq_factorization(P[0][i])
            ind = mat_file['u_uncalib']['index'][0, 0][i][0]

            covisible = []

            for j in range(P.shape[1]):
                if(P_covis[i, j] > 0):
                    covisible.append((uuids[j], P_covis[i, j]))

            covisible.sort(key=lambda x: x[1], reverse=True)

            nbr_points = covisible[0][1]

            covisible = [(k, v/nbr_points) for k, v in covisible]

            sfm_images.append(sfm_image(os.path.join(dir_, "images", im_names[i][0][0][0]),
                                        covisible, Q, R, uuids[i], U[0:3, ind[0]-1]))

        return sfm_images
    """
    def get_sparse_depthmap(self, p_global, K, P, shape_):

        n, m, _ = shape_

        p_camera = np.matmul(P, p_global)

        p_im = np.matmul(np.matmul(K,P), p_global)
        p_im /= p_im[2]
        p_im = np.round(p_im).astype(np.int64)

        bool_1 = (p_im[0,:] >= 0) * (p_im[0,:] < n) * (p_im[1,:] >= 0) * (p_im[1,:] < m)
        depth_im = np.zeros((n, m), dtype = np.float32)

        i = 0
        for i_dx in p_im.T:
            if bool_1[i]:
                if not depth_im[i_dx[0], i_dx[1]] == 0:
                    depth_im[i_dx[0], i_dx[1]] = (depth_im[i_dx[0], i_dx[1]]+p_camera[2,i])/2 #OBS indices
                else:
                    depth_im[i_dx[0], i_dx[1]] = p_camera[2,i]
            i += 1
        return depth_im"""

    def rq_factorization(self, A):
        """RQ-factorization of matrix A"""
        m, n = A.shape
        e = np.eye((m))
        p = np.rot90(e)

        q0, r0 = np.linalg.qr(np.dot(p, np.dot(A[:, 0:m].T, p)))
        r = np.dot(p, np.dot(r0.T, p))
        q = np.dot(p, np.dot(q0.T, p))

        fix = np.diag(np.sign(np.diag(r)))
        r = np.dot(r, fix)
        q = np.dot(fix, q)

        if n > m:
            q = np.concatenate(
                (q, np.dot(np.linalg.inv(r), A[:, m:n])), axis=1)

        return r, q

    def process(self):

        if self.dataset_name[0] == "*":
            subfolders = [name for name in os.listdir(
                self.dataset_carl_sfm) if os.path.isdir(os.path.join(self.dataset_carl_sfm, name))]

            self.dataset_name = subfolders

        for dir_ in self.dataset_name:
            path = os.path.join(self.dataset_carl_sfm, dir_)
            data_mat = glob(path+'/*.mat')
            mat = scipy.io.loadmat(data_mat[0])
            self.convert(self.create_camera_files(mat, path), dir_)

    def convert(self, sfm_images, dir):

        dict_file = {}

        output_folder_dataset = os.path.join(self.output_folder, dir)

        if os.path.exists(output_folder_dataset):
            shutil.rmtree(output_folder_dataset)

        for i, im in enumerate(sfm_images):

            image = Image.open(im.getFileName())
            image = np.array(image)
            #image = imread(im.getFileName())

            dim = image.shape

            sy = self.shape[0]/dim[0]
            sx = self.shape[1]/dim[1]

            image_r = resize(image, self.shape)

            path = im.getFileName().split("/")

            name = path[-1]

            im.rescale(sx, sy)

            im.setFileName(os.path.join(output_folder_dataset, "image", name))

            image_folder_path = os.path.join(output_folder_dataset, "image")

            if not os.path.exists(image_folder_path):

                os.makedirs(image_folder_path)

            imsave(im.getFileName(), (image_r*255.0).astype(np.uint8))

            im.image = None
            im.depth = None

            dict_file.update({im.getId(): im.__dict__})

            print("Done %i out of %i" % (i+1, len(sfm_images)), end="\r")

        print("")

        with open(os.path.join(output_folder_dataset, "sfm_images.pickle"), 'wb') as file:
            pickle.dump(dict_file, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('config', type=str)
    parser.add_argument('--input_folder', "--i", default="/data/carl_sfm_indoor",
                        type=str, required=False)
    parser.add_argument('--datasets', "--d", default="door fountain Fort_Channing_gate",
                        type=str, required=False)

    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)

            sfm_data_converter(config, args.input_folder,
                               args.datasets).process()

        except yaml.YAMLError as exc:
            print(exc)
