from __future__ import division
import Converters.colmap_reader as cm
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


class sfm_colmap_converter:
    def __init__(self, config, input_folder, datasets):
        self.dataset_dir = config["sfm_dataset"]["root_dir"]
        self.dataset_colmap_dir = input_folder
        self.dataset_name = datasets.split(" ")
        self.shape = (config['dataset']['image_height'],
                      config['dataset']['image_width'], 3)
        self.output_folder = os.path.join(self.dataset_dir, "processed")

    def create_camera_files(self,  dataset):

        cameras, images, points3D = cm.read_model(os.path.join(
            self.dataset_colmap_dir, dataset, "dense/sparse"), ".bin")

        image_path = os.path.join(
            self.dataset_colmap_dir, dataset, "dense/images")

        uuids = dict()

        sfm_images = []

        for key, image in images.items():
            uuid_val = dataset+"_"+image.name  # str(uuid.uuid4())
            uuids.update({key: uuid_val})

        for key, image in images.items():
            name = os.path.join(image_path, image.name)

            camera = cameras[image.camera_id]

            params = camera.params

            K = np.array([[params[0], 0.0, params[2]],
                          [0.0, params[1], params[3]],
                          [0.0, 0.0, 1.0]], dtype=np.float32)

            R = cm.qvec2rotmat(image.qvec)

            t = np.expand_dims(image.tvec, axis=1)

            P = np.concatenate((R, t), axis=1)

            nbr_points = 0

            covisible = dict()

            uuid_val = uuids[key]

            for obj in image.point3D_ids:
                if obj > -1:
                    point = points3D[obj]

                    for im_id in point.image_ids:
                        id = uuids[im_id]

                        if id in covisible:
                            covisible[id] += 1
                        else:
                            covisible.update({id: 1})

                    nbr_points += 1

            covisible.update({uuid_val: nbr_points})

            covisible = [(k, v/nbr_points) for k, v in covisible.items()]

            covisible.sort(key=lambda x: x[1], reverse=True)

            U = np.zeros((3, nbr_points), dtype=np.float32)

            nbr_points = 0

            for obj in image.point3D_ids:
                if obj > -1:
                    point = points3D[obj]

                    U[:, nbr_points] = point.xyz

                    nbr_points += 1

            sfm_images.append(
                sfm_image(name, covisible, P, K, uuid_val, U))

        return sfm_images

    def process(self):

        for dir_ in self.dataset_name:
            #path = os.path.join(self.dataset_dir, dir_)
            self.convert(self.create_camera_files(dir_), dir_)

    def convert(self, sfm_images, dir):

        dict_file = {}

        output_folder_dataset = os.path.join(self.output_folder, dir)

        if os.path.exists(output_folder_dataset):
            shutil.rmtree(output_folder_dataset)

        for i, im in enumerate(sfm_images):

            image = Image.open(im.getFileName())
            image = np.array(image)
            # image = imread(im.getFileName())

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
    parser.add_argument('--input_folder', "--i", default="/data/colmap",
                        type=str, required=False)
    parser.add_argument('--datasets', "--d", default="Kallerum",
                        type=str, required=False)

    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)

            sfm_colmap_converter(config, args.input_folder,
                                 args.datasets).process()

        except yaml.YAMLError as exc:
            print(exc)
