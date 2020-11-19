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


class sub_sample_sfm:
    def __init__(self, config, name, N_samples):
        self.shape_size = config['model']['shape_size']
        self.min_covisibility = config['SFM']['min_covisibility']
        self.nbr_covisible_cameras = config['SFM']['nbr_covisible_cameras']
        self.buffer_th = config['SFM']['buffer_th']

        self.image_height = config['dataset']['image_height']
        self.image_width = config['dataset']['image_width']

        self.dataset_dir = config["sfm_dataset"]["root_dir"]
        self.processed_dir = os.path.join(self.dataset_dir, "processed")
        self.N_samples = N_samples

        self.name = name

        path = os.path.join(self.processed_dir, self.name)

        if os.path.exists(path):
            shutil.rmtree(path)

        folders = [name for name in os.listdir(
            self.processed_dir) if os.path.isdir(os.path.join(self.processed_dir, name))]

        datasets = []

        for folder in folders:

            current_folder = os.path.join(self.processed_dir, folder)

            subfolders = [name for name in os.listdir(
                current_folder) if os.path.isdir(os.path.join(current_folder, name))]

            N = 0
            for subfolder in subfolders:

                if subfolder != "image" and subfolder != "dense":

                    datasets.append(os.path.join(folder, subfolder))
                    N += 1

            if N == 0:
                datasets.append(folder)

        self.sfm_dataset = sfm_loader(config, datasets)
        self.sfm_dataset.load(loadImages=False)
        self.sfm_ids = []

        for id in self.sfm_dataset.getSFMDataset().keys():
            self.sfm_ids.append(id)

        random.shuffle(self.sfm_ids)

        self.index = 0

        self.current_index = 0

    def process(self):

        dict_file = {}

        path = os.path.join(self.processed_dir, self.name)

        if os.path.exists(path):
            shutil.rmtree(path)

        os.makedirs(path)

        path_dense = os.path.join(self.processed_dir, self.name, "dense")

        os.makedirs(path_dense)

        trainable_ids = set()

        for i, sfm_id in enumerate(self.sfm_ids):

            covisible, _ = self.findCovisibleCameras(
                self.sfm_dataset.getImage(sfm_id))

            if len(covisible) != self.nbr_covisible_cameras:
                continue

            for im in covisible:

                if im.getId() in dict_file:
                    continue

                im_copy = copy.copy(im)

                im_copy.load()

                im_copy.save(path)

                im_copy.depth = None
                im_copy.image = None

                dict_file.update({im_copy.getId(): im_copy.__dict__})

            trainable_ids.add(sfm_id)

            print("Done %i out of %i" %
                  (len(trainable_ids), min([len(self.sfm_ids), self.N_samples])), end="\r")

            if len(trainable_ids) >= self.N_samples:
                break

        print("")

        with open(os.path.join(path, "sfm_images.pickle"), 'wb') as file:
            pickle.dump(dict_file, file)

    def findCovisibleCameras(self, sfm_image):
        covisible_images = []

        contained_itself = False

        for covisible in sfm_image.getCovisibleCameras():

            if covisible[1] > self.min_covisibility:
                covisible_images.append(
                    self.sfm_dataset.getImage(covisible[0]))

            if covisible[0] == sfm_image.getId():
                contained_itself = True

            if len(covisible_images) >= (self.nbr_covisible_cameras):
                break

        if not contained_itself:
            print("findCovisibleCameras: not covisible with itself!")

        if len(covisible_images) < self.nbr_covisible_cameras:
            return covisible_images, False

        return covisible_images, True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('config', type=str)
    parser.add_argument('--name', '--n', default="subsample",
                        type=str, required=False)
    parser.add_argument('--samples', '--s', default=10000,
                        type=str, required=False)

    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)

            sub_sample_sfm(config, args.name, int(args.samples)).process()

        except yaml.YAMLError as exc:
            print(exc)
