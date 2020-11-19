import tensorflow as tf
from Sfm.sfm_loader import sfm_loader
from Sample.PhotometricSample import PhotometricSample
import numpy as np
import random
import shutil
import os
import copy
from threading import Condition, Thread, RLock
from time import sleep


class Sfm_loader_thread(Thread):
    def __init__(self, config):
        Thread.__init__(self)
        self.shape_size = config['model']['shape_size']
        self.min_covisibility = config['SFM']['min_covisibility']
        self.nbr_covisible_cameras = config['SFM']['nbr_covisible_cameras']
        self.buffer_th = config['SFM']['buffer_th']

        self.image_height = config['dataset']['image_height']
        self.image_width = config['dataset']['image_width']

        self.sfm_dataset = sfm_loader(config)
        self.sfm_dataset.load(loadImages=False)
        self.sfm_ids = []

        for id in self.sfm_dataset.getSFMDataset().keys():
            sfm_im = self.sfm_dataset.getImage(id)

            covisible = sfm_im.getCovisibleCameras()

            all_loaded = True
            nbr_covisible = 0

            for i, id_c in enumerate(covisible):

                if i >= self.nbr_covisible_cameras:
                    break

                tmp = self.sfm_dataset.getImage(id_c[0])

                if tmp is None:
                    all_loaded = False
                else:
                    nbr_covisible += 1

            if nbr_covisible >= self.nbr_covisible_cameras:
                self.sfm_ids.append(id)

        random.shuffle(self.sfm_ids)

        self.index = 0

        self.loaded_images = dict()
        self.de_ref_images = dict()
        self.terminateFlag = False
        self.condition = Condition(RLock())
        self.current_index = 0
        self.new_epoch = False

    def getNbrBatches(self):
        return int(len(self.sfm_ids))

    def terminate(self):

        self.condition.acquire()
        self.terminateFlag = True
        self.condition.notify_all()
        self.condition.release()

    def reset(self):
        self.condition.acquire()
        self.current_index = 0
        self.condition.release()

    def run(self):

        while not self.terminateFlag:

            for sfm_id in self.sfm_ids:

                if self.terminateFlag:
                    return True

                self.load_covisible_images(
                    self.sfm_dataset.getImage(sfm_id))

                self.condition.acquire()
                while len(self.loaded_images) > self.buffer_th:

                    self.condition.wait()

                    self.update_reference()

                    self.condition.notify_all()

                    if self.terminateFlag:
                        self.condition.release()
                        return True

                self.condition.release()

                self.update_reference()

                sleep(0.001)

            self.condition.acquire()
            while not self.new_epoch:
                self.condition.wait()
                self.update_reference()

                if self.terminateFlag:
                    self.condition.release()
                    return True

            random.shuffle(self.sfm_ids)
            self.new_epoch = False

            if self.terminateFlag:
                self.condition.release()
                return True

            self.condition.release()

        print("sfm thread shutting down")

    def load_covisible_images(self, sfm_image):

        covisible, _ = self.findCovisibleCameras(sfm_image)

        for image in covisible:
            if image.getId() in self.loaded_images:

                self.update_loaded_images(image.getId(), None, 1)

            else:

                image_copy = copy.copy(image)

                image_copy.load()

                image_copy.resize((self.image_height, self.image_width))

                self.update_loaded_images(image_copy.getId(), image_copy, 1)

    def update_loaded_images(self, id, image, nbr):

        self.condition.acquire()

        if image is not None:
            self.loaded_images.update({id: [image, nbr]})
        else:
            self.loaded_images[id][1] += nbr

            if self.loaded_images[id][1] <= 0:
                del self.loaded_images[id]

        self.condition.notify_all()
        self.condition.release()

    def update_reference(self):
        self.condition.acquire()

        for k, v in self.de_ref_images.items():
            self.update_loaded_images(k, None, -v)

        self.de_ref_images.clear()

        self.condition.notify_all()
        self.condition.release()

    def dereference(self, covisible):
        self.condition.acquire()

        for image in covisible:
            if image is None:
                continue

            if image.getId() in self.de_ref_images:
                self.de_ref_images[image.getId()] += 1
            else:
                self.de_ref_images.update({image.getId(): 1})

        self.condition.notify_all()
        self.condition.release()

    def wait_for_image(self, image):

        if not self.condition.acquire(timeout=5):
            return None

        while image.getId() not in self.loaded_images:
            if not self.condition.wait(5):
                return None

            if self.terminateFlag:
                self.condition.release()
                return None

        image_copy = copy.copy(self.loaded_images[image.getId()][0])

        self.condition.release()

        return image_copy

    def wait_for_covisible_images(self, covisible):
        copy_covisible = []

        for image in covisible:

            image_copy = self.wait_for_image(image)

            if image_copy is not None:
                copy_covisible.append(image_copy)

        return copy_covisible

    def fetch_images(self, sfm_image):

        covisible, _ = self.findCovisibleCameras(sfm_image)

        copy_covisible = self.wait_for_covisible_images(covisible)

        self.dereference(copy_covisible)

        return copy_covisible

    def fetch_next(self):

        sfm_im = self.sfm_dataset.getImage(self.sfm_ids[self.current_index])

        imgs = self.fetch_images(sfm_im)

        self.condition.acquire()

        self.current_index += 1

        newEpoch = False

        if self.current_index >= len(self.sfm_ids):
            self.current_index = 0
            self.new_epoch = True

            newEpoch = self.new_epoch

        self.condition.notify_all()
        self.condition.release()

        return imgs, newEpoch

    def getName(self):
        return "unsupervised"

    def findCovisibleCameras(self, sfm_image):
        covisible_images = []

        tmp = sfm_image.getCovisibleCameras()

        tmp.sort(key=lambda x: x[1], reverse=True)

        for covisible in tmp:

            if covisible[1] > self.min_covisibility:

                image = self.sfm_dataset.getImage(covisible[0])

                if image is not None:
                    covisible_images.append(image)

            if len(covisible_images) >= (self.nbr_covisible_cameras):
                break

        if len(covisible_images) < self.nbr_covisible_cameras:
            return covisible_images, False

        return covisible_images, True
