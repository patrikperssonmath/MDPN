import tensorflow as tf
import numpy as np
from PIL import Image
import os
from Graphics.Graphics3 import Graphics3
from tensorflow_addons.image import interpolate_bilinear
import shutil
import csv
from Sfm.sfm_image import sfm_image
from Converters.KeyFrame import KeyFrame
from Converters.VoxelMap import VoxelMap
import time
from skimage.transform import resize
import pickle
import argparse
import yaml


class convert_rgbd_to_sfm:
    def __init__(self, config, datafolder):
        self.sfm_dataset_dir = config["sfm_dataset"]["root_dir"]
        self.datafolder = datafolder
        self.dimensions = (config['dataset']['image_width'],
                           config['dataset']['image_height'])

        self.g = Graphics3()

    def process(self):

        subfolders = [name for name in os.listdir(
            self.datafolder) if os.path.isdir(os.path.join(self.datafolder, name))]

        for folder in subfolders:
            path_dataset = os.path.join(self.datafolder, folder)

            self.process_folder(path_dataset)

    def generate_sfm_images(self, path_dataset, row_count):

        samples = dict()

        for idx in range(row_count):
            image, depth, P, K, flag = self.load_image(
                path_dataset, idx)

            if not flag:
                continue

            id = os.path.join(path_dataset,
                              'frame-{0:06d}'.format(idx))

            sfm_image_data = sfm_image(idx, [], P[0:3, 0:4], K, id, None)

            sfm_image_data.depth = depth
            sfm_image_data.image = image

            samples.update({id: sfm_image_data})

            print("processed: {0} of {1}".format(str(idx), str(row_count)))

        return samples

    def getNumberSamples(self, path_dataset):
        path_dataset_csv = os.path.join(path_dataset, "data.csv")

        with open(path_dataset_csv, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')

            row_count = sum(1 for row in reader)

        return row_count

    def convert_to_tensor(self, image_i):

        tf_image_i = tf.convert_to_tensor(
            image_i.getImage(), dtype=tf.float32)
        tf_image_depth_i = tf.convert_to_tensor(
            image_i.getDepth(), dtype=tf.float32)

        tf_image_depth_i = tf.expand_dims(tf_image_depth_i, axis=-1)

        T_i, Tinv_i = image_i.getTransformations()

        T_i = tf.convert_to_tensor(T_i, dtype=tf.float32)
        Tinv_i = tf.convert_to_tensor(Tinv_i, dtype=tf.float32)

        calib_i = tf.convert_to_tensor(
            image_i.getCalibVec(), dtype=tf.float32)

        return tf_image_i, tf_image_depth_i, T_i, Tinv_i, calib_i

    def process_folder(self, path_dataset):

        voxels = VoxelMap()

        row_count = self.getNumberSamples(path_dataset)

        #row_count = 500

        samples = self.generate_sfm_images(path_dataset, row_count)

        if not samples:
            return

        g = Graphics3()

        angle = tf.constant(np.cos(80.0*np.pi/180.0), dtype=tf.float32)

        keyframes = []

        overlap_graph = dict()
        parallax_graph = dict()

        for i in range(row_count):

            id_i = os.path.join(path_dataset, 'frame-{0:06d}'.format(i))

            sample = samples[id_i]

            keyframe = KeyFrame(sample)

            t0 = time.perf_counter()

            X, id = keyframe.unproject(g)

            t1 = time.perf_counter()

            voxels.update(X, id, g)

            t2 = time.perf_counter()

            print("iteration {0} of {1}, ms: {2}, {3}".format(
                i, row_count, 1000.0*(t2-t1), 1000.0*(t1-t0)))

        cache = dict()

        for i in range(row_count):

            id_i = os.path.join(path_dataset, 'frame-{0:06d}'.format(i))

            sample = samples[id_i]

            keyframe_ref = KeyFrame(sample)

            t1 = time.perf_counter()

            X, id = keyframe_ref.unproject(g)

            covisible_ids = voxels.fetch(X, g)

            t2 = time.perf_counter()

            keyframes = [keyframe_ref]

            del covisible_ids[id[0]]

            covisible_ids = [(k, v) for k, v in covisible_ids.items()]

            covisible_ids = [
                x for x in covisible_ids if keyframes[0].rel_distance(samples[x[0]], 0.01)]

            def myFunc(e):
                return e[1]

            M = min([len(covisible_ids), 30])

            N = min([len(covisible_ids), 4])

            covisible_ids.sort(key=myFunc, reverse=True)

            covisible_ids = covisible_ids[0:M]

            while len(keyframes) < (N+1) and len(covisible_ids) > 0:

                max_id = -1
                max_val = -float('Inf')

                for id in covisible_ids:
                    sample_j = KeyFrame(samples[id[0]])

                    min_val = float('Inf')

                    for frame in keyframes:

                        id_ij = frame.image.getId() + id[0]

                        if id_ij in cache:

                            overlap, parallax = cache[id_ij]

                        else:

                            overlap, parallax = sample_j.compare(
                                frame.image, g, angle)

                            id_ji = id[0] + frame.image.getId()

                            cache.update({id_ij: [overlap, parallax]})
                            cache.update({id_ji: [overlap, parallax]})

                        if parallax < min_val:  # and overlap > 0.5:
                            min_val = parallax

                    if max_val < min_val:
                        max_val = min_val
                        max_id = id[0]

                if isinstance(max_id, str):

                    candidate = KeyFrame(samples[max_id])

                    overlaps = []

                    for frame in keyframes:
                        id_ij = frame.image.getId() + candidate.image.getId()
                        overlap, parallax = cache[id_ij]

                        overlaps.append(overlap)

                    if min(overlaps) > 0.40:

                        keyframes.append(candidate)

                        print("added keyframe!")

                    covisible_ids = [
                        x for x in covisible_ids if x[0] != max_id]

            t3 = time.perf_counter()

            print("iteration retrive {0} of {1}, ms: {2}, total ms: {3}".format(
                i, row_count, 1000.0*(t2-t1), 1000.0*(t3-t1)))

            covisibility = []

            test = set()

            for frame in keyframes:

                if frame.image.getId() != keyframe_ref.image.getId():

                    id_ij = keyframe_ref.image.getId() + frame.image.getId()

                    overlap, parallax = cache[id_ij]

                    _, _, total_id = self.splitPathAndId(
                        frame.image.getId())

                    covisibility.append((total_id, overlap))

                else:
                    _, _, total_id = self.splitPathAndId(
                        frame.image.getId())

                    covisibility.append((total_id, 1.0))

                if frame.image.getId() in test:
                    print("error!! ################ ")

                test.add(frame.image.getId())

            if len(covisibility) >= 5:

                keyframe_ref.image.covisible = covisibility

                """

                path = os.path.join("test", str(i))

                if os.path.exists(path):
                    shutil.rmtree(path)

                os.makedirs(path)


                for key in keyframes:

                    im = Image.fromarray(
                        (key.image.image*255.0).astype(np.uint8))

                    im.save(os.path.join(path, "test_{0}.jpg").format(
                        key.image.getFileName()))

                """

        self.store_sfm_images(samples)

    def splitPathAndId(self, pathid):
        tmp = pathid.split("/")

        id = tmp[-1]
        rel_path = os.path.join(tmp[-3], tmp[-2])

        total_id = tmp[-3]+"_"+tmp[-2]+"_"+id

        return rel_path, id, total_id

    def store_sfm_images(self, sfm_images):
        sfm_images_store = dict()

        for image_id, image in sfm_images.items():

            if len(image.covisible) < 5:
                continue

            rel_path, id, total_id = self.splitPathAndId(image.getId())

            path = os.path.join(
                self.sfm_dataset_dir, "processed", rel_path)

            if not os.path.exists(path):
                os.makedirs(path)

            path_dense = os.path.join(path, "dense")

            if not os.path.exists(path_dense):
                os.makedirs(path_dense)

            im = Image.fromarray((image.image*255.0).astype(np.uint8))

            im_name = os.path.join(path, id+'.jpg')
            im.save(im_name)

            im_depth = Image.fromarray((image.depth*1000.0).astype(np.uint32))

            im_depth.save(os.path.join(
                path_dense, id+'.png'))

            uuid = total_id

            image.covisible.sort(key=lambda x: x[1], reverse=True)

            sfm_im = sfm_image(im_name, image.covisible,
                               image.P, image.K, uuid, None)

            sfm_images_store.update({uuid: sfm_im.__dict__})

        with open(os.path.join(path, "sfm_images.pickle"), 'wb') as file:
            pickle.dump(sfm_images_store, file)

    def load_image(self, path_dataset, idx):

        img_path = os.path.join(
            path_dataset, 'frame-{0:06d}.color.jpg'.format(idx))

        if not os.path.exists(img_path):
            img_path = os.path.join(
                path_dataset, 'frame-{0:06d}.color.png'.format(idx))

        depth_path = os.path.join(
            path_dataset, 'frame-{0:06d}.depth.pgm'.format(idx))

        pose_path = os.path.join(
            path_dataset, 'frame-{0:06d}.pose.txt'.format(idx))

        if not os.path.exists(pose_path):
            return [], [], [], [], False

        calib_path = os.path.join(
            path_dataset, 'frame-{0:06d}.csv'.format(idx))

        im = Image.open(img_path)

        image = np.array(im).astype(np.float32) / 255.0

        depth = Image.open(depth_path)

        depth = np.array(depth).astype(np.float32) / 1000.0

        sx = self.dimensions[0]/im.width
        sy = self.dimensions[1]/im.height

        mask = (depth != 0).astype(np.float32)

        depth_r = resize(depth, (self.dimensions[1], self.dimensions[0]))
        mask_r = resize(mask, (self.dimensions[1], self.dimensions[0]))
        image = resize(image, (self.dimensions[1], self.dimensions[0]))

        mask_r = mask_r == 1

        mask_r = mask_r.astype(np.float32)

        depth = depth_r*mask_r

        f = open(calib_path, "r")

        l = f.readline()

        parts = l.split(",")

        fx = float(parts[0])*sx
        fy = float(parts[1])*sy
        x0 = float(parts[2])*sx
        y0 = float(parts[3])*sy

        f.close()

        f = open(pose_path, "r")

        lines = f.readlines()

        P = []

        for l in lines:

            parts = l.split(" ")

            p1 = float(parts[0])
            p2 = float(parts[1])
            p3 = float(parts[2])
            p4 = float(parts[3])

            P.append([p1, p2, p3, p4])

        f.close()

        P = np.array(P)
        K = np.array([[fx, 0.0, x0], [0.0, fy, y0], [0.0, 0.0, 1.0]])

        return image, depth, np.linalg.inv(P), K, True


if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    parser = argparse.ArgumentParser()

    parser.add_argument('config', type=str)
    parser.add_argument('--input_folder', '--i', default="/data/scannet/rgbd_dataset/",
                        type=str, required=False)

    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)

            convert_rgbd_to_sfm(config, args.input_folder).process()

        except yaml.YAMLError as exc:
            print(exc)
