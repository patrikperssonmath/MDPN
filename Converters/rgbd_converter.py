import tensorflow as tf
import numpy as np
from PIL import Image
import os
from Graphics.Graphics3 import Graphics3
from tensorflow_addons.image import interpolate_bilinear
import shutil
import argparse
import yaml


class rgbd_converter:
    def __init__(self, config, input_folder, output_folder):
        self.config = config
        self.dataset_input_dir = input_folder
        self.dataset_output_dir = output_folder
        self.g = Graphics3()

    def getCalibrationData(self, path):

        var = dict()

        f = open(os.path.join(path, "_info.txt"), "r")

        lines = f.readlines()

        for l in lines:
            l = l.strip('\n')
            parts = l.split("=")

            var.update({parts[0].strip(' '): parts[1:]})
            print(l)

        f.close()

        rgb_w = int(var["m_colorWidth"][0])
        rgb_h = int(var["m_colorHeight"][0])
        depth_w = int(var["m_depthWidth"][0])
        depth_h = int(var["m_depthHeight"][0])
        depth_shift = float(var["m_depthShift"][0])

        size = int(var["m_frames.size"][0])

        m_calibrationColorIntrinsic = np.reshape(np.array(
            var["m_calibrationColorIntrinsic"][0].strip().split(" "), dtype=np.float32), (1, 4, 4))

        m_calibrationDepthIntrinsic = np.reshape(np.array(
            var["m_calibrationDepthIntrinsic"][0].strip().split(" "), dtype=np.float32), (1, 4, 4))

        return rgb_w, rgb_h, depth_w, depth_h, depth_shift, size, m_calibrationColorIntrinsic, m_calibrationDepthIntrinsic

    def process(self):

        dataset = self.dataset_input_dir

        self.process_dataset(dataset)

    def process_dataset(self, path):

        path_conv = self.dataset_output_dir

        subfolders = [name for name in os.listdir(
            path) if os.path.isdir(os.path.join(path, name))]

        for folder in subfolders:

            path_dataset = os.path.join(path, folder)
            path_output = os.path.join(path_conv, folder)

            _, _, depth_w, depth_h, depth_shift, size, Kc, Kd = self.getCalibrationData(
                path_dataset)

            Kc = tf.constant(Kc[:, 0:3, 0:3])
            Kd = tf.constant(Kd[:, 0:3, 0:3])

            if os.path.exists(path_output):
                shutil.rmtree(path_output)

            os.makedirs(path_output)

            for idx in range(size):
                csv = open(os.path.join(path_output, "data.csv"), "a+")

                img_path = os.path.join(
                    path_dataset, 'frame-{0:06d}.color.jpg'.format(idx))

                if not os.path.exists(img_path):
                    img_path = os.path.join(
                        path_dataset, 'frame-{0:06d}.color.png'.format(idx))

                depth_path = os.path.join(
                    path_dataset, 'frame-{0:06d}.depth.pgm'.format(idx))

                im = Image.open(img_path)

                im = tf.expand_dims(tf.convert_to_tensor(
                    np.array(im), dtype=tf.float32), axis=0)/255.0

                depth_im = Image.open(depth_path)
                depth = (np.array(depth_im).astype(np.float32)/depth_shift)
                depth = tf.reshape(tf.convert_to_tensor(
                    depth, dtype=tf.float32), shape=[1, depth_h, depth_w, 1])

                s = 1.1

                s_d = tf.reshape(np.array([[s, 0.0, 1.0],
                                           [0.0, s, 1.0],
                                           [0.0, 0.0, 1.0]], dtype=np.float32), shape=[1, 3, 3])

                Kd_zoom = s_d * Kd

                im_zoom = self.convert_to_calibration(Kc, Kd_zoom, im, depth)

                mask = tf.cast(tf.greater(depth, 0.0), tf.float32)

                depth_zoom = self.convert_to_calibration(
                    Kd, Kd_zoom, depth, depth)
                mask_zoom = self.convert_to_calibration(
                    Kd, Kd_zoom, mask, mask)

                mask_zoom = tf.equal(mask_zoom, 1.0)

                depth_zoom = tf.where(mask_zoom, depth_zoom,
                                      tf.zeros_like(depth_zoom))

                im = Image.fromarray(
                    (im_zoom[0]*255.0).numpy().astype(np.uint8))
                depth_im = Image.fromarray(
                    (depth_zoom[0, :, :, 0]*1000.0).numpy().astype(np.uint32))

                img_path = os.path.join(
                    path_output, 'frame-{0:06d}.color.png'.format(idx))
                depth_path = os.path.join(
                    path_output, 'frame-{0:06d}.depth.pgm'.format(idx))
                mask_path = os.path.join(
                    path_output, 'frame-{0:06d}.mask.png'.format(idx))

                calibration_path = os.path.join(
                    path_output, 'frame-{0:06d}.csv'.format(idx))

                pose_path_input = os.path.join(
                    path_dataset, 'frame-{0:06d}.pose.txt'.format(idx))

                pose_path_output = os.path.join(
                    path_output, 'frame-{0:06d}.pose.txt'.format(idx))

                if os.path.exists(pose_path_input):
                    shutil.copyfile(pose_path_input,
                                    pose_path_output)

                K = Kd_zoom[0].numpy()

                f = open(calibration_path, "w")

                f.write("{0},{1},{2},{3},".format(
                    K[0, 0], K[1, 1], K[0, 2], K[1, 2]))

                f.close()

                im.save(img_path)
                depth_im.save(depth_path)

                mask_zoom = tf.cast(mask_zoom, tf.float32)

                mask_im = Image.fromarray(
                    (mask_zoom[0, :, :, 0]*255.0).numpy().astype(np.uint8))

                # mask_im.save(mask_path)

                csv.write("{0},{1},{2},{3}\n".format(
                    img_path, depth_path, mask_path, calibration_path))

                csv.close()

                print(folder+" {0} of {1}".format(idx, size))

    @tf.function
    def convert_to_calibration(self, Kc, Kd, image, depth):

        shape_i = tf.shape(image)
        shape_d = tf.shape(depth)

        grid_d = self.g.generate_homogeneous_points(depth)

        grid_d = tf.reshape(grid_d, [shape_d[0], shape_d[1]*shape_d[2], 3])
        grid_d = tf.transpose(grid_d, perm=[0, 2, 1])

        Kd_inv = tf.linalg.inv(Kd)

        x = tf.matmul(tf.matmul(Kc, Kd_inv), grid_d)
        x = tf.transpose(x, perm=[0, 2, 1])

        I_int = interpolate_bilinear(image, x[:, :, 0:2], indexing='xy')

        Im_transformed = tf.reshape(
            I_int, [shape_d[0], shape_d[1], shape_d[2], shape_i[-1]])

        return Im_transformed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('config', type=str)
    parser.add_argument('--input_folder', '--i', default="/database/shape_raw_data/scannet/exported",
                        type=str, required=False)

    parser.add_argument('--output_folder', '--o' default="/database/shape_dataset/scannet",
                        type=str, required=False)

    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)

            rgbd_converter(config, args.input_folder,
                           args.output_folder).process()

        except yaml.YAMLError as exc:
            print(exc)
