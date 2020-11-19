
import yaml
import argparse
from Trainer.Manager import Manager
import tensorflow as tf


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
    parser.add_argument('--dataset', '--d', default="",
                        type=str, required=False)

    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)

            if args.dataset:

                config["sfm_dataset"]["datasets"] = [args.dataset]

            Manager(config).run()

        except yaml.YAMLError as exc:
            print(exc)
