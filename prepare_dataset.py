import cv2
import os
import sys

from multiprocessing import Pool
from multiprocessing import cpu_count
from tqdm import tqdm

import numpy as np


class Constants(object):

    @staticmethod
    def output_directory():
        return 'output'

    @staticmethod
    def default_dataset_path():
        return 'small_dataset/data'

    @staticmethod
    def batch_size():
        return 1000


class ResizeImage(object):
    def __init__(self, path_to_dataset):
        self.path_to_dataset = path_to_dataset

    def __call__(self, image):
        return resize_to_std_resolution(image, self.path_to_dataset)


def resize_to_std_resolution(image_name, path_to_dataset):
    image = cv2.imread(os.path.join(path_to_dataset, image_name))
    high_res_image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC)
    low_res_image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
    low_res_image = cv2.resize(low_res_image, (128, 128), interpolation=cv2.INTER_CUBIC)
    return np.array([low_res_image, high_res_image], dtype=low_res_image.dtype)


def fetch_image_names(path_to_dataset):
    return next(os.walk(path_to_dataset))[2]


def save_to_disk_in_batches(array, chunk_size, output_path):
    num_of_chunks = int(len(array) / chunk_size)
    i = 0
    for chunk in tqdm(np.array_split(array, num_of_chunks), desc='Flushing to Disk'):
        np.save(os.path.join(output_path, '{}.npy'.format(str(i))), chunk, fix_imports=False)
        i += 1


if __name__ == '__main__':
    if ('LOCAL' not in os.environ) and len(sys.argv) != 2:
        print('Usage: python prepare_dataset.py <path_to_dataset>')
        sys.exit(1)

    path = sys.argv[1] if len(sys.argv) == 2 else Constants.default_dataset_path()
    images = fetch_image_names(path)

    pool = Pool(cpu_count())
    dataset_array = np.array([example for example in
                              tqdm(pool.imap(ResizeImage(path), images), total=len(images), desc='Processing Images')])
    pool.close()
    pool.join()

    os.makedirs(Constants.output_directory(), exist_ok=True)
    save_to_disk_in_batches(dataset_array, Constants.batch_size(), Constants.output_directory())
