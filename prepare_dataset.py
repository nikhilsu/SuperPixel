import cv2
import os
import sys

from multiprocessing import Pool
from multiprocessing import cpu_count
from tqdm import tqdm

import numpy as np


class ResizeImage(object):
    def __init__(self, path_to_dataset):
        self.path_to_dataset = path_to_dataset

    def __call__(self, image):
        return resize_to_std_resolution(image, self.path_to_dataset)


def resize_to_std_resolution(image_name, path_to_dataset):
    os.makedirs(os.path.join(path_to_dataset, 'high'), exist_ok=True)
    os.makedirs(os.path.join(path_to_dataset, 'low'), exist_ok=True)
    image = cv2.imread(os.path.join(path_to_dataset, 'data', image_name))
    high_res_image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC)
    low_res_image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
    low_res_image = cv2.resize(low_res_image, (128, 128), interpolation=cv2.INTER_CUBIC)
    return np.array([low_res_image, high_res_image], dtype=low_res_image.dtype)


def fetch_image_names(path_to_dataset):
    return next(os.walk(os.path.join(path_to_dataset, 'data')))[2]


def save_to_disk_in_batches(array, size, output_path):
    chunk_size = int(len(array) / size)
    i = 0
    for chunk in tqdm(np.array_split(array, chunk_size), desc='Flushing to Disk'):
        np.save(os.path.join(output_path, '{}.npy'.format(str(i))), chunk, fix_imports=False)
        i += 1


if __name__ == '__main__':
    if ('LOCAL' not in os.environ) and len(sys.argv) != 2:
        print('Usage: python prepare_dataset.py <path_to_dataset>')
        sys.exit(1)

    path = sys.argv[1] if len(sys.argv) == 2 else 'small_dataset'
    batch_size = 1000
    images = fetch_image_names(path)

    pool = Pool(cpu_count())
    dataset_array = np.array([example for example in tqdm(pool.imap(ResizeImage(path), images), total=len(images))])
    pool.close()
    pool.join()

    os.makedirs('output', exist_ok=True)
    save_to_disk_in_batches(dataset_array, batch_size, 'output')
