import cv2
import os
import sys

from multiprocessing import Pool
from multiprocessing import cpu_count
from tqdm import tqdm


def resize_to_std_resolution(image_name):
    os.makedirs(os.path.join(path_to_dataset, 'high'), exist_ok=True)
    os.makedirs(os.path.join(path_to_dataset, 'low'), exist_ok=True)
    image = cv2.imread(os.path.join(path_to_dataset, 'data', image_name))
    high_res_image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC)
    low_res_image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
    low_res_interpolated = cv2.resize(low_res_image, (128, 128), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(path_to_dataset, 'high', image_name), high_res_image)
    cv2.imwrite(os.path.join(path_to_dataset, 'low', image_name), low_res_interpolated)


def fetch_image_names(path_to_dataset):
    image_names = []
    for _, _, filenames in os.walk(os.path.join(path_to_dataset, 'data')):
        image_names.extend(filenames)
        break
    return image_names


if __name__ == '__main__':
    if ('LOCAL' not in os.environ) and len(sys.argv) != 2:
        print('Usage: python prepare_dataset.py <path_to_dataset>')
    else:
        path_to_dataset = sys.argv[1] if len(sys.argv) == 2 else 'dataset'
        images = fetch_image_names(path_to_dataset)

        pool = Pool(cpu_count())

        for _ in tqdm(pool.imap(resize_to_std_resolution, images), total=len(images)):
            pass
        pool.close()
        pool.join()
