import cv2
import os
import sys


def resize_to_std_resolution(path_to_dataset):
    images = []
    dir_name = None
    for d, _, filenames in os.walk(os.path.join(path_to_dataset, 'data')):
        images.extend(filenames)
        dir_name = d
        break
    os.makedirs(os.path.join(path_to_dataset, 'high'), exist_ok=True)
    os.makedirs(os.path.join(path_to_dataset, 'low'), exist_ok=True)
    for file in images:
        image = cv2.imread(os.path.join(dir_name, file))
        high_res_image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC)
        low_res_image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
        low_res_interpolated = cv2.resize(low_res_image, (128, 128), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(path_to_dataset, 'high', file), high_res_image)
        cv2.imwrite(os.path.join(path_to_dataset, 'low', file), low_res_interpolated)


if __name__ == '__main__':
    if ('LOCAL' not in os.environ) and len(sys.argv) != 2:
        print('Usage: python prepare_dataset.py <path_to_dataset>')
    else:
        path = sys.argv[1] if len(sys.argv) == 2 else 'dataset'
        resize_to_std_resolution(path)
