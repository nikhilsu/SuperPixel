import numpy as np
import sys
import os


def split_combination(np_array_path):
    split1 = np.load(np_array_path)
    sizes = [100, 200, 500, 1000, 2000, 5000]

    for s in sizes:
        np.random.shuffle(split1)
        arr = split1[:s]
        np.save('dataset/{}.npy'.format(str(s)), arr, fix_imports=False)
        print(s, arr.shape)


def split_half(np_array_path):
    np_array = np.load(np_array_path)
    split_index = int(len(np_array) / 2)
    file_name, ext = os.path.splitext(np_array_path)
    np.save('{}_0{}'.format(file_name, ext), np_array[:split_index])
    np.save('{}_1{}'.format(file_name, ext), np_array[split_index:])


if __name__ == '__main__':
    if ('LOCAL' not in os.environ) and len(sys.argv) < 2:
        print('Usage: python dataset_split.py <path_to_fat_numpy_array> ["half"]')
        sys.exit(1)

    path = sys.argv[1] if len(sys.argv) >= 2 else '../dataset/500.npy'
    half = 'LOCAL' in os.environ or len(sys.argv) == 3
    if half:
        split_half(path)
    else:
        split_combination(path)
