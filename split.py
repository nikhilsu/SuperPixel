import numpy as np

split1 = np.load('dataset/10000_1.npy')

sizes = [100, 200, 500, 1000, 2000, 5000]

for s in sizes:
    np.random.shuffle(split1)
    arr = split1[:s]
    np.save('dataset/{}.npy'.format(str(s)), arr, fix_imports=False)
    print(s, arr.shape)
