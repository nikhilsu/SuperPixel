# SuperPixel
Replicating the Super Resolution paper - https://arxiv.org/pdf/1707.00737.pdf

#### Dependencies
`pip install -r requirements.txt`
- `OpenCV`
- `Keras`
- `Numpy`
- `Matplotlib`

##### Prior to training
- Ensure the [dataset](https://drive.google.com/file/d/1lPZzXfoacc-5V99avIJ8XOo4sak-dHCp/view?usp=sharing) is downloaded into the project directory
- The dataset directory should look like the following
```
|--- dataset
      |
      |---- 0.npy
      |---- 1.npy
      ...
      |---- 20.npy
```

##### Train Model

```
python main.py <path_to_dataset_directory>
```
