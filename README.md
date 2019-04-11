# SuperPixel
Replicating the Super Resolution paper - https://arxiv.org/pdf/1707.00737.pdf

#### Dependencies
`pip install -r requirements.txt`
- `OpenCV`
- `Keras`
- `Numpy`

##### Prior to training
- Ensure the [dataset](https://drive.google.com/drive/folders/1GSaVehKdBp8NWG-QYJ_2nX6fiOFr5Z5H?usp=sharing) is downloaded into the project directory
- The directory structure should look as follows
```
SuperPixel
|
|--- dataset
      |
      |---- 10000.npy
```

##### Train Model

```
python main.py <path_to_dataset_numpy_arr>
```
