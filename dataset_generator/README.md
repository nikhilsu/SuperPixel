# Creating Dataset for SuperPixel
Replicating the Super Resolution paper - https://arxiv.org/pdf/1707.00737.pdf

#### Dependencies
- ```OpenCV```
- ```Keras```
- ```Numpy```

#### Create Dataset

```
python prepare_dataset.py <path_to_raw_images>
```
##### Output
- Dataset will be generated as numpy array of image pairs (low-res, high-res).
- Each numpy arrays hold a max of 10,000 pairs.
- The numpy arrays will be stored in the current working directory under the folder `output`.

**Note**: *Use the `dataset_split.py` script to split numpy arrays into smaller chunks*
