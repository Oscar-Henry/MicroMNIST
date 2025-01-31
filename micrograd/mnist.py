import os
import gzip
import pickle
import numpy as np

DATASET_DIR = "C:\\Users\\hosca\\Documents\\MNIST\\DATASET"
SAVE_FILE = DATASET_DIR + "\\mnist.pkl"
FILE_NAME = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

TRAIN_NUM = 60000
TEST_NUM = 10000
image_dimension = (1, 28, 28)
image_size = 784

def _load_label(file_name: str):
    file_path = DATASET_DIR + "\\" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")
    
    return labels

def _load_img(file_name):
    file_path = DATASET_DIR + "\\" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")    
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, image_size)
    print("Done")
    
    return data

def _convert_numpy():
    dataset = {}
    dataset['train_img'] =  _load_img(FILE_NAME['train_img'])
    dataset['train_label'] = _load_label(FILE_NAME['train_label'])    
    dataset['test_img'] = _load_img(FILE_NAME['test_img'])
    dataset['test_label'] = _load_label(FILE_NAME['test_label'])
    
    return dataset

def init_mnist():
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(SAVE_FILE, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    if not os.path.exists(SAVE_FILE):
        init_mnist()
        
    with open(SAVE_FILE, 'rb') as f:
        dataset = pickle.load(f)
    
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
            
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])
    
    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])     