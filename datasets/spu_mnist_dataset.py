import os
import torch
from PIL import Image
import pandas as pd
import pickle
from torchvision import datasets

import numpy as np
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy

def make_environment(images, labels, fracs):
    """
    Simply converts grayscale MNIST to color and emits example, label, g triple.
    :param fracs: sets the number of examples of each g id.
    """
    idxs = np.random.permutation(np.arange(len(labels)))
    images, labels = images[idxs], labels[idxs]
    labels = (labels < 5).float()
    assert np.isclose(sum(fracs), 1)
    nums = [int(frac*len(images)) for frac in fracs]
    X, Y, gs = [], [], []
    num = 0
    for gi in range(3):
        gs += [gi]*nums[gi]
        _x, _y = images[num:num+nums[gi]], labels[num:num+nums[gi]]
        _im = torch.stack([_x, _x, _x], dim=1)
        if gi==0:
            colors = _y
        elif gi==1:
            colors = _y[np.random.permutation(np.arange(len(_y)))]
        elif gi==2:
            colors = 1-_y
        _im[torch.arange(len(_x)), colors.long(), :, :] *= 0
        X.append(_im)
        Y.append(_y)
            
    X, Y = torch.cat(X, dim=0), torch.cat(Y, dim=0)
    gs = torch.tensor(gs)
    return {
      'X': (X.float()),
      'y': Y[:, None],
      'g': gs
    }

class SpuMNISTDataset(WILDSDataset):
    def __init__(self, root_dir='data', download=False, split_scheme='official'):
        required_attrs = ['_dataset_name', '_data_dir',
                          '_split_scheme', '_split_array',
                          '_y_array', '_y_size',
                          '_metadata_fields', '_metadata_array']

        self._dataset_name = "rmnist"
        self._data_dir = os.path.join(root_dir, self._dataset_name)
        
        mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
        mnist_train = (mnist.data[:10000], mnist.targets[:10000])
        mnist_val = (mnist.data[10000:12000], mnist.targets[10000:12000])
        mnist = datasets.MNIST('~/datasets/mnist', train=False, download=True)
        mnist_test = (mnist.data, mnist.targets)
        train_data = make_environment(mnist_train[0], mnist_train[1], [0.49, 0.49, 0.02])
        val_data = make_environment(mnist_val[0], mnist_val[1], [0.34, 0.33, 0.33])
        test_data = make_environment(mnist_test[0], mnist_test[1], [0.34, 0.33, 0.33])
        
        _x_array, _y_array, _split_array, _g_array = [], [], [], []
        i = 0
        for di, d in enumerate([train_data, val_data, test_data]):
            x, y = d['X'], d['y']
            g = d['g']
            for j in range(len(y)):
                _x_array.append(np.transpose(x[j].numpy(), axes=[2, 1, 0]).astype(np.uint8))
                _y_array.append(y[j])
                _g_array.append(g[j])
            _split_array += [di]*len(y)
        
        _y_array = np.array(_y_array)
        _g_array = np.array(_g_array)
        self._input_array = _x_array
        self._y_array = torch.LongTensor(_y_array)
        self._split_array = np.array(_split_array)
        # partition the train in to val and test
        self._split_scheme = split_scheme
        self._y_size = 1
        self._n_classes = 2

        self._metadata_array = torch.stack(
            (torch.LongTensor(_g_array), self._y_array),
            dim=1
        )
        self._metadata_fields = ['group', 'y']
        self._metadata_map = {
            'group': [' grp0', ' grp1', ' grp2'], 
            'y': ['%d' for d in range(self._n_classes)]
        }
        
        self._original_resolution = (28, 28)
                
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['group']))
        self._metric = Accuracy()
        
        super().__init__(root_dir, download, split_scheme)
    
    def get_input(self, idx):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """
        gi = self._metadata_array[idx][0].item()
        pil_img = Image.fromarray(self._input_array[idx]).convert('RGB')
        return pil_img
        
    def eval(self, y_pred, y_true, metadata):
        return self.standard_group_eval(
            self._metric,
            self._eval_grouper,
            y_pred, y_true, metadata)
    
if __name__ == '__main__':
    dset = SpuMNISTDataset('data')
    train, val, test = dset.get_subset('train'), dset.get_subset('val'), dset.get_subset('test')
    print ("Train, val, test sizes:", len(train), len(val), len(test))
    for _ in range(500):
        dset.get_input(np.random.choice(len(dset)))