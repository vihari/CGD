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

# copied this routine from: https://github.com/facebookresearch/InvariantRiskMinimization/blob/master/code/colored_mnist/main.py
def make_environment(images, labels, fracs):
    """
    fracs: fraction of color correlated majority, medium and minority
    medium corresp[onds to thr group with no spurious correlations
    """
    idxs = np.random.permutation(np.arange(len(labels)))
    images, labels = images[idxs], labels[idxs]
    # Assign a binary label based on the digit; flip label with probability 0.25
    labels = (labels < 5).float()
    assert np.isclose(sum(fracs), 1)
    nums = [int(frac*len(images)) for frac in fracs]
    X, Y, gs = [], [], []
    num = 0
    for gi in range(2):
        gs += [gi]*nums[gi]
        _x, _y = images[num:num+nums[gi]], labels[num:num+nums[gi]]
        _im = torch.stack([_x, _x, _x], dim=1)
        if gi==0:
            colors = _y
        elif gi==1:
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

class CMNISTDDataset(WILDSDataset):
    def __init__(self, root_dir='data', download=False, split_scheme='official', group_ratio=100):
        required_attrs = ['_dataset_name', '_data_dir',
                          '_split_scheme', '_split_array',
                          '_y_array', '_y_size',
                          '_metadata_fields', '_metadata_array']

        self._dataset_name = "cmnist"
        self._data_dir = os.path.join(root_dir, self._dataset_name)
        
        mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
        mnist_train = (mnist.data[:50000], mnist.targets[:50000])
        mnist_val = (mnist.data[50000:], mnist.targets[50000:])
        mnist = datasets.MNIST('~/datasets/mnist', train=False, download=True)
        mnist_test = (mnist.data, mnist.targets)
        p = 1./(group_ratio+1)
        train_data = make_environment(mnist_train[0], mnist_train[1], [1-p, p])
        val_data = make_environment(mnist_val[0], mnist_val[1], [0.5, 0.5])
        test_data = make_environment(mnist_test[0], mnist_test[1], [0.5, 0.5])
        
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
            'group': [' majority', ' minority'], 
            'y': [' 0', '1']
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
        return Image.fromarray(self._input_array[idx]).convert('RGB')
        
    def eval(self, y_pred, y_true, metadata):
        return self.standard_group_eval(
            self._metric,
            self._eval_grouper,
            y_pred, y_true, metadata)
    
if __name__ == '__main__':
    dset = CMNISTDDataset('data')
    train, val, test = dset.get_subset('train'), dset.get_subset('val'), dset.get_subset('test')
    print ("Train, val, test sizes:", len(train), len(val), len(test))
