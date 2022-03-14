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
from wilds.common.metrics.loss import Loss
from wilds.common.metrics.loss import ElementwiseLoss

def make_environment(num, frac, train=True):
    gs = torch.from_numpy(np.random.choice(3, p=frac, size=num))
    DIM = 1000
    X = torch.cat([torch.randn([num, 2]), torch.zeros([num, DIM-2])], dim=1)
    n = num
    Y1 = (X[:, 0] > 0).type(torch.float32)
    angle = 30*np.pi/180
    Y2 = ((np.cos(angle))*X[:, 0] + (np.sin(angle))*X[:, 1] > 0).type(torch.float32)
    Y3 = ((np.cos(2*angle))*X[:, 0] + (np.sin(2*angle))*X[:, 1] > 0).type(torch.float32)
    _Y = torch.where(gs==1, Y2, Y3)
    Y = torch.where(gs==0, Y1, _Y)
    print (X[:10], Y[:10], gs[:10])
    
    print ("label stats:", np.unique(Y.numpy(), return_counts=True))
    print ("g stats:", np.unique(gs.numpy(), return_counts=True))
    return {'X': X, 'y': Y, 'g': gs}

class RotSimpleDataset(WILDSDataset):
    def __init__(self, root_dir='data', download=False, split_scheme='official'):
        required_attrs = ['_dataset_name', '_data_dir',
                          '_split_scheme', '_split_array',
                          '_y_array', '_y_size',
                          '_metadata_fields', '_metadata_array']

        self._dataset_name = "rot_simple"
        self._data_dir = os.path.join(root_dir, self._dataset_name)

        train_data = make_environment(1000, [0.499, 0.499, 0.002], train=True)
        val_data = make_environment(1000, [0.34, 0.33, 0.33], train=False)
        test_data = make_environment(10000, [0.34, 0.33, 0.33], train=False)
        
        _x_array, _y_array, _split_array, _g_array = [], [], [], []
        i = 0
        for di, d in enumerate([train_data, val_data, test_data]):
            x, y = d['X'], d['y']
            g = d['g']
            for j in range(len(y)):
                _x_array.append(x[j])
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
            'group': ['       grp1', '       grp2', '       grp3'], 
            'y': [' 0', '1']
        }
                        
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['group']))
        self._metric = Loss(loss_fn=torch.nn.CrossEntropyLoss())
        
        super().__init__(root_dir, download, split_scheme)
    
    def get_input(self, idx):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """
        return self._input_array[idx]
        
    def eval(self, y_pred, y_true, metadata):
        return self.standard_group_eval(
            self._metric,
            self._eval_grouper,
            y_pred, y_true, metadata)
    
if __name__ == '__main__':
    dset = RotSimpleDataset('data')
    train, val, test = dset.get_subset('train'), dset.get_subset('val'), dset.get_subset('test')
    print ("Train, val, test sizes:", len(train), len(val), len(test))