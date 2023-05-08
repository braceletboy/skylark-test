'''
@file: sampler.py

This file contains the various custom samplers required for performing the
experiments.

@author: Rukmangadh Sai Myana
@mail: rukman.sai@gmail.com
'''

import torch


class MyMnistSampler(torch.utils.data.sampler.Sampler):
    '''
    Sampler for selecting appropriate classes from the mnist dataset. A sampler
    is supposed to provide the indices of the dataset that need to be sampled
    through it's __iter__ method.
    '''

    def __init__(self, labels, data_source):
        '''
        Initialize the object with the given parameters.

        @param labels: The labels that are to be recognized for sampling.
        @param data_source: The dataset.
        '''
        self.mask = torch.tensor([1 if data_source[idx][1] in labels else 0 for
                                  idx in range(len(data_source))])
        self.data_source = data_source

    def __iter__(self):
        return iter([idx.item() for idx in torch.nonzero(self.mask,
                                                         as_tuple=False)])

    def __len__(self):
        return len(self.data_source)
