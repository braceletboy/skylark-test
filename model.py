'''
@file: run.py

This file contains all the models used in the experiments.

@author: Rukmangadh Sai Myana
@mail: rukman.sai@gmail.com
'''


import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryAugmentedCNN(nn.Module):
    '''
        Reference: https://proceedings.mlr.press/v48/santoro16.pdf

        Pytorch Implementation on found online. Implementing by referring to
        above paper.
    '''

    def __init__(self, args):
        '''Initialize

        Parameters
        ----------
        args
            Command line arguments
        '''
        self.args = args
        self.external_memory = torch.Tensor(
            self.args.num_classes, self.args.memory_embed
        )
        self.sigmoid_gate = nn.Parameter(torch.zeros(self.args.num_classes))
        self.num_reads = 0
        self.usage_weights = (
            torch.ones(self.args.num_classes)/self.args.num_classes
        )
        self.controller = Controller(self.args)
        self.output_layer = nn.Linear(
            self.args.hidden_embed + self.memory_embed,
            self.args.num_classes
        )

    def forward(self, inputs):
        '''Forward Pass
        '''
        hidden, keys = self.controller(inputs)
        read_weights = F.softmax(
            torch.matmul(keys, self.external_memory))
        if self.num_reads == 0:
            least_used_weights = torch.zeros(self.args.num_classes)
        else:
            least_used_weights, _ = (
                self.usage_weights <= torch.kthvalue(
                    self.usage_weights, self.num_reads)
                )
        write_weights = (
            F.sigmoid(self.sigmoid_gate) * read_weights +
            (1 - F.sigmoid(self.sigmoid_gate)) * least_used_weights
        )
        read = torch.matmul(read_weights, self.external_memory)
        self.num_reads += 1
        self.usage_weights = (
            self.args.gamma * self.usage_weights +
            read_weights + write_weights
        )
        self.external_memory = self.external_memory + torch.matmul(
            write_weights, self.external_memory)
        outputs = self.output_layer(torch.concat(hidden, read))
        return outputs


class Controller(nn.Module):
    '''The controller of the Memory augmented CNN'''

    def __init__(self, args):
        '''Initialize

        Parameters
        ----------
        args
            Command line arguments
        '''
        self.args = args
        self.conv_network = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU()  # in recent papers GeLU is preferred to RELU
                )
                for (in_channels, out_channels, kernel_size) in
                self.args.cnn_config
            ]
        )
        self.head = nn.Linear(self.args.conv_embed, self.args.num_classes)

    def forward(self, inputs):
        '''Forward Pass
        '''
        hidden = self.conv_network(inputs)
        keys = self.head(inputs)
        return hidden, keys
