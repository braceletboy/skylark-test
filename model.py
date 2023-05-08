'''
@file: run.py

This file contains all the models used in the experiments.

The following is a legend of the short forms used in this file:

    N - batch size
    C - channels
    H, H' - height
    W, W' - width
    R - no. of reads
    Wr - no. of writes
    E - no. of memory entries
    M - memory entry size
    Cl - no. of classes

@author: Rukmangadh Sai Myana
@mail: rukman.sai@gmail.com
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class MemoryAugmentedCNN(nn.Module):
    '''
        Memory Augmented Convolutional Neural Network for Continual Learning.

        I adapted the neural network from the following paper:

        ["One-shot Learning with Memory-Augmented Neural Networks"][1]

        Pytorch and tensorflow Implementation for the above paper was not found
        online. Hence, I am implementing from scratch by referring to the
        paper.

        [1]: https://arxiv.org/pdf/1605.06065.pdf
    '''

    def __init__(self, args):
        '''Initialize

        Parameters
        ----------
        args
            Command line arguments
        '''
        self.args = args
        self.num_reads = args.read_heads
        self.num_writes = args.write_heads
        self.memory_size = args.memory_embed
        self.gamma = args.gamma

        # components
        self.controller = Controller(
            args.conv_config, self.num_reads, self.num_writes,
            args.feat_size, self.memory_size
        )
        self.external_memory = torch.Tensor(
            self.args.num_entries, self.memory_size  # E,M
        )
        self.sigmoid_gate = nn.Parameter(
            torch.zeros(self.args.num_entries)  # E
        )

        self.conv_size = args.feat_size
        self.conv_channels = args.conv_config[-1][1]
        self.num_conv_features = (
            self.conv_size[0]*self.conv_size[1]*self.conv_channels
        )
        self.output_layer = nn.Linear(
            self.num_conv_features + self.num_reads*self.memory_size,
            self.args.num_classes
        )

        self._initialize_weights()  # initialize the attention weights

    def _initialize_weights(self,):
        '''Initialize the previous usage, read and least usage weights'''
        self.prev_read_weights = (
            torch.ones(self.args.num_entries)/self.args.num_entries  # uniform
        )
        self.prev_usage_weights = None
        self.prev_least_used_weights = None

    def forward(self, inputs) -> torch.Tensor:
        '''Forward Pass

        Parameters
        ----------
        inputs
            Input Image Batch

        Returns
        -------
        torch.Tensor
            The logits for the given batch of classification inputs
        '''
        # controller forward - N,CHW | N,R,M | N,Wr,M
        features, read_keys, write_keys = self.controller(inputs)

        # reading - N,R,M
        read_vectors, current_read_weights = self.read(read_keys)
        read_vectors = read_vectors.flatten(start_dim=1)  # N,RM

        # writing
        if not self.args.no_write:
            current_usage_weights, current_least_used_weights = self.write(
                write_keys, current_read_weights)

        # classification output - N,Cl
        logits = self.output_layer(torch.cat((features, read_vectors), dim=-1))

        self.prev_read_weights = current_read_weights
        self.prev_usage_weights = current_usage_weights
        self.prev_least_used_weights = current_least_used_weights

        return logits

    def read(self, read_keys: torch.Tensor) -> Tuple[torch.Tensor]:
        '''Perform all read operations on the memory and return read vectors

        Parameters
        ----------
        read_keys
            The keys from the read heads for reading from the memory


        Returns
        -------
        Tuple[torch.Tensor]
            The vectors read by the heads and the read weights as a tuple
        '''
        # quickly calculate attentions and read vectors
        attentions = torch.einsum(
            'nrm,tm->nrt', read_keys, self.external_memory  # N,R,E
        )
        read_weights = F.softmax(attentions, dim=-1)  # N,R,E
        read_vectors = torch.einsum(
            'nrt,tm->nrm', read_weights, self.external_memory  # N,R,M
        )
        return read_vectors, read_weights

    def write(
        self,
        write_keys: torch.Tensor,
        current_read_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        '''Perform a write operation on the memory

        Parameters
        ----------
        write_keys
            The keys for writing to the memory
        current_read_weights
            The reading weights for the current sample

        Returns
        -------
        Tuple[torch.Tensor]
            The usage weights and least used weights
        '''
        if self.prev_least_used_weights is not None:
            current_write_weights = (
                F.sigmoid(self.sigmoid_gate)*self.prev_read_weights +
                (1-F.sigmoid(self.sigmoid_gate))*self.prev_least_used_weights
            )  # N,R,E
        else:
            current_write_weights = self.prev_read_weights  # N,R,E

        # write to memory - here I am averaging the write weights along the R
        # dimension because it was not specified in the paper what to do with
        # the write weights when we have multiple read heads
        actual_write_weights = current_write_weights.mean(dim=1)  # N,E
        write_vectors = torch.einsum(
            'nt,nwm->nwtm', actual_write_weights, write_keys  # N,Wr,E,M
        )

        # E,M - we sum over the write and batch dimensions because the update
        # needs to be done for each write head and for each sample in the batch
        write_matrix = write_vectors.sum(dim=0).sum(dim=0)
        self.external_memory = self.external_memory + write_matrix

        # calcuate usage and least used weights
        if self.prev_usage_weights is not None:
            current_usage_weights = (
                self.gamma*self.prev_usage_weights +
                current_read_weights + current_write_weights  # N,R,E
            )
        else:
            current_usage_weights = (
                current_read_weights + current_write_weights  # N,R,E
            )
        ksmallest = torch.kthvalue(
            current_usage_weights, self.num_reads, dim=-1  # scalar
        )
        current_least_used_weights = (
            current_usage_weights <= ksmallest).int()  # N,R,E
        return current_usage_weights, current_least_used_weights


class Controller(nn.Module):
    '''The controller of the Memory augmented CNN'''

    def __init__(
        self,
        conv_config: List[Tuple[int]],
        num_reads: int,
        num_writes: int,
        feat_size: Tuple[int],
        memory_size: int,

    ) -> None:
        '''Initialize

        Parameters
        ----------
        conv_config
            The configuration of the convolutional layers
        num_reads
            The number of read heads
        num_writes
            The number of write heads
        feat_size
            The size of the features outputted by the convolution layers
        memory_size
            The size of the memory entries
        '''
        self.conv_config = conv_config
        self.conv_network = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU()  # in recent papers GeLU is preferred to RELU
                )
                for (in_channels, out_channels, kernel_size) in conv_config
            ]
        )

        # calculate output feature size
        self.feat_size = feat_size
        self.feat_channels = conv_config[-1][1]
        self.num_features = (
            self.feat_size[0]*self.feat_size[1]*self.feat_channels
        )

        self.read_heads = nn.ModuleList([
            nn.Linear(self.num_features, memory_size)for _ in range(num_reads)
        ])

        self.write_heads = nn.ModuleList([
            nn.Linear(self.num_features, memory_size)
            for _ in range(num_writes)])

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor]:
        '''Forward Pass

        Paramters
        ---------
        inputs
            Image input batch (NxCxH'xW')

        Returns
        -------
        Tuple[List[torch.Tensor], List[torch.Tensor]]
            A tuple of the extracted features and read keys
        '''
        features = self.conv_network(inputs)  # N,C,H,W
        features = features.flatten(start_dim=1)  # N,CHW

        read_keys = []
        for head in self.read_heads:
            key = head(features)  # N,M
            read_keys.append(key)
        read_keys = torch.stack(read_keys, dim=1)  # N,R,M

        write_keys = []
        for head in self.write_heads:
            key = head(features)  # N,M
            write_keys.append(key)
        write_keys = torch.stack(write_keys, dim=1)  # N,Wr,M

        return features, read_keys, write_keys
