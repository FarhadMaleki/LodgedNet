import tsfm
import torch
import os.path
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from lodgingdataset import LodgingDataset
from utils import load_train_valid_test_datasets


class LodgingDataLoader(object):
    def __init__(self, train_dir, test_dir, dataset_mean,
                 dataset_std, batch_size, file_name_pattern, handcrafted,
                 train_tsfm, test_tsfm, feature_tsfm):
        # Get all data for training; this include validation data too
        train_data = LodgingDataset(train_dir, file_name_pattern, handcrafted,
                                    transform=train_tsfm,
                                    feature_transform=feature_tsfm)
        # split train dataset into train_dataset and valid_dataset
        random_indices = np.random.permutation(len(train_data))
        train_ratio, valid_ratio = 0.80, 0.20
        n = int(train_ratio*len(train_data))
        train_indices = random_indices[:n]
        valid_indices = random_indices[n:]
        # Split the dataset to training and validation datasets
        train_dataset = torch.utils.data.Subset(train_data, train_indices)
        valid_dataset = torch.utils.data.Subset(train_data, valid_indices)
        # get test dataset
        test_dataset = LodgingDataset(test_dir, file_name_pattern, handcrafted,
                                      transform=test_tsfm,
                                      feature_transform=feature_tsfm)
        self._dataloaders = load_train_valid_test_datasets(train_dataset,
                                                           valid_dataset,
                                                           test_dataset,
                                                           batch_size,
                                                           num_workers=0)
        self._dataset_sizes = {'train': len(train_dataset),
                               'valid': len(valid_dataset),
                               'test': len(test_dataset)}

    @property
    def dataloaders(self):
        return self._dataloaders

    @property
    def dataset_sizes(self):
        return self._dataset_sizes


class LodgingImageLoader(object):
    def __init__(self, train_dir, test_dir, dataset_mean, dataset_std,
                 batch_size, file_name_pattern, train_tsfm, test_tsfm):
        # Get all data for training; this include validation data too
        train_data = torchvision.datasets.ImageFolder(train_dir,
                                                      transform=train_tsfm)
        # split train dataset into train_dataset and valid_dataset
        random_indices = np.random.permutation(len(train_data))
        train_ratio, valid_ratio = 0.80, 0.20
        n = int(train_ratio*len(train_data))
        train_indices = random_indices[:n]
        valid_indices = random_indices[n:]
        # Split the dataset to training and validation datasets
        train_dataset = torch.utils.data.Subset(train_data, train_indices)
        valid_dataset = torch.utils.data.Subset(train_data, valid_indices)
        # get test dataset
        test_dataset = torchvision.datasets.ImageFolder(test_dir,
                                                        transform=test_tsfm)
        self._dataloaders = load_train_valid_test_datasets(train_dataset,
                                                           valid_dataset,
                                                           test_dataset,
                                                           batch_size,
                                                           num_workers=0)
        self._dataset_sizes = {'train': len(train_dataset),
                               'valid': len(valid_dataset),
                               'test': len(test_dataset)}

    @property
    def dataloaders(self):
        return self._dataloaders

    @property
    def dataset_sizes(self):
        return self._dataset_sizes