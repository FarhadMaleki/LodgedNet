import tsfm
import torch
import os.path
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from lodgingdataset import LodgingDataset
from models import DeepLodge
from utils import train_model, train_tfl_model
from dataloader import LodgingDataLoader, LodgingImageLoader
from utils import get_transformations, get_stats


###############################################################################
###############################################################################
def run(model_ft, train_dir, test_dir, crop, num_channels, batch_size,
        num_epochs):
    # Get dataset stats
    dataset_mean, dataset_std = get_stats(crop)
    # parameters for GLCM texture feature extraction
    glcm_pars = {'distances': [1, 2, 4, 5],
                 'angles': [0, np.pi/2, 3*np.pi/4, np.pi],
                 'levels': 256,
                 'symmetric': False,
                 'normed': True,
                 'prop': 'contrast'
                 }
    # parameters for LBP texture feature extraction
    lbp_pars = {'radius': 1, 'num_points': 8, 'num_uniform_bins': 10,
                'num_var_bins': 16, 'hist_range': (0, 7000)}
    handcrafted = {'GLCM': glcm_pars, 'LBP': lbp_pars}
    # Set the image file extension
    file_name_pattern='.tif'
    # Define transformations
    train_tsfm, test_tsfm, feature_tsfm = get_transformations(num_channels, crop)
    # Create dataloader

    loader = LodgingDataLoader(train_dir, test_dir, dataset_mean, dataset_std,
                               batch_size, file_name_pattern, handcrafted,
                               train_tsfm, test_tsfm, feature_tsfm)
    dataloaders = loader.dataloaders
    dataset_sizes = loader.dataset_sizes
    # Determine devide type: GPU versus CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Building the network and training it
    ###########################################################################
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    ###########################################################################
    optimizer_ft = optim.Adam(model_ft.parameters())
    model_ft = train_model(model_ft, dataloaders, criterion, optimizer_ft,
                           num_epochs=num_epochs, device=device,
                           dataset_sizes=dataset_sizes)
    return model_ft, loader


def run_tfl(model_ft, train_dir, test_dir, crop, batch_size, num_epochs):
    # Get dataset stats
    dataset_mean, dataset_std = get_stats(crop)
    file_name_pattern='.tif'
    # Define transformations
    num_channels = 3

    train_tsfm = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=dataset_mean[:num_channels],
                                                          std=dataset_std[:num_channels])])

    test_tsfm = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=dataset_mean[:num_channels],
                                                         std=dataset_std[:num_channels])])
    # Create dataloader
    loader = LodgingImageLoader(train_dir, test_dir, dataset_mean, dataset_std,
                               batch_size, file_name_pattern, train_tsfm,
                               test_tsfm)
    dataloaders = loader.dataloaders
    dataset_sizes = loader.dataset_sizes
    # Determine devide type: GPU versus CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Building the network and training it
    ###########################################################################
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    ###########################################################################
    optimizer_ft = optim.Adam(model_ft.parameters())
    model_ft = train_tfl_model(model_ft, dataloaders, criterion, optimizer_ft,
                               num_epochs=num_epochs, device=device,
                               dataset_sizes=dataset_sizes)
    return model_ft, loader