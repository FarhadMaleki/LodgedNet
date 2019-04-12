""" This module contains utility functions.
"""
import time
import copy
import torch
import numpy as np
import skimage.util as util
from skimage.feature import greycomatrix, greycoprops
from skimage.feature import local_binary_pattern
from torchvision import transforms
import tsfm
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# This method was implemented by Sara Mardanisamani 
def glcm_feature(image, distances, angles, levels, symmetric, normed, prop):
    """ Extract GLCM features from an image.

    Args:
        image: A 3D-array representing an image. image must be represented
            as channel last.

    Returns:
        numpy.array: A one dimensionality array representing GLCM features.

    """
    try:
        feature = []
        channel_feature_size = len(distances) * len(angles)
        num_channels = image.shape[2]
        for i in range(num_channels):
            img = image[:, :, i]
            img = util.img_as_ubyte(img, force_copy=True)
            # create Concurrence matrix
            glcm = greycomatrix(img, distances=distances, angles=angles,
                                levels=levels, symmetric=symmetric,
                                normed=normed)

            stats = greycoprops(glcm, prop=prop)
            channel_glcm = np.reshape(stats, (channel_feature_size, ))
            feature.extend(channel_glcm)
    except:
        logger.error(np.max(img))
        raise
    return feature


# This method was implemented by Sara Mardanisamani 
def lbp_feature(image, radius, num_points, num_uniform_bins, num_var_bins,
                hist_range):
    """ Extract LBP features from an image.

    Args:
        image: A 3D-array representing an image. image must be represented
            as channel last.

    Returns:
        numpy.array: A one dimensionality array representing LBP features.

    """
    feature = []
    num_channels = image.shape[2]
    for i in range(num_channels):
        img = image[:, :, i]
        img = util.img_as_ubyte(img, force_copy=True)
        # Create Local Binary Pattern features
        lbp = local_binary_pattern(img, num_points, radius, method='uniform')
        hist, _ = np.histogram(lbp, bins=num_uniform_bins)
        feature.extend(hist)
        lbp_var = local_binary_pattern(img, num_points, radius, method='var')
        hist_var, _ = np.histogram(lbp_var, bins=num_var_bins,
                                   range=hist_range)
        feature.extend(hist_var)
    return feature


def train_tfl_model(model, dataloaders, criterion, optimizer, num_epochs, device,
                    dataset_sizes):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for img, labels in dataloaders[phase]:
                img = img.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.int64)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(img)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * img.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train_model(model, dataloaders, criterion, optimizer, num_epochs, device,
                dataset_sizes):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for img, feature, labels in dataloaders[phase]:
                img = img.to(device, dtype=torch.float)
                feature = feature.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.int64)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(img, feature)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * img.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def get_stats(crop):
    assert crop in {'wheat', 'canola'}
    WHEAT_MEAN = [0.326, 0.385, 0.339, 0.504, 0.424]
    WHEAT_STD = [0.127, 0.139, 0.126, 0.156, 0.146]
    CANOLA_MEAN = [0.302, 0.356, 0.288, 0.498, 0.424]
    CANOLA_STD = [0.113, 0.143, 0.121, 0.158, 0.159]
    if crop == 'wheat':
        return copy.copy(WHEAT_MEAN), copy.copy(WHEAT_STD)
    if crop == 'canola':
        return copy.copy(CANOLA_MEAN), copy.copy(CANOLA_STD)


# Define transformations
def get_transformations(num_channels, crop):
    assert num_channels in {3, 5}
    assert crop in {'canola', 'wheat'}
    dataset_mean, dataset_std = get_stats(crop)
    initial_tsfm = []
    scale_factor = 70
    dim = (64, 128)
    if crop == 'wheat':
        initial_tsfm = [tsfm.Rescale(scale_factor, anti_aliasing=True,
                                     mode='constant')]
    # Define transformations
    train_tsfm = tsfm.Compose(initial_tsfm +
                              [tsfm.RandomHorizontalFlip(),
                               tsfm.RandomVerticalFlip(),
                               tsfm.RandomCrop(dim),
                               tsfm.ToTensor(),
                               transforms.Normalize(mean=dataset_mean,
                                                    std=dataset_std)])
    test_tsfm = tsfm.Compose(initial_tsfm +
                             [tsfm.RandomCrop(dim),
                              tsfm.ToTensor(),
                              transforms.Normalize(mean=dataset_mean,
                                                   std=dataset_std)])
    feature_tsfm = tsfm.ToTensor(is_image=False)
    return train_tsfm, test_tsfm, feature_tsfm

def calculate_tfl_test_acc(model_ft, testloader, device):
    corrects = 0
    total = 0
    time_elapsed = 0
    operation_times = []
    # Iterate over data.
    for img, labels in testloader:
        img = img.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.int64)
        with torch.set_grad_enabled(False):
            since = time.time()
            outputs = model_ft(img)
            _, preds = torch.max(outputs, 1)
            operation_times.append(time.time() - since)
        corrects += torch.sum(preds == labels.data)
        total += labels.size(0)
    detection_time = np.mean(operation_times)
    detection_time_sd = np.std(operation_times)
    msg = ('Detection time in millisecond for each image in test set '
           'Mean: {:10.9f}\t STD: {}')
    logger.info(msg.format(detection_time * 1000, detection_time_sd * 1000))
    test_acc = corrects.double() / total
    return test_acc


def calculate_test_acc(model_ft, testloader, device):
    corrects = 0
    total = 0
    time_elapsed = 0
    operation_times = []
    # Iterate over data.
    for img, feature, labels in testloader:
        img = img.to(device, dtype=torch.float)
        feature = feature.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.int64)
        with torch.set_grad_enabled(False):
            since = time.time()
            outputs = model_ft(img, feature)
            _, preds = torch.max(outputs, 1)
            operation_times.append(time.time() - since)
        corrects += torch.sum(preds == labels.data)
        total += labels.size(0)
    detection_time = np.mean(operation_times)
    detection_time_sd = np.std(operation_times)
    msg = ('Detection time in millisecond for each image in test set '
           'Mean: {:10.9f}\t STD: {}')
    logger.info(msg.format(detection_time * 1000, detection_time_sd * 1000))
    test_acc = corrects.double() /total
    return test_acc


def load_train_valid_test_datasets(train_dataset, valid_dataset, test_dataset,
                                   batch_size, num_workers=0):
    dataloaders = {}
    dataloaders["train"] = torch.utils.data.DataLoader(train_dataset,
                                                       shuffle=True,
                                                       batch_size=batch_size,
                                                       num_workers=num_workers)
    dataloaders["valid"] = torch.utils.data.DataLoader(valid_dataset,
                                                       shuffle=True,
                                                       batch_size=batch_size,
                                                       num_workers=num_workers)
    dataloaders["test"] = torch.utils.data.DataLoader(test_dataset,
                                                      shuffle=False,
                                                      batch_size=batch_size,
                                                      num_workers=num_workers)

    return dataloaders
