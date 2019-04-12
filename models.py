""" This module contains the modules used in this research.

"""
import torch
import torch.nn as nn
from  torch.nn import functional as F


class DeepLodge(nn.Module):
    """ Build the model using handcrafted features.

    Args:
        feature_size (int): The length of handcrafted features.
        num_channels (int): Number of image channels, e.g. 3 for RGB.

    """
    def __init__(self, feature_size=None, num_channels=3):
        super(DeepLodge, self).__init__()
        image_dim=(64, 128)
        self.cnn_feature_size = int(image_dim[0] * image_dim[1] / 256)
        self.feature_size = feature_size
        self.num_channels = num_channels
        self.conv1_1 = nn.Conv2d(self.num_channels, 16, 3, padding=1)
        self.conv1_2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv2_1 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv2_1_2 = nn.Conv2d(32, 32, 3, padding=1)

        self.conv2_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv2_3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv2_4 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc_1 = nn.Linear(self.cnn_feature_size * 64 + feature_size * num_channels, 128)
        self.fc_2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.5)
        self.dropout2d = nn.Dropout2d(0.5)

    def forward(self, img, feature):
        """ Implement model inference.

        Args:
            img (numpy.ndarray): An input image.
            features (numpy.array): A one-dimensional array representing
                handcrafted features.

        Returns:
            The inference vector for the input.

        """
        img = F.relu(self.conv1_1(img))
        img = F.relu(self.conv1_2(img))
        img = self.dropout2d(img)
        img = F.max_pool2d(img, 2, 2)
        img = F.relu(self.conv2_1(img))
        img = F.relu(self.conv2_1_2(img))
        img = self.dropout2d(img)
        img = F.max_pool2d(img, 2, 2)
        img = F.relu(self.conv2_2(img))
        img = self.dropout2d(img)
        img = F.relu(self.conv2_3(img))
        img = F.max_pool2d(img, 2, 2)
        img = self.dropout2d(img)
        img = F.relu(self.conv2_4(img))
        img = self.dropout2d(img)
        img = F.max_pool2d(img, 2, 2)
        img = img.view(-1, self.cnn_feature_size * 64)
        feature = feature.view(-1, self.num_channels * self.feature_size)
        img = torch.cat((img, feature), dim=1)
        img = self.dropout(img)
        img = F.relu(self.fc_1(img))
        img = self.dropout(img)
        return self.fc_2(img)


def redefine_classifier(model, transfer=True):
    """Replace the classifier of the trained network with a new classifier.

    Args:
        model: A network that contains a classifier attribute. The classifier
            attribute must contain a in_features attribute, which is the number
            features resulted from the CNN component of the model, i.e. the input
            to the fully connected component (i.e. the classifier) of the model.

    Returns:
        A model that only its classifier will be trainable, i.e. the
            requires_grad is true only for the classifier parameters.

    Raises:
        AttributeError: If model does not have a classifier attribute or if the
            classifier attribute does not have in_features as its attribute.

    """
    try:
        if transfer is True:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(p=0.5),
                                         nn.Linear(num_features, 128),
                                         nn.ReLU(),
                                         nn.Dropout(p=0.5),
                                         nn.Linear(128, 2))
    except AttributeError:
        msg = ('model must have a classifier attribute that has in_features'
               ' as its attribute.')
        raise AttributeError(msg)
