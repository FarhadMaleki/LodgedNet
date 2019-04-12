"""The following code from compiled from Pytorh source code.
"""

import torch
from skimage import transform, io, img_as_float
import random
import numpy as np


class Rescale(object):
    """Rescale an image to a given size.

    Args:
        smallest (int): Rescale the image so that min(width, hight) == smallest,
        	where width, hight are width an height of the image before rescaling.
        	if height > width, then image will be rescaled to
            (output_size * height / width, output_size); otherwise, image will be
            rescaled to (output_size , output_size * width // height.
        	This transformation preserves the image asspect ration.
    """

    def __init__(self, smallest, **kwargs):
        assert isinstance(smallest, int)
        self.smallest = smallest
        self.kwargs = kwargs

    def __call__(self, image):
        h, w = image.shape[:2]
        if h > w:
            new_h, new_w = self.smallest * h / w, self.smallest
        else:
            new_h, new_w = self.smallest, self.smallest * w / h
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w), **self.kwargs)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomHorizontalFlip(object):
    """Horizontally flip the given skimage Image, of a sample,
    	randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        """
        Args:
            image (skimag Image): Image to be flipped.

        Returns:
            skimage Image: Randomly flipped image.
        """
        if random.random() < self.p:
            image = np.fliplr(image).copy()
        return image

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    """Vertically flip the given skimage Image, of a sample,
    	randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        """
        Args:
            image (skimag Image): Image to be flipped.

        Returns:
            skimage Image: Randomly flipped image.
        """
        if random.random() < self.p:
            image = np.flipud(image).copy()
        return image

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomCrop(object):
    """Randomly Crops the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. 
        	If int, an square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        assert new_h < h, 'The new heigh is larger than the inital height'
        assert new_w < w, 'The new width is larger than the inital width'
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top + new_h, left: left + new_w]
        return image

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, is_image=True):
        self.is_image = is_image

    def __call__(self, image):

        # swap color axis because since numpy image are H x W x C
        # but torch images are C X H X W
        if self.is_image is True:
            image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        return image

    def __repr__(self):
        return self.__class__.__name__ + '()'

class LabelToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, label):

        return torch.from_numpy(label)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)

        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
