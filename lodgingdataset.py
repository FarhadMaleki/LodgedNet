""" This module implement the dataset class for reading image data
"""
import os
import os.path
import numpy as np
from torch.utils.data import Dataset
from skimage import io, img_as_float
from utils import glcm_feature, lbp_feature


class LodgingDataset(Dataset):
    """Lodging dataset."""

    def __init__(self, data_dir, file_name_pattern, handcrafted,
                 transform=None, feature_transform=None):
        """
        Args:
            data_dir (string): Directory a set of subdirectories
                each containing images for one class of dataset.
                file_name_pattern (string): The file extension used for
                    selecting images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        glcm_pars = handcrafted['GLCM']
        lbp_pars = handcrafted['LBP']
        self.transform = transform
        self.feature_transform = feature_transform
        self.data_dir = data_dir
        classes = sorted([d for d in os.listdir(self.data_dir)
                          if (os.path.isdir(os.path.join(self.data_dir, d)) and
                              not d.startswith('.'))])
        self.cat2id = {c: i for i, c in enumerate(classes)}
        self.id2cat = {i: c for i, c in enumerate(classes)}
        self.image_pths = []
        self.catids = []
        self.class_size = {}
        self.features = []
        for i, cat in enumerate(classes):
            cat_dir = os.path.join(self.data_dir, cat)
            img_adrses = sorted([os.path.join(self.data_dir, cat, img_adrs)
                                 for img_adrs in os.listdir(cat_dir)
                                 if (os.path.isfile(os.path.join(cat_dir,
                                                                 img_adrs)) and
                                     img_adrs.endswith(file_name_pattern))])
            self.class_size[cat] = len(img_adrses)
            self.catids.extend([i] * self.class_size[cat])
            self.image_pths.extend(sorted(img_adrses))
            for adrs in sorted(img_adrses):
                img = io.imread(adrs)
                glcm = glcm_feature(img, **glcm_pars)
                lbp = lbp_feature(img, **lbp_pars)
                self.features.append(np.concatenate([glcm, lbp], axis=0))
        msg = '{} directory has no file with {} as file name extension.'
        for c, i in self.cat2id.items():
            error_msg = msg.format(c, file_name_pattern)
            assert self.class_size.get(c, 0) > 0, error_msg
        self.num_cats = len(self.cat2id)

    def __len__(self):
        return len(self.image_pths)

    def __getitem__(self, idx):
        img_name = self.image_pths[idx]
        image = io.imread(img_name)
        image = img_as_float(image)
        catid = self.catids[idx]
        # Apply transormations
        if self.feature_transform is not None:
            feature = self.feature_transform(self.features[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, feature, catid
