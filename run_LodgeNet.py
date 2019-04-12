import time
import numpy as np
from models import DeepLodge, redefine_classifier
from runner import run
import os.path
import torch
from utils import calculate_test_acc
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# crop must be either 'wheat' or 'canola'
crop = 'wheat'
# num_channels must be either 3 or 5
num_channels = 3
if num_channels == 3:
    data_format = 'RGB'
else:
    data_format = 'BGRRededgeNIR'
# Set dataset information
data_dir = "."
train_dir = os.path.join(data_dir, "data/{}/{}/train".format(crop, data_format))
test_dir = os.path.join(data_dir, "data/{}/{}/test".format(crop, data_format))
batch_size = 16
num_epochs = 50
# Model creation
model_ft = DeepLodge(feature_size=42, num_channels=num_channels)
###############################################################################
num_params = sum(p.numel() for p in model_ft.parameters())
logger.info('Number of parameters: {}'.format(num_params))
###############################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft, loader = run(model_ft, train_dir, test_dir, crop, num_channels,
                            batch_size, num_epochs)

dataloaders = loader.dataloaders
testloader = dataloaders["test"]
test_acc = calculate_test_acc(model_ft, testloader, device)
logger.info('\nTest Acc: {:4f}'.format(test_acc.item()))
