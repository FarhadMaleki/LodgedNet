import time
from models import DeepLodge, redefine_classifier
from runner import run_tfl
import os.path
import torch
import torchvision
import numpy as np
from utils import calculate_test_acc, calculate_tfl_test_acc
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Set dataset information
data_dir = "."
train_dir = os.path.join(data_dir, "data/wheat/RGB/train")
test_dir = os.path.join(data_dir, "data/wheat/RGB/test")
# crop must be either 'wheat' or 'canola'
crop = 'wheat'
batch_size = 16
num_epochs = 50
# inception_v3 requires input images to be 299 x 299
#'inception_v3': torchvision.models.inception_v3, 
models = {'alexnet': torchvision.models.alexnet,
          'vgg16': torchvision.models.vgg16,
          'vgg19': torchvision.models.vgg19,
          'resnet18': torchvision.models.resnet18,
          'resnet50': torchvision.models.resnet50,
          'resnet101': torchvision.models.resnet101,
          'densenet169': torchvision.models.densenet169,
          'densenet201': torchvision.models.densenet201,
          'squeezenet1_1': torchvision.models.squeezenet1_1}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
accuracy = {}
for name, model in models.items():
    logger.info('='*10)
    logger.info("Model: {}".format(name))
    # Model creation
    model_ft = model(pretrained=True)
    ###############################################################################
    ###############################################################################

    model_ft, loader = run_tfl(model_ft, train_dir, test_dir, crop, batch_size,
                               num_epochs)
    dataloaders = loader.dataloaders
    testloader = dataloaders["test"]
    test_acc = calculate_tfl_test_acc(model_ft, testloader, device)
    accuracy[name] = test_acc
    logger.info('\nTest Acc: {:4f}'.format(test_acc.item()))
with open('TransferLearningResults4{}.log'.format(crop), 'w') as fout:
    fout.write('{}\t{}\n'.format('Architecture', 'Accuracy'))
    for name, acc in accuracy.items():
        fout.write('{}\t{}\n'.format(name, acc))
