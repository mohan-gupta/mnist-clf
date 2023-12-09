import torch

import torchvision
import torchvision.transforms as transforms

import config

def get_loaders():
    train_dataset = torchvision.datasets.MNIST(root=config.DATA_PATH,
                                          train=True,
                                          transform = transforms.ToTensor(),
                                          download=True)
    
    test_dataset = torchvision.datasets.MNIST(root=config.DATA_PATH,
                                          train=False,
                                          transform = transforms.ToTensor(),
                                          )
    
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                          batch_size = config.TRAIN_BS,
                                          shuffle= True)
    
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = config.TEST_BS,
                                          shuffle= False)
    
    return {
        "test_loader": test_loader,
        "train_loader": train_loader
    }