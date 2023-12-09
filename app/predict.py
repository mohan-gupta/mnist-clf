import torch
import torchvision.transforms as transforms

from app.model import MnistModel

import os

DIR = os.path.dirname(__file__)

def get_model():
    checkpoint = torch.load(os.path.join(DIR, "../model/checkpoint.bin"), map_location="cpu")
    net = MnistModel(10)
    net.load_state_dict(checkpoint['model'])
    
    return net

def predict(net, image):
    net.eval()
    img_tnsr = transforms.ToTensor()(image)
    img_tnsr = img_tnsr.unsqueeze(0)
    
    logits, _ = net(img_tnsr)
    probs = torch.softmax(logits, dim=-1)
    pred = torch.argmax(probs, dim=-1)
    
    return pred.item()