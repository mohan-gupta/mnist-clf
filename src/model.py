import torch
import torch.nn as nn
import torch.nn.functional as F

class MnistModel(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 16, (3,3), padding=(1,1))
        self.maxpool1 = nn.MaxPool2d(2,2)
        
        self.conv2 = nn.Conv2d(16, 32, (3,3), padding=(1,1))
        self.maxpool2 = nn.MaxPool2d(2,2)
        
        self.conv3 = nn.Conv2d(32, 64, (3,3), padding=(1,1))
        self.maxpool3 = nn.MaxPool2d(2,2)
        
        self.dropout = nn.Dropout(0.3)
        
        self.linear = nn.Linear(576, 64)
        self.lin_dropout = nn.Dropout(0.3)
        self.output = nn.Linear(64, num_classes)
        
    def forward(self, image, target=None):
        bs, _, _, _ = image.size()
        x = F.relu(self.conv1(image))
        x = self.maxpool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.maxpool3(x)
        
        x = x.view(bs, -1)
        
        x = self.dropout(x)
        
        x = F.relu(self.linear(x))
        x = self.lin_dropout(x)
        out = self.output(x)
        
        if target is not None:
            loss = nn.CrossEntropyLoss()(out, target)
            
            return out, loss
        
        return out, None

if __name__ == "__main__":
    image = torch.randn((2, 1, 28, 28))
    label = torch.tensor([9, 3], dtype=torch.long)
    model = MnistModel(10)
    
    out, loss = model(image, label)
    print(loss)
    