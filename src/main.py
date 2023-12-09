import torch

import config
from dataset import get_loaders
from model import MnistModel
from engine import train, evaluate

def save_checkpoint(model_state, optimizer_state, val_loss, epoch):
    checkpoint = {
        'model': model_state,
        'optimizer': optimizer_state,
        'val_loss': val_loss,
        'epoch': epoch
    }
    
    torch.save(checkpoint, config.MODEL_PATH)

def main():
    loaders = get_loaders()       
    
    model = MnistModel(config.NUM_CLASSES)
    model.to(config.DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = config.LR)
    
    for epoch in range(config.EPOCHS):
        print(f"Epoch: {epoch+1}")
        
        train_loss, train_acc = train(model, loaders['train_loader'], optimizer)
        val_loss, val_acc = evaluate(model, loaders['test_loader'])
        
        print(f"Train Loss: {train_loss} and Val Loss: {val_loss}")
        print(f"Train Accuracy: {train_acc:.3f} and Val Accuracy: {val_acc:.3f}")
        
    save_checkpoint(
        model.state_dict(),
        optimizer.state_dict(),
        val_loss,
        config.EPOCHS
        )
    print("model saved")
    
    
if __name__ == "__main__":
    main()
