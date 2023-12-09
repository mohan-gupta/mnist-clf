from tqdm import tqdm

import torch

import config

def accuracy_score(target, pred):
    target_len = target.shape[0]
    
    acc = (target == pred).sum()/target_len
    return acc

def train_one_step(model, batch, optimizer):
    optimizer.zero_grad()
    
    image, target = batch[0].to(config.DEVICE), batch[1].to(config.DEVICE)
    
    logit, loss = model(image = image, target = target)
    
    with torch.no_grad():
        probs = torch.softmax(logit, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        accuracy = accuracy_score(target.cpu().numpy(), preds.cpu().numpy())
    
    loss.backward()
    optimizer.step()
    
    return loss, accuracy

def train(model, data_loader, optimizer):
    model.train()
    
    num_batches = len(data_loader)
    train_loss = 0
    train_accuracy = 0
    loop = tqdm(data_loader, total=num_batches)
    
    for batch in loop:
        loss, accuracy = train_one_step(model, batch, optimizer)
        
        train_loss += loss.item()
        train_accuracy += accuracy
        
        loop.set_postfix({"loss": loss.item()})
    
    return train_loss/num_batches, train_accuracy/num_batches

def evaluate(model, data_loader):
    model.train()
    
    num_batches = len(data_loader)
    eval_loss = 0
    eval_acc = 0
    loop = tqdm(data_loader, total=num_batches)
    
    with torch.no_grad():
        for batch in loop:
            image, target = batch[0].to(config.DEVICE), batch[1].to(config.DEVICE)
        
            logits, loss = model(image = image, target = target)
            
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            accuracy = accuracy_score(target.cpu().numpy(), preds.cpu().numpy())
            
            eval_loss += loss.item()
            eval_acc += accuracy
            
            loop.set_postfix({"loss": loss.item()})
    
    return eval_loss/num_batches, eval_acc/num_batches