import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import datasets, transforms, models, v2
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
import math

import matplotlib.pyplot as plt
from nt_xtent import NT_Xent
from simclr import SimCLR
from contrastive_dataset import ContrastiveDataset, get_contrastive_dataloaders
import numpy as np
from torchvision.transforms import v2

def parse_args():
    parser = argparse.ArgumentParser(description='Supervised training')
    parser.add_argument('--data-dir', type=str, required=True, help='path to training data')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--output-dir', type=str, default="./output")
    return parser.parse_args()










    

def build_model(num_classes):
    # Initialize ResNet-18 from scratch (no pretrained weights)
    model = models.resnet18(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def train(model, train_loader, criterion, optimizer):
    num_images = 0
    num_correct = 0
    total_loss = 0
    model.train()
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs,labels)

        # clearing old gradients from the last step
        optimizer.zero_grad()
        # computing the derivative of the loss w.r.t. the parameters
        loss.backward()
        # optimizer takes a step in updating parameters based on the gradients of the parameters.
        optimizer.step()
        
        total_loss += loss.item()
        num_correct += torch.sum(torch.argmax(outputs, dim=1) == labels.data).item()
        num_images += inputs.size(0)
        
    classification_acc = num_correct / float(num_images) *100
    total_loss = total_loss / num_images
    return total_loss, classification_acc

def evaluation(model, validation_loader,criterion):
    total_loss = 0
    num_correct = 0
    num_images = 0
    model.eval()
    for inputs, labels in validation_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        preds = outputs.argmax(dim=1)
        num_correct += (preds == labels).sum().item()
        num_images += inputs.size(0)

    classification_acc = num_correct / float(num_images) * 100
    total_loss = total_loss / num_images
    return total_loss, classification_acc


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    train_loader, validation_loader = get_dataloaders(trans)

    model = build_model(num_classes=0).to(device)

    # Cross-entropy loss for supervised training
    criterion = nn.CrossEntropyLoss()
    # SGD optimizer initialized for from-scratch training
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    
    
    # Step LR scheduler: reduce lr every 50 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
 
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = evaluation(model, validation_loader, criterion, device)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        scheduler.step()

        print(f'Epoch {epoch}/{args.epochs} '  \
              f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '  \
              f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))  

    # First plot: Training & Validation Loss
    axes[0].plot(np.arange(args.epochs), train_loss_list, color='blue', label='Train Loss')
    axes[0].plot(np.arange(args.epochs), val_loss_list, color='red', label='Validation Loss')
    axes[0].set_title('Train and Validation Loss (LR: {:.0e})'.format(LR))
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Second plot: Validation Accuracy
    axes[1].plot(np.arange(args.epochs), train_acc_list, color='blue', label='Train Accuracy')
    axes[1].plot(np.arange(args.epochs), val_acc_list, color='red', label='Validation Accuracy')
    axes[1].set_title('Train and Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # Save the figure
    plt.savefig(f'train_val_loss_accuracy_{LR}_{description}.png')
    
    # Show the plots
    plt.show()






if __name__ == '__main__':
    run_simCLR()
