import torch.nn as nn
from torchvision.transforms import v2 as T
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LinearClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def train(classifier, train_loader, criterion, optimizer, encoder):
    # -- train classifier --
        classifier.train()
        total_loss, correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                feats = encoder(x)           
            logits = classifier(feats)      
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            n += x.size(0)

        train_loss = total_loss / n
        train_acc  = 100. * correct / n
        
        return train_loss, train_acc


def evaluation(classifier, val_loader, criterion,encoder):
    # -- validate classifier --
    classifier.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            feats = encoder(x)
            logits = classifier(feats)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            n += x.size(0)

    val_loss = total_loss / n
    val_acc  = 100. * correct / n

    return val_loss, val_acc


def run_linear_eval(encoder, data_dir, batch_size, num_workers, epochs, lr, weight_decay, output_dim, args):

    # freeze the encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # attach the linear probe
    classifier = LinearClassifier(feature_dim=output_dim, num_classes=len(os.listdir(os.path.join(data_dir, 'train'))))
    classifier = classifier.to(device)

    
    transform_train = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    transform_val = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, 'test'),  transform=transform_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # Optimizer / loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=lr,
                          momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in range(1, epochs+1):
        train_loss, train_acc = train(classifier,train_loader,criterion,optimizer,encoder)
        val_loss, val_acc = evaluation(classifier,train_loader,criterion,optimizer,encoder)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        scheduler.step()
        print(f"[Linear Eval] Epoch {epoch}/{epochs} — "
              f"Train {train_loss:.3f}/{train_acc:.1f}% — "
              f"Val {val_loss:.3f}/{val_acc:.1f}%")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  

    # First plot: Training & Validation Loss
    axes[0].plot(np.arange(args.epochs), train_loss_list, color='blue', label='Train Loss')
    axes[0].plot(np.arange(args.epochs), val_loss_list, color='red', label='Validation Loss')
    axes[0].set_title('Train and Validation Loss (LR: {:.0e})'.format(args.lr))
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
    plt.savefig(f'train_val_loss_accuracy_{args.lr}.png')
    
    # Show the plots
    plt.show()

    plt.close()