import torch.nn as nn
from torchvision.transforms import v2 as T
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim

import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LinearClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)



def run_linear_eval(encoder, data_dir, batch_size, num_workers, epochs, lr, weight_decay, output_dim):
    # 1) Freeze the encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # 2) Attach your linear probe
    classifier = LinearClassifier(feature_dim=output_dim, num_classes=len(os.listdir(os.path.join(data_dir, 'train'))))
    classifier = classifier.to(device)

    # 3) Data loaders (standard supervised transforms)
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

    # 4) Optimizer / loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=lr,
                          momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # 5) Training loop
    for epoch in range(1, epochs+1):
        # -- train classifier --
        classifier.train()
        total_loss, correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                feats = encoder(x)           # [B×feature_dim]
            logits = classifier(feats)      # [B×num_classes]
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

        scheduler.step()
        print(f"[Linear Eval] Epoch {epoch}/{epochs} — "
              f"Train {train_loss:.3f}/{train_acc:.1f}% — "
              f"Val {val_loss:.3f}/{val_acc:.1f}%")
