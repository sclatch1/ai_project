import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import datasets, transforms, models
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
import math

import matplotlib.pyplot as plt
from nt_xtent import NT_Xent
from contrastive_dataset import ContrastiveDataset, get_contrastive_dataloaders
import numpy as np
from torchvision.transforms import v2 as T
from linear_classifier import run_linear_eval
from supconresnet import build_supconresnet
from loss import SupConLoss
from spp import build_model_spp

#import tensorboard_logger as tb_logger


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    parser.add_argument("--method", type=str, default="SupCon", choices=['SupCon', 'SimCLR'], help='choose between SupCon and SimCLR')

    args = parser.parse_args()
    return args





def random_crop():
    """
    Random crop and resize to 224x224 We use standard Inception-style random cropping (Szegedy et al., 2015). 
    The crop of random size (uniform from 0.08 to 1.0 in area) of the original size and a random aspect ratio 
    (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop is finally resized to the original size.
    Additionally, the random crop (with resize) is always followed by a random horizontal/left-to-right flip with 50% probability.
    """
    crop = T.Compose([
      T.RandomResizedCrop(size=(224,224), scale=(0.08, 1.0), ratio=(3/4, 4/3)),
      T.RandomHorizontalFlip(p=0.5)  
    ])
    return crop
    


def color_distortion(s=1.0):
    """
      implementation taken from the paper
    """
    
    # s is the strength of color distortion.
    color_jitter = T.ColorJitter(0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = T.RandomApply([color_jitter], p=0.8)
    rnd_gray = T.RandomGrayscale(p=0.2)
    color_distort = T.Compose(
        [
            rnd_color_jitter,
            rnd_gray
        ]
    )
    return color_distort



def gaussian_blur():
    """
    We blur the image 50% of the time using a Gaussian kernel. We randomly sample σ ∈ [0.1, 2.0], and the 
    kernel size is set to be 10% of the image height/width.
    """
    
    width, height = 224, 224
    k = int(0.1 * min(height, height))  
    k = k if k % 2 == 1 else k + 1   # needs to be odd and an integer


    blur = T.RandomApply([T.GaussianBlur(
        sigma=(0.1,2.0),
        kernel_size=k
    )], p=0.5)
    return blur



def train(model, loader, criterion, optimizer, method):
    model.train()
    total_loss = 0
    for (x_i, x_j), labels in loader:
        optimizer.zero_grad()
        x_i, x_j = x_i.to(device), x_j.to(device)
        if method == 'SimCLR':
            z_i, z_j = model(x_i, x_j)
            loss = criterion(z_i, z_j)
            batch_size = x_i.size(0)
        else:  # SupCon
            images = torch.cat([x_i, x_j], dim=0)
            features = model.encoder(images)
            bsz = labels.size(0)
            f1, f2 = torch.split(features, bsz, dim=0)
            feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = criterion(feats, labels.to(device))
            batch_size = bsz
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_size

    return total_loss / len(loader.dataset)



def evaluation(model, validation_loader, criterion):
    model.eval()
    total_loss = 0.0
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            num_correct += (preds == labels).sum().item()
            num_samples += inputs.size(0)

    avg_loss = total_loss / num_samples
    accuracy = 100.0 * num_correct / num_samples
    return avg_loss, accuracy


def run_simCLR(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # SimCLR augmentations
    transform = T.Compose([
        random_crop(),
        color_distortion(),
        gaussian_blur(),
        T.ToTensor() 
    ])

    # Contrastive data loader
    train_loader = get_contrastive_dataloaders(
        args.data_dir, args.batch_size, args.num_workers, transform
    )

    base_encoder = build_model_spp(num_classes=15).to(device)


    base_encoder.classifier = nn.Identity()
    base_encoder.fc7 = nn.Identity()


    num_ftrs = base_encoder.fc6.out_features
    
    base_encoder.fc7 = nn.Identity()
    model = build_supconresnet(base_encoder, 128, num_ftrs).to(device)
    # SimCLR model with projection head
    if args.method == 'SimCLR':
        criterion = NT_Xent(batch_size=args.batch_size, temperature=0.5)
    else:
        criterion = SupConLoss(temperature=0.07)
        criterion = criterion.to(device)


    # Optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    for epoch in range(1, args.epochs + 1):
        loss = train(model, train_loader, criterion, optimizer, args.method)

        print(f"Epoch [{epoch}/{args.epochs}] Loss: {loss:.4f}")

    checkpoint = {
    'epoch': args.epochs,
    'encoder_state_dict': base_encoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'args': vars(args),
    }
    torch.save(checkpoint, os.path.join(args.output_dir, 'simclr_encoder.pth'))
    print(f"✔  Saved encoder checkpoint to {args.output_dir}/simclr_encoder.pth")

    run_linear_eval(
            encoder=base_encoder,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            epochs=args.epochs,
            lr=0.01,
            weight_decay=1e-4,
            output_dim=num_ftrs,
            args=args,
            checkpoint_path=os.path.join(args.output_dir, 'linear_eval_checkpoint.pth')
        )

if __name__ == '__main__':
    args = parse_args()
    run_simCLR(args)
    