
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import random

# Cutout Augmentation
class Cutout(object):
    def __init__(self, n_holes=1, length=8):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask).expand_as(img)
        img = img * mask
        return img

# Dataset
class CImageDataset(Dataset):
    def __init__(self, root_dir, label_file=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        if label_file:
            with open(label_file, 'r') as f:
                lines = f.readlines()[1:]
            self.image_paths = [line.strip().split(',')[0] for line in lines]
            self.labels = [int(line.strip().split(',')[1]) for line in lines]
        else:
            self.image_paths = sorted(os.listdir(root_dir))
            self.labels = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.labels:
            label = self.labels[idx]
            return image, label
        else:
            return image, self.image_paths[idx].split('.')[0]

# Model
class CombinedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet34(weights='DEFAULT')
        self.efficientnet = models.efficientnet_b0(weights='DEFAULT')

        self.resnet.fc = nn.Identity()
        self.efficientnet.classifier = nn.Identity()

        for param in self.resnet.parameters():
            param.requires_grad = True
        for param in self.efficientnet.parameters():
            param.requires_grad = True

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 + 1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 100)
        )

    def forward(self, x):
        r = self.resnet(x)
        e = self.efficientnet(x)
        x = torch.cat([r, e], dim=1)
        return self.fc(x)

# EarlyStopping
class EarlyStopping:
    def __init__(self, patience=15):
        self.patience = patience
        self.counter = 0
        self.best_acc = 0
        self.early_stop = False

    def __call__(self, acc, model):
        if acc > self.best_acc:
            self.best_acc = acc
            self.counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Main
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_dataset = CImageDataset('CImage/train', 'CImage/train.txt', transform_train)
    train_len = int(len(train_dataset) * 0.9)
    val_len = len(train_dataset) - train_len
    train_set, val_set = random_split(train_dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2)

    model = CombinedModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)
    early_stopping = EarlyStopping(patience=15)

    for epoch in range(1, 101):
        model.train()
        total, correct = 0, 0
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        acc = correct / total
        print(f'Epoch {epoch} | Val Acc: {acc:.4f}')
        early_stopping(acc, model)
        if early_stopping.early_stop:
            break

    # Inference
    test_dataset = CImageDataset('CImage/test', 'CImage/test.txt', transform_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    results = []
    with torch.no_grad():
        for x, numbers in tqdm(test_loader):
            x = x.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            results.extend(zip(numbers, predicted.cpu().numpy()))

    with open('output.txt', 'w') as f:
        f.write('number, label\n')
        for number, label in results:
            f.write(f"{number}, {label:02d}\n")

if __name__ == '__main__':
    main()
