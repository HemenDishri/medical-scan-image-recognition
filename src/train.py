import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from medmnist import PathMNIST
import medmnist
from torchvision import transforms
from model import SimpleCNN
import numpy as np
from sklearn.metrics import accuracy_score

# Prepare transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# Load MedMNIST dataset (PathMNIST as example)
train_dataset = PathMNIST(split='train', transform=transform, download=True)
test_dataset = PathMNIST(split='test', transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(in_channels=3, num_classes=9).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.squeeze().to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Evaluation
def evaluate():
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.squeeze().to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    acc = accuracy_score(targets, preds)
    print(f"Test Accuracy: {acc*100:.2f}%")

if __name__ == '__main__':
    train(num_epochs=5)
    evaluate()