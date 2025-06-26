import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Data Preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

batch_size = 64
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Invertible Convolutional Network (ICN) for Super-Resolution
class InvertibleConvLayer(nn.Module):
    def __init__(self, in_channels):
        super(InvertibleConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

    def inverse(self, y):
        return nn.functional.conv2d(y, self.conv.weight.inverse(), stride=1, padding=1)

class ICN(nn.Module):
    def __init__(self, num_layers, in_channels):
        super(ICN, self).__init__()
        self.layers = nn.ModuleList([InvertibleConvLayer(in_channels) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def inverse(self, y):
        for layer in reversed(self.layers):
            y = layer.inverse(y)
        return y

# Function to create low-resolution images (downsampled)
def downsample(images, scale_factor=2):
    return nn.functional.interpolate(images, scale_factor=1/scale_factor, mode='bilinear', align_corners=False)

# Training the Model for Super-Resolution
model = ICN(num_layers=5, in_channels=3)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, _ in train_loader:
        # Create low-resolution images
        low_res_images = downsample(images, scale_factor=2)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(low_res_images)
        loss = criterion(output, images)  # Compare with original high-resolution images
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Function to test super-resolution and visualize results
def super_resolve_and_display(model, images):
    model.eval()
    with torch.no_grad():
        low_res_images = downsample(images, scale_factor=2)
        high_res_output = model(low_res_images)
    
    plt.figure(figsize=(12, 4))
    for i in range(len(images)):
        # Original high-res image
        plt.subplot(3, len(images), i + 1)
        plt.imshow((images[i].permute(1, 2, 0) + 1) / 2)
        plt.axis('off')
        
        # Low-res image
        plt.subplot(3, len(images), i + 1 + len(images))
        plt.imshow((low_res_images[i].permute(1, 2, 0) + 1) / 2)
        plt.axis('off')
        
        # Super-resolved image
        plt.subplot(3, len(images), i + 1 + 2 * len(images))
        plt.imshow((high_res_output[i].permute(1, 2, 0) + 1) / 2)
        plt.axis('off')
    plt.show()

# Test and visualize the super-resolution model on some sample images
sample_images, _ = next(iter(train_loader))
super_resolve_and_display(model, sample_images[:5])
