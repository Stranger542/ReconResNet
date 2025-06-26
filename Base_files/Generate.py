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

# Invertible Convolutional Network
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

# Training the Model
model = ICN(num_layers=5, in_channels=3).cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, _ in train_loader:
        images = images.cuda()
        
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, images)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Image Generation
def generate_images(model, num_samples):
    model.eval()
    with torch.no_grad():
        latent_samples = torch.randn(num_samples, 3, 32, 32).cuda()
        generated_images = model.inverse(latent_samples)
    return generated_images.cpu()

# Generate and visualize images
num_samples = 5
generated_images = generate_images(model, num_samples)

plt.figure(figsize=(10, 5))
for i in range(num_samples):
    plt.subplot(1, num_samples, i + 1)
    plt.imshow((generated_images[i].permute(1, 2, 0) + 1) / 2)
    plt.axis('off')
plt.show()
