import sys
import json

from utils import load_model_with_placement
import torchvision
from torchvision import transforms
import torch.utils.data


def train_single_batch(model, data, criterion, optimizer):
    output = model(data[0])
    loss = criterion(output, data[1])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


placement = 'cuda:0'
if len(sys.argv) > 1:
    placement_path = sys.argv[1]
    with open(placement_path) as f:
        placement = json.load(f)

model, criterion, optimizer, input_device, output_device = load_model_with_placement(placement, lr=0.01, classes=10)

preprocess = transforms.Compose([
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.MNIST('./mnist_data', train=True, download=True, transform=preprocess)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

test_dataset = torchvision.datasets.MNIST('./mnist_data', train=False, download=True, transform=preprocess)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

model.train()

for epoch in range(50):
    running_loss = 0.0
    for i, batch in enumerate(train_loader):
        batch = batch[0].to(input_device), batch[1].to(output_device)
        loss = train_single_batch(model, batch, criterion, optimizer)
        running_loss += loss.item()

        if i % 50 == 49:
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 50}')
            running_loss = 0

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        data = data[0].to(input_device), data[1].to(output_device)
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {correct / total:.2%}')