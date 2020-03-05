import time
import sys
import json

from torchvision import transforms
import torchvision
import torch.utils.data
import torch

from resnet import resnet50
from utils import load_model_with_placement

BATCH_SIZE = 128


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = torchvision.datasets.FakeData(transform=preprocess, size=500)

train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True
)


def train_single_batch(model, data, criterion, optimizer):
    output = model(data[0])
    loss = criterion(output, data[1])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def benchmark_with_placement(batches=50, placement='cuda:0', lr=0.01):
    print('Starting benchmark...')

    model, criterion, optimizer, input_device, output_device = load_model_with_placement(placement, lr=lr)

    model.train()
    batch_times = []

    b = 0
    while b < batches + 1:
        for data in train_loader:
            print(f'Batch {b + 1}/{batches + 1}', end='')

            torch.cuda.synchronize()
            data = data[0].to(input_device), data[1].to(output_device)

            start = time.time()
            train_single_batch(model, data, criterion, optimizer)
            torch.cuda.synchronize()
            end = time.time()

            batch_times.append((end - start) * 1000)

            print(f' {batch_times[-1]}ms')

            b += 1
            if b >= batches + 1:
                break

    return batch_times[1:]


if __name__ == '__main__':
    placement = 'cpu:0'
    if len(sys.argv) > 1:
        placement_path = sys.argv[1]
        with open(placement_path) as f:
            placement = json.load(f)

    print(benchmark_with_placement(placement=placement))
