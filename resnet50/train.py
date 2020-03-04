import time
import sys
import json

from torchvision import transforms
import torchvision
import torch.utils.data
import torch

from resnet import resnet50

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

    model = resnet50(pretrained=False, placement=placement if isinstance(placement, dict) else None)

    if isinstance(placement, str):
        input_device = output_device = torch.device(placement)
        model.to(input_device)
    else:
        input_device = placement['conv1']
        output_device = placement['fc1000']

    criterion = torch.nn.CrossEntropyLoss().to(output_device)  # TODO Move loss to correct device
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    model.train()
    batch_times = []

    b = 0
    while b < batches + 1:
        for data in train_loader:
            print(f'Batch {b + 1}/{batches + 1}', end='')

            data = data[0].to(input_device), data[1].to(output_device)

            start = time.time()
            train_single_batch(model, data, criterion, optimizer)
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

    benchmark_with_placement(placement=placement)
