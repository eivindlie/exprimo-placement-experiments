import torch
import time


def benchmark_bandwidth(tensor_size, source_device, target_device):
    source_device = torch.device(source_device)
    target_device = torch.device(target_device)
    t = torch.rand((tensor_size // 4,))
    t.to(source_device)
    torch.cuda.synchronize()

    start_time = time.time()
    t.to(target_device)
    torch.cuda.synchronize()
    end_time = time.time()
    transfer_time = end_time - start_time

    tensor_size = t.nelement() * t.element_size()
    bandwidth = tensor_size / transfer_time  # Bytes / second
    bandwidth = (bandwidth * 8) / 10**6  # Mbit/s

    return bandwidth


if __name__ == '__main__':
    result_file = './bandwidth.csv'
    source_device = 'cpu'
    target_device = 'cuda:0'
    tensor_sizes = [10**i for i in range(3, 11)]

    with open(result_file, 'w') as f:
        f.write('')

    for tensor_size in tensor_sizes:
        print(f'Benchmarking tensor of size {tensor_size / 10**6:.3f}MB... ', end='')
        bandwidth = benchmark_bandwidth(tensor_size, source_device, target_device)
        print(f'{bandwidth}Mbit/s')
        with open(result_file, 'a') as f:
            f.write(f'{tensor_size}, {bandwidth}\n')
