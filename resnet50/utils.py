import torch

from resnet import resnet50

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def load_model_with_placement(placement, lr=0.01):
    device_lookup = {
        0: 'cpu:0',
        1: 'cpu:0',
        2: 'cuda:0',
        3: 'cuda:1'
    }

    if placement is None:
        placement = 'cpu:0'
    elif isinstance(placement, dict):
        translated_placement = {}
        for layer_name, device in placement.items():
            translated_placement[layer_name] = device_lookup[device]
        placement = translated_placement

    model = resnet50(pretrained=False, placement=placement)

    if isinstance(placement, str):
        input_device = output_device = torch.device(placement)
        model.to(input_device)
    else:
        input_device = placement['conv1']
        output_device = placement['fc1000']

    criterion = torch.nn.CrossEntropyLoss().to(output_device)  # TODO Move loss to correct device
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    return model, criterion, optimizer, input_device, output_device
