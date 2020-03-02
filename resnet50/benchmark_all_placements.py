import resnet50
import os
import json

from train import train_with_placement

placement_directory = ''
results_file = ''

with open(results_file, 'w') as f:
    f.write('')

for file in os.listdir(placement_directory):
    if file.endswith('.json'):
        batch_times = train_with_placement(os.path.join(placement_directory, file))

    with open(results_file, 'a') as f:
        f.write(f'{file}, {",".join(map(lambda x: str(x), batch_times))}\n')
