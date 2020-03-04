import resnet
import os
import json

train_with_placement = lambda x: None

placement_directory = ''
results_file = ''

with open(results_file, 'w') as f:
    f.write('')

for file in os.listdir(placement_directory):
    if file.endswith('.json'):
        batch_times = train_with_placement(os.path.join(placement_directory, file))

    with open(results_file, 'a') as f:
        f.write(f'{file}, {",".join(map(lambda x: str(x), batch_times))}\n')
