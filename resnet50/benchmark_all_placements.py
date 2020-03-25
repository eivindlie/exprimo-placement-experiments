import os
import argparse
import json

from benchmark import benchmark_with_placement

parser = argparse.ArgumentParser(description='Benchmark ResNet-50 on (fake) ImageNet data for all assignments in '
                                             'provided folder')
parser.add_argument('--placement_dir', '-p', dest='placement_dir', default='.', help='The directory that contains '
                                                                                     'device assignment files (in '
                                                                                     'JSON format)')
parser.add_argument('--results_file', '-r', dest='results_file', default='benchmark_results.csv',
                    help='The CSV file that benchmark results will be written to.')

args = parser.parse_args()
placement_directory = args.placement_dir
results_file = args.results_file

with open(results_file, 'w') as f:
    f.write('')

dir_list = os.listdir(placement_directory)
for i, file in enumerate(dir_list):
    if file.endswith('.json'):
        with open(os.path.join(placement_directory, file)) as f:
            placement = json.load(f)

        print(f'Benchmarking assignment {i + 1}/{len(dir_list)}: {file}')

        batch_times = benchmark_with_placement(placement=placement)

        with open(results_file, 'a') as f:
            f.write(f'{file}, {",".join(map(lambda x: str(x), batch_times))}\n')
