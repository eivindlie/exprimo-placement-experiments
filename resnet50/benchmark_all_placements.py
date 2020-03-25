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


def generation_filter(file):
    if not file.endswith('.json'):
        return False
    generation = int(file.replace('gen_', '').replace('.json', ''))
    return generation % 10 == 0

dir_list = os.listdir(placement_directory)
dir_list = list(filter(generation_filter, dir_list))

for i, file in enumerate(dir_list):
    if file.endswith('.json'):
        with open(os.path.join(placement_directory, file)) as f:
            placement = json.load(f)

        print(f'Benchmarking assignment {i + 1}/{len(dir_list)}: {file}')

        batch_times = benchmark_with_placement(placement=placement, batches=10)

        with open(results_file, 'a') as f:
            f.write(f'{file}, {",".join(map(lambda x: str(x), batch_times))}\n')
