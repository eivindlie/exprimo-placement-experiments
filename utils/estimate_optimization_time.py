"""
Estimates the optimization time when running a genetic algorithm with real execution benchmarks, as opposed to the
usage of an execution simulator. This is estimated as the sum of evaluation times, as this will dominate the total
execution time. The calculation is carried out using benchmarks of the BEST individual of selected generations,
providing a lower bound, as there will be many worse solutions in the population at any time.
"""

import pandas as pd

BENCHMARK_FILE = '../../exprimo/experiment_results/sim_real_comp/benchmark_results.csv'
GENERATIONS = 500
POPULATION_SIZE = 100

benchmark_raw = pd.read_csv(BENCHMARK_FILE, delimiter=',', header=None)
benchmark = pd.DataFrame()
benchmark['generation'] = benchmark_raw.iloc[:, 0].map(lambda x:
                                                       int(x.replace('gen_', '').replace('.json', '')))
benchmark['time'] = benchmark_raw.iloc[:, 1:].mean(axis=1)
benchmark = benchmark.set_index('generation').sort_index()

execution_time = 0
current_time = benchmark.iloc[0, 0]
for g in range(GENERATIONS):
    if g in benchmark.index:
        current_time = benchmark.at[g, 'time']
    execution_time += current_time * POPULATION_SIZE

print(f'Estimated optimization time: {execution_time / 1000:.2f}s = {execution_time / (60 * 1000):.2f}min '
      f'= {execution_time / (60 * 60 * 1000):.2f}h')
