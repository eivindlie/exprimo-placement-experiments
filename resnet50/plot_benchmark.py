import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

benchmark_results = pd.read_csv('../../exprimo/experiment_results/sim_real_comp/benchmark_results.csv', header=None)
real_times = pd.read_csv('../../exprimo/experiment_results/sim_real_comp/nets/scores.csv',
                         names=['generation', 'Simulated time'], index_col='generation')

benchmark_avg = pd.DataFrame()
benchmark_avg['generation'] = benchmark_results.iloc[:, 0].map(lambda x:
                                                               int(x.replace('gen_', '').replace('.json', '')))
benchmark_avg['Real time (mean)'] = benchmark_results.iloc[:, 1:].mean(axis=1)
benchmark_avg = benchmark_avg.set_index('generation').sort_index()

all_scores = real_times.join(benchmark_avg, how='inner').sort_index()

ax = all_scores.plot()
ax.set_xlabel('Generation')
ax.set_ylabel('Batch execution time (ms)')
plt.title('Real vs. simulated execution time')

plt.show()

correlation = all_scores['Simulated time'].corr(all_scores['Real time (mean)'])
print(f'Correlation: {correlation:.4f}')
