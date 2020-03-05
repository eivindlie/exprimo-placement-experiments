import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

benchmark_results = pd.read_csv('../../exprimo/experiment_results/sim_real_comp/benchmark_results.csv', header=None)
real_times = pd.read_csv('../../exprimo/experiment_results/sim_real_comp/nets/scores.csv',
                         names=['generation', 'real_score'], index_col='generation')

benchmark_avg = pd.DataFrame()
benchmark_avg['generation'] = benchmark_results.iloc[:, 0].map(lambda x:
                                                               int(x.replace('gen_', '').replace('.json', '')))
benchmark_avg['benchmark_score'] = benchmark_results.iloc[:, 1:].mean(axis=1)
benchmark_avg = benchmark_avg.set_index('generation').sort_index()

all_scores = real_times.join(benchmark_avg, how='inner').sort_index()

all_scores.plot()
plt.show()


