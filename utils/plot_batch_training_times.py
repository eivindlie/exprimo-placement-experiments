import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')

FILE_PATH = '~/logs/e2_batch_times/e2-2_batch_times_with_all_batches.csv'

# Set whether to subtract and divide by average, yielding residuals as fraction of average training time
NORMALIZE = True

# Set whether to create individual plots of each generation, or the average over all generations
PLOT_INDIVIDUAL = False


def plot_times(data, title, output_file=None):
    plt.plot(data)
    plt.xlabel('Batch')
    plt.ylabel('Residual batch training time (fraction of mean)')
    plt.title(title)

    if output_file:
        plt.savefig(output_file, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    results = pd.read_csv(FILE_PATH, header=None, index_col=0)

    if NORMALIZE:
        means = results.mean(axis=1)
        results = results.sub(means, axis=0).divide(means, axis=0)

    if PLOT_INDIVIDUAL:
        for i in range(results.shape[0]):
            generation = results.index[0].replace('gen_', '').replace('.json', '')
            times = results.iloc[i, 1:]

            output_file = os.path.expanduser(f'~/logs/batch_training_time/gen_{generation}.pdf')

            plot_times(times, f'Generation {generation}', output_file)
    else:
        avg_results = results.mean(axis=0)
        plot_times(avg_results, 'Average batch time residuals (with last batch in dataset)',
                   output_file=os.path.expanduser(
                       f'~/logs/e2_batch_times/e2-2_batch_training_time_with_last_batch.svg'))
