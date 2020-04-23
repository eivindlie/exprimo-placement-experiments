import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')

FILE_PATH = '../../exprimo/experiment_results/e2-1_batch_times_without_last_batch.csv'

# Set whether to subtract and divide by average, yielding residuals as fraction of average training time
NORMALIZE = True

# Set whether to create individual plots of each generation, or the average over all generations
PLOT_INDIVIDUAL = False


def plot_times(data, title, output_file=None):
    plt.plot(data)
    plt.xlabel('Batch')
    plt.ylabel('Training time (ms)')
    plt.title(title)

    if output_file:
        plt.savefig(output_file, transparent=False)

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

            output_file = f'../../exprimo/experiment_results/plots/batch_training_times/gen_{generation}.png'

            plot_times(times, f'Generation {generation}', output_file)
    else:
        avg_results = results.mean(axis=0)
        plot_times(avg_results, 'Average batch time residuals',
                   output_file='../../exprimo/experiment_results/plots/batch_training_times/average.png')
