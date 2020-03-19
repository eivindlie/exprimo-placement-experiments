import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

FILE_PATH = '../../exprimo/experiment_results/sim_real_comp/resnet_benchmark_with_all_batches.csv'


def plot_times(data, title, output_file=None):
    plt.plot(data)
    plt.xlabel('Batch')
    plt.ylabel('Training time (ms)')
    plt.title(title)

    if output_file:
        plt.savefig(output_file)

    plt.show()


if __name__ == '__main__':
    results = pd.read_csv(FILE_PATH, header=None)

    for i in range(results.shape[0]):
        generation = results.iloc[i, 0].replace('gen_', '').replace('.json', '')
        times = results.iloc[i, 1:]

        output_file = f'../../exprimo/experiment_results/plots/batch_training_times/gen_{generation}.png'

        plot_times(times, f'Generation {generation}', output_file)
