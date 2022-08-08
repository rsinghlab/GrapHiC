'''
    This file should contain all the visualization scripts
'''
import matplotlib
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from src.utils import create_entire_path_directory

REDMAP = LinearSegmentedColormap.from_list("bright_red", [(1,1,1),(1,0,0)])


def plot_distribution_with_precentiles(values, graph_name):
    '''
        This function draws out a plot comparing the distribution of contact counts
        with respect to the percentiles
        @params: values <np.array> values or input distribution
        @returns None
    '''
    x_vals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 99.5, 99.9, 99.95, 99.99]
    y_vals = list(map(lambda x: np.percentile(values, x), x_vals))

    x_vals = list(map(lambda x: str(x), x_vals))
    
    plt.xticks(rotation = 45) 
    plt.plot(x_vals, y_vals)
    plt.savefig('outputs/graphs/{}'.format(graph_name), format='png')
    plt.close()


def visualize(inputs, outputs, targets, batch_idx, output_folder):
    create_entire_path_directory(output_folder)
    for idx in range(inputs.shape[0]):
        input = inputs[idx, 0, :, :].to('cpu').numpy()
        target = targets[idx, 0, :, :].to('cpu').numpy()
        output = outputs[idx, 0, :, :].detach().to('cpu').numpy()
        f, (ax0, ax1, ax2) = plt.subplots(1, 3,  sharex=True, sharey=True)
        ax0.matshow(input, cmap=REDMAP)
        ax0.set_title('Input')
        ax0.axis('off')


        ax1.matshow(target, cmap=REDMAP)
        ax1.set_title('Target')
        ax1.axis('off')

        ax2.matshow(output, cmap=REDMAP)
        ax2.set_title('Generated')
        ax2.axis('off')
        
        plt.savefig(os.path.join(output_folder, 'bidx:{}_sampleidx{}.png'.format(batch_idx, idx) ))
        plt.close()

