'''
    This file should contain all the visualization scripts
'''
import matplotlib
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from src.utils import create_entire_path_directory
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_squared_error

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



def visualize_multiple_samples(input, graphic_best, graphic_rad21, graphic_positional, hicnn_predicted, hicreg_predicted, target, idx, output_folder):
    f, (ax0, ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 7,  sharex=True, sharey=True)
    ax0.matshow(input, cmap=REDMAP)
    ax0.set_title('Input:\n {:.4f}'.format(mean_squared_error(input, target)))
    ax0.axis('off')

    ax1.matshow(hicreg_predicted, cmap=REDMAP)
    ax1.set_title('HiCReg:\n {:.4f}'.format(mean_squared_error(hicreg_predicted, target)))
    ax1.axis('off')
    
    ax2.matshow(hicnn_predicted, cmap=REDMAP)
    ax2.set_title('HiCNN:\n {:.4f}'.format(mean_squared_error(hicnn_predicted, target)))
    ax2.axis('off')

    ax3.matshow(graphic_positional, cmap=REDMAP)
    ax3.set_title('G-Pos:\n {:.4f}'.format(mean_squared_error(graphic_positional, target)))
    ax3.axis('off')

    ax4.matshow(graphic_rad21, cmap=REDMAP)
    ax4.set_title('G-Rad21:\n {:.4f}'.format(mean_squared_error(graphic_rad21, target)))
    ax4.axis('off')
    
    ax5.matshow(graphic_best, cmap=REDMAP)
    ax5.set_title('G-Best:\n {:.4f}'.format(mean_squared_error(graphic_best, target)))
    ax5.axis('off')


    ax6.matshow(target, cmap=REDMAP)
    ax6.set_title('Target:\n {:.4f}'.format(mean_squared_error(target, target)))
    ax6.axis('off')


    plt.savefig(os.path.join(output_folder, 'chrom:{}_i{}_j{}'.format(idx[0], idx[2], idx[3])))
    plt.close()

def visualize_samples(samples, indexes, path):
    print(path)
    for i in range(samples.shape[0]): 
        idx = indexes[i, :]
        
        plt.matshow(samples[i, 0, :, :], cmap=REDMAP)
        plt.axis('off')
        plt.savefig(os.path.join(path, 'chrom-{}_i-{}_j-{}'.format(idx[0], idx[2], idx[3])))
        plt.close()
        