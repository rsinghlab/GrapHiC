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
from matplotlib.gridspec import GridSpec

REDMAP = LinearSegmentedColormap.from_list("bright_red", [(1,1,1),(1,0,0)])

epi_labels = ['CTCF', 'RAD-21', 'DNASE-Seq']
epi_colors = ['#1E88E5', '#FFC107', '#D81B60']

















def plot_hic_with_epi_features(hic, epis, 
    output,                    
    colorbar=False,
    colorbar_orientation='vertical',
    fontsize=24,
    epi_yaxis=False
):
    x_ticks = False
    
    n = hic.shape[0]
    
    
    rs = [0.8, 0.05] + [0.1, 0.01] * len(epis)
    rs = np.array(rs[:-1])
    
    
    # Calculate figure height
    fig_height = 12 * np.sum(rs)
    rs = rs / np.sum(rs)  # normalize to 1 (ratios)
    fig = plt.figure(figsize=(12, fig_height))

    # Split the figure into rows with different heights
    gs = GridSpec(len(rs), 1, height_ratios=rs)

    # Ready for plotting heatmap
    ax0 = plt.subplot(gs[0, :])
    # Define the rotated axes and coordinates
    coordinate = np.array([[[(x + y) / 2, y - x] for y in range(n + 1)] for x in range(n + 1)])
    X, Y = coordinate[:, :, 0], coordinate[:, :, 1]
    # Plot the heatmap
    vmax = np.percentile(hic, 99.75)
    
    im = ax0.pcolormesh(X, Y, hic, vmin=0, vmax=vmax, cmap='Reds')
    ax0.axis('off')
    ax0.set_ylim([0, n])
    ax0.set_xlim([0, n])
    if colorbar:
        if colorbar_orientation == 'horizontal':
            _left, _width, _bottom, _height = 0.12, 0.25, 1 - rs[0] * 0.25, rs[0] * 0.03
        elif colorbar_orientation == 'vertical':
            _left, _width, _bottom, _height = 0.9, 0.02, 1 - rs[0] * 0.7, rs[0] * 0.5
        else:
            raise ValueError('Wrong orientation!')
        cbar = plt.colorbar(im, cax=fig.add_axes([_left, _bottom, _width, _height]),
                            orientation=colorbar_orientation)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.outline.set_visible(False)


    # Ready for plotting 1D signals
    if epi_labels:
        assert len(epis) == len(epi_labels)
    if epi_colors:
        assert len(epis) == len(epi_colors)

    for i, epi in enumerate(epis):
        print(epi.shape)
        ax1 = plt.subplot(gs[2 + 2 * i, :])

        if epi_colors:
            ax1.fill_between(np.arange(n), 0, epi, color=epi_colors[i])
        else:
            ax1.fill_between(np.arange(n), 0, epi)
        
        ax1.spines['left'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)

        if not epi_yaxis:
            ax1.set_yticks([])
            ax1.set_yticklabels([])
        else:
            ax1.spines['right'].set_visible(True)
            ax1.tick_params(labelsize=fontsize)
            ax1.yaxis.tick_right()

        if i != len(epis) - 1:
            ax1.set_xticks([])
            ax1.set_xticklabels([])
        # ax1.axis('off')
        # ax1.xaxis.set_visible(True)
        # plt.setp(ax1.spines.values(), visible=False)
        # ax1.yaxis.set_visible(True)

        ax1.set_xlim([-0.5, n - 0.5])
        if epi_labels:
            ax1.set_ylabel(epi_labels[i], fontsize=fontsize, rotation=0)
    
    ax1.spines['bottom'].set_visible(True)
    if x_ticks:
        tick_pos = np.linspace(0, n - 1, len(x_ticks))
        ax1.set_xticks(tick_pos)
        ax1.set_xticklabels(x_ticks, fontsize=fontsize)
    else:
        ax1.set_xticks([])
        ax1.set_xticklabels([])

    plt.savefig(output)
    plt.close()

















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

def visualize_samples(samples, indexes, input_encodings, enc_order, path):
    for i in range(samples.shape[0]): 
        idx = indexes[i, :]
        input_encoding = input_encodings[i, 0, :, :]
        hic = samples[i, 0, :, :]
        
        epis = []
        for epi in epi_labels:
            index_of_epi = enc_order.item()[idx[0]].index(epi)
            epis.append(input_encoding[:, index_of_epi])
        
        plot_hic_with_epi_features(
            hic, epis, 
            os.path.join(path, 'chrom-{}_i-{}_j-{}'.format(idx[0], idx[2], idx[3]))
        )
        # fig, axs = plt.subplots(3, gridspec_kw={'height_ratios': [8, 1, 1]})
        
        # axs[0].imshow(np.triu(samples[i, 0, :, :]), cmap='Reds')
        # axs[0].axis('off')
        # axs[0].colorbar()
        
        # axs[1].plot(np.array(range(rad_21.shape[0])), rad_21)
        # axs[1].axis('off')
        # axs[2].plot(np.array(range(histone.shape[0])), histone)
        # axs[2].axis('off')

        # plt.tight_layout()
        
        # plt.savefig(os.path.join(path, 'chrom-{}_i-{}_j-{}'.format(idx[0], idx[2], idx[3])))
        # plt.close()
        