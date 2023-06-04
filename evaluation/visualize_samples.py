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

epi_labels = ['H3K27ME3', 'CTCF', 'DNASE-Seq']
epi_colors = ['#1E88E5' , '#FFC107', '#D81B60']


def plot_hic_with_epi_features(hic, epis, 
    output,                    
    colorbar=False,
    colorbar_orientation='vertical',
    fontsize=34,
    epi_yaxis=False
):
    
    # if os.path.exists(output):
    #     return 
    
    x_ticks = False
    
    n = hic.shape[0]
    
    
    rs = [0.9, 0.05] + [0.1, 0.01] * len(epis)
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
    vmax = np.percentile(hic, 99.5)
    
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
    

def percentile_cutoff(hic, percentile=98):
    cutoff = np.percentile(hic, percentile)
    hic = np.minimum(cutoff, hic)
    hic = np.maximum(hic, 0)
    return hic
    

def visualize_hic_square_format(hic, output):
    f, (ax0) = plt.subplots(1, 1)
    ax0.matshow(hic, cmap=REDMAP)
    ax0.axis('off')
    plt.tight_layout()
    plt.savefig(output, bbox_inches='tight')
    plt.close()
    



def visualize_samples(samples, indexes, input_encodings, enc_order, path):
    for i in range(samples.shape[0]): 
        idx = indexes[i, :]
        #input_encoding = input_encodings[i, 0, :200, :]
        hic = percentile_cutoff(samples[i, 0, :200, :200])
        
        # epis = []
        # for epi in epi_labels:
        #     index_of_epi = enc_order.item()[idx[0]].index(epi)
        #     epis.append(input_encoding[:, index_of_epi])
        
        visualize_hic_square_format(hic, os.path.join(path, 'chrom-{}_i-{}_j-{}.png'.format(idx[0], idx[2], idx[3])))
        # plot_hic_with_epi_features(
        #     hic, epis, 
        #     os.path.join(path, 'chrom-{}_i-{}_j-{}.png'.format(idx[0], idx[2], idx[3]))
        # )