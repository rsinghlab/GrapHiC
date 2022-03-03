from dataset import HiCDataset
from src import globals
from src.raw_hic_dataset_parser import read_hic_header, parse_out_all_hic_files
from src.globals import PATH_TO_HIC_RAW_FILES, PATH_TO_GRAPHIC_DATASETS
from src.models import TransformerConvGrapHiC
import os, itertools
from scipy import spatial
import torch
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import json
from torch_geometric.data import DataLoader
from tqdm import tqdm

HYPERPARAMETERS = {
    'feature_size':1, 
    'embedding_size':64, 
    'n_heads':4, 
    'n_layers':5, 
    'dropout_rate':0.5, 
    'edge_dim':1, 
    'latent_dim':128,
    'set2set_steps': 4,
    'decoder_0': 256,
    'decoder_1': 512,
    'decoder_dropout': 0.5
}


 



model = TransformerConvGrapHiC.TransformerConvGrapHiC(HYPERPARAMETERS)

chromosomes = list(range(2, 23))
cell_lines = ['GM12878', 'IMR90', 'K562']


def compare_embeddings():
    results = {
        'GM12878': {},
        'IMR90': {},
        'K562': {}
    }

    for cell_line in cell_lines:
        for chromosome in chromosomes:
            files = HiCDataset(PATH_TO_GRAPHIC_DATASETS, 'NONE', cell_line, chromosome, 'LOW_RES', regenerate=False)
            encodings = {}
            for dataset in globals.DATASETS['LOW_RES'][cell_line]:
                chrom_size = (globals.CHROMOSOME_SIZES[str(chromosome)])/globals.RESOLUTION 
                for submat in range(0, int(chrom_size), globals.SUB_MATRIX_SIZE):
                    print('Encoding chromosome {} submat {} of dataset {} that belongs to cell line {}'.format(chromosome, submat, dataset, cell_line))
                    data = files.get(dataset, submat)
                    data_loader = DataLoader([data], batch_size=1, shuffle=True)

                    for _, batch in enumerate(tqdm(data_loader)):
                        
                        if dataset in encodings.keys():
                            encodings[dataset].append(model.encode(batch.x, batch.edge_attr, batch.edge_index, batch.batch).cpu().detach().numpy()[0])
                        else:
                            encodings[dataset] = [model.encode(batch.x, batch.edge_attr, batch.edge_index, batch.batch).cpu().detach().numpy()[0]]
    
            for dataset in encodings.keys():
                submat_encodings = encodings[dataset]
                mean = []
                count = 0
                for encoding in submat_encodings:
                    encoding = encoding/np.linalg.norm(encoding)
                    if len(mean) == 0:
                        mean = encoding
                    else:
                        mean += encoding
                    
                    count+= 1 
                
                mean = mean/count
                if chromosome not in results[cell_line].keys():
                    results[cell_line][chromosome] = {}
                results[cell_line][chromosome][dataset] = mean
    
    for chromosome in chromosomes:
        combined_encodings = []
        labels = []
        for cell_line in cell_lines:    
            for dataset in globals.DATASETS['LOW_RES'][cell_line]:
                combined_encodings.append(results[cell_line][chromosome][dataset])
                labels.append(cell_line)
        
        combined_encodings = np.array(combined_encodings)
        labels = np.array(labels)

        
        print(combined_encodings, labels)

        pca = PCA(n_components=3)
        z = pca.fit_transform(combined_encodings)

        df = pd.DataFrame()
        df['y'] = labels
        df['comp-1'] = z[:, 0]
        df['comp-2'] = z[:, 1]

        print(df['comp-1'], df['comp-2'])

        sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                        palette=sns.color_palette("hls", 3),
                        data=df).set(title="Encodeded data projected in PCA space")

        plt.savefig('pca_plot_chr{}_embedding_size32.png'.format(chromosome))

compare_embeddings()