from dataset import get_dataloader
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

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hic_data_loader = get_dataloader(globals.DATASET_REQUIREMENTS)





# low_res_gm12878_files = HiCDataset(PATH_TO_GRAPHIC_DATASETS, 'NONE', 'GM12878', 11, 'LOW_RES', regenerate=False)
# low_res_imr90_files = HiCDataset(PATH_TO_GRAPHIC_DATASETS, 'NONE', 'IMR90', 11, 'LOW_RES', regenerate=False)
# low_res_k562_files = HiCDataset(PATH_TO_GRAPHIC_DATASETS, 'NONE', 'K562', 11, 'LOW_RES', regenerate=False)

# model = TransformerConvGrapHiC.TransformerConvGrapHiC(
#     1, 32, 4, 5, 0.5, 0.5, 1, 1 
# )

# # print(low_res_gm12878_files.len())
# # print(low_res_k562_files.len())
# # print(low_res_imr90_files.len())

# gm12878_encodings = {}
# for dataset in globals.DATASETS['LOW_RES']['GM12878']:
#     for submat in range(0, 13500, 1000):
#         print(dataset, submat)
#         data = low_res_gm12878_files.get(dataset, submat)
#         if dataset in gm12878_encodings.keys():
#             gm12878_encodings[dataset].append(model(data.x, data.edge_attr, data.edge_index, 1).cpu().detach().numpy()[0])
#         else: 
#            gm12878_encodings[dataset] = [model(data.x, data.edge_attr, data.edge_index, 1).cpu().detach().numpy()[0]]

# for dataset in gm12878_encodings.keys():
#     submat_encodings = gm12878_encodings[dataset]
#     mean = []
#     count = 0
#     for encoding in submat_encodings:
#         encoding = encoding/np.linalg.norm(encoding)
#         if len(mean) == 0:
#             mean = encoding
#         else:
#             mean += encoding
        
#         count+= 1 
    
#     mean = mean/count
#     gm12878_encodings[dataset] = mean
    

# imr90_encodings = {}
# for dataset in globals.DATASETS['LOW_RES']['IMR90']:
#     for submat in range(0, 13500, 1000):
#         print(dataset, submat)
#         data = low_res_imr90_files.get(dataset, submat)
#         if dataset in imr90_encodings.keys():
#             imr90_encodings[dataset].append(model(data.x, data.edge_attr, data.edge_index, 1).cpu().detach().numpy()[0])
#         else:
#             imr90_encodings[dataset] = [model(data.x, data.edge_attr, data.edge_index, 1).cpu().detach().numpy()[0]]

# for dataset in imr90_encodings.keys():
#     submat_encodings = imr90_encodings[dataset]
#     mean = []
#     count = 0
#     for encoding in submat_encodings:
#         encoding = encoding/np.linalg.norm(encoding)
#         if len(mean) == 0:
#             mean = encoding
#         else:
#             mean += encoding
        
#         count+= 1 
    
#     mean = mean/count
#     imr90_encodings[dataset] = mean


# k562_encodings = {}
# for dataset in globals.DATASETS['LOW_RES']['K562']:
#     for submat in range(0, 13500, 1000):
#         print(dataset, submat)
#         data = low_res_k562_files.get(dataset, submat)
#         if dataset in k562_encodings.keys():
#             k562_encodings[dataset].append(model(data.x, data.edge_attr, data.edge_index, 1).cpu().detach().numpy()[0])
#         else:
#             k562_encodings[dataset] = [model(data.x, data.edge_attr, data.edge_index, 1).cpu().detach().numpy()[0]]

# for dataset in k562_encodings.keys():
#     submat_encodings = k562_encodings[dataset]
#     mean = []
#     count = 0
#     for encoding in submat_encodings:
#         encoding = encoding/np.linalg.norm(encoding)
#         if len(mean) == 0:
#             mean = encoding
#         else:
#             mean += encoding
        
#         count+= 1 
    
#     mean = mean/count
#     k562_encodings[dataset] = mean

# print(gm12878_encodings)
# print(imr90_encodings)
# print(k562_encodings)


# combined_encodings = []
# labels = []
# for dataset in gm12878_encodings: 
#     combined_encodings.append(gm12878_encodings[dataset])
#     labels.append('gm12878')

# for dataset in imr90_encodings: 
#     combined_encodings.append(imr90_encodings[dataset])
#     labels.append('imr90')

# for dataset in k562_encodings: 
#     combined_encodings.append(k562_encodings[dataset])
#     labels.append('k562')



# combined_encodings = np.array(combined_encodings)
# labels = np.array(labels)

# print(combined_encodings.shape, labels.shape)

# print(combined_encodings, labels)



# #print(gm12878_encodings)
# # print(imr90_encodings)
# # print(k562_encodings)
# pca = PCA(n_components=3)
# z = pca.fit_transform(combined_encodings)

# df = pd.DataFrame()
# df['y'] = labels
# df['comp-1'] = z[:, 0]
# df['comp-2'] = z[:, 1]

# print(df['comp-1'], df['comp-2'])

# sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
#                 palette=sns.color_palette("hls", 3),
#                 data=df).set(title="Encodeded data projected in tSNE space")

# plt.savefig('pca_plot_chr_11_embedding_size32.png')


# encode_0_chr_18_s0 = low_res_gm12878_files.get('encode-0', 5000)
# encode_1_chr_18_s0 = low_res_gm12878_files.get('encode-1', 5000)
# encode_2_chr_18_s0 = low_res_gm12878_files.get('encode-2', 5000)
# encode_3_chr_18_s0 = low_res_gm12878_files.get('encode-3', 5000)
# encode_4_chr_18_s0 = low_res_gm12878_files.get('encode-4', 5000)





# x1 = model(encode_0_chr_18_s0.x, encode_0_chr_18_s0.edge_attr, encode_0_chr_18_s0.edge_index, 1)
# x2 = model(encode_1_chr_18_s0.x, encode_1_chr_18_s0.edge_attr, encode_1_chr_18_s0.edge_index, 1)
# x3 = model(encode_2_chr_18_s0.x, encode_2_chr_18_s0.edge_attr, encode_2_chr_18_s0.edge_index, 1)
# x4 = model(encode_3_chr_18_s0.x, encode_3_chr_18_s0.edge_attr, encode_3_chr_18_s0.edge_index, 1)
# x5 = model(encode_4_chr_18_s0.x, encode_4_chr_18_s0.edge_attr, encode_4_chr_18_s0.edge_index, 1)

# print(x1, x2, x3, x4, x5)

# embeddings = list(itertools.combinations([x1, x2, x3, x4, x5], 2))

# for embedding in embeddings:
#     print(1 - spatial.distance.cosine(embedding[0].cpu().detach().numpy(), embedding[1].cpu().detach().numpy()))

