from dataset import HiCDataset
from src.raw_hic_dataset_parser import read_hic_header
from src.globals import PATH_TO_HIC_RAW_FILES, PATH_TO_GRAPHIC_DATASETS
from torch_geometric.nn.conv import TransformerConv
import os


#print(read_hic_header(os.path.join(PATH_TO_HIC_RAW_FILES, 'GM12878', 'HighRes', 'primary.hic'))['chromsizes'])

low_res_gm12878_files = HiCDataset(PATH_TO_GRAPHIC_DATASETS, 'NONE', 'GM12878', 18, 'LOW_RES', regenerate=False)

print(low_res_gm12878_files.len())



encode_0_chr_18_s0 = low_res_gm12878_files.get('encode-0', 0)
encode_1_chr_18_s0 = low_res_gm12878_files.get('encode-1', 0)
encode_2_chr_18_s0 = low_res_gm12878_files.get('encode-2', 0)
encode_3_chr_18_s0 = low_res_gm12878_files.get('encode-3', 0)

print(encode_1_chr_18_s0)


# conv_layer_one = TransformerConv(1, 
#                                 128, 
#                                 heads=4, 
#                                 concat=False,
#                                 beta=True,
#                                 edge_dim=1)

# print(encode_0_chr.edge_index.dtype)


# x = conv_layer_one(encode_0_chr.x, encode_0_chr.edge_index, encode_0_chr.edge_attr).relu() 

# print(x)
