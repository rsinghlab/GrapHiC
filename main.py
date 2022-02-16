from ast import parse
from dataset import HiCDataset
from src.raw_hic_dataset_parser import parse_out_all_hic_files
from src.globals import PATH_TO_HIC_RAW_FILES, PATH_TO_GRAPHIC_DATASETS
from torch_geometric.nn.conv import TransformerConv


#parse_out_all_hic_files(PATH_TO_HIC_RAW_FILES)



low_res_gm12878_files = HiCDataset(PATH_TO_GRAPHIC_DATASETS, 'NONE', 'GM12878', 18, 'LOW_RES')



print(low_res_gm12878_files.len())

encode_0_chr = low_res_gm12878_files.get('encode-0')

conv_layer_one = TransformerConv(1, 
                                128, 
                                heads=4, 
                                concat=False,
                                beta=True,
                                edge_dim=1)

print(encode_0_chr.edge_index.dtype)


x = conv_layer_one(encode_0_chr.x, encode_0_chr.edge_index, encode_0_chr.edge_attr).relu() 

print(x)
