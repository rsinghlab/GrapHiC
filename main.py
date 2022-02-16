from ast import parse
from dataset import HiCDataset
from src.raw_hic_dataset_parser import parse_out_all_hic_files
from src.globals import PATH_TO_HIC_RAW_FILES, PATH_TO_GRAPHIC_DATASETS
from torch_geometric.nn.conv import TransformerConv


#parse_out_all_hic_files(PATH_TO_HIC_RAW_FILES)



low_res_gm12878_files = HiCDataset(PATH_TO_GRAPHIC_DATASETS, 'NONE', 'GM12878', 18, 'LOW_RES')



print(low_res_gm12878_files.len())
print(low_res_gm12878_files.get('encode-0').x.shape[1])


conv_layer_one = TransformerConv(1, 
                                self.encoder_embedding_size, 
                                heads=4, 
                                concat=False,
                                beta=True,
                                edge_dim=self.edge_dim)