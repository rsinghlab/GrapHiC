import os
import json
import torch
import hashlib

from src.run import run

from src.models.GrapHiC import GrapHiC
from src.models.HiCPlus import HiCPlus
from src.parse_hic_files import parse_hic_file, download_file
from src.epigentic_encodings import parse_node_encoding_file
from src.dataset_creator import create_dataset_from_hic_files
from src.utils import EPIGENETIC_FILES_DIRECTORY, HIC_FILES_DIRECTORY, PARSED_EPIGENETIC_FILES_DIRECTORY, PARSED_HIC_FILES_DIRECTORY, DATASET_DIRECTORY


### These are default parameters, to run interesting experiments change these parameters ###
hic_data_resolution = 10000

# These hyperparameters go to GrapHiC model controlling the training batch size, optimizer parameters 
HYPERPARAMETERS = {
    'epochs'            : 100, 
    'batch_size'        : 64, # Change to control per batch GPU memory requirement
    'optimizer_type'    : 'ADAM',
    'learning_rate'     : 0.001,
    'momentum'          : 0.9,
    'num_heads'         : 4,
}

# These parameters are used by the dataset creator function to describe how to divide the chromosome matrices
cropping_params = {
    'sub_mat'   :400,
    'stride'    :50,
    'bounds'    :40,
    'padding'   :True
}

# These parameters are also used by the dataset creator function but this set contains the normalization parameters
normalization_params = {
    'norm'              : True,  # To normalize or not
    'remove_zeros'      : True,  # Remove zero before percentile calculation
    'set_diagonal_zero' : False, # Remove the diagonal before percentile calculation
    'percentile'        : 99.0,  # Percentile 
    'rescale'           : True,  # After applying cutoff, rescale between 0 and 1
    'chrom_wide'        : True,  # Apply it on chromosome scale #TODO: Sample wise normalization isn't implemented
    'draw_dist_graphs'  : False  # Visualize the distribution of the chromosome
}

# Epigenetic features list
epi_features_list = [
    'RAD-21', 'RNA-Pol2','CTCF', 'ATAC-Seq', 'DNASE-Seq', 'H3K27ME3', 'H3K27AC', 
    'H3K36ME3', 'H3K4ME1', 'H3K4ME2', 'H3K4ME3', 'H3K79ME2', 'H3K9AC', 'H4K20ME1', 
    'H3K9ME3'
]

# Some other dataset creation paramters
positional_encoding_method = 'graph' # Required for GrapHiC, can take values between ['constant', 'monotonic', 'transformer', 'graph']
non_informative_row_resolution_method = 'intersection' # This finds common non-informative rows in both dataset and then removes them. Can take ['ignore', 'target', 'intersection']
noise_augmentation_method = 'none' # This function adds noise to all the input samples should improve training in certain cases. Can take ['none', 'random', 'uniform', 'gaussian']
node_embedding_concat_method = 'concat'


base_files = ['hic026', 'encode0', 'hic033']



# These files should exist, (currently not using all of them but would at some point)
hic_file_paths = {
    'GM12878-geo-raoetal': {
            'local_path' : os.path.join(HIC_FILES_DIRECTORY, 'GM12878', 'geo-raoetal.hic'),
            'remote_path': 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63525&format=file&file=GSE63525%5FGM12878%5Finsitu%5Fprimary%2Breplicate%5Fcombined%5F30%2Ehic'
    },
    'GM12878-encode-0': {
            'local_path' : os.path.join(HIC_FILES_DIRECTORY, 'GM12878', 'encode-0.hic'),
            'remote_path': 'https://www.encodeproject.org/files/ENCFF799QGA/@@download/ENCFF799QGA.hic'
    },
    'GM12878-encode-1': {
            'local_path' : os.path.join(HIC_FILES_DIRECTORY, 'GM12878', 'encode-1.hic'),
            'remote_path': 'https://www.encodeproject.org/files/ENCFF473CAA/@@download/ENCFF473CAA.hic'
    },
    'GM12878-geo-026': {
            'local_path' : os.path.join(HIC_FILES_DIRECTORY, 'GM12878', 'geo-026.hic'),
            'remote_path': 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM1551575&format=file&file=GSM1551575%5FHIC026%5F30%2Ehic'
    },
}


epigenetic_factor_paths = {
    'GM12878': {
        'H3K27AC': {
            'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneGm12878H3k27acStdSig.bigWig',
            'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'h3k27ac.bigwig'),
        },
        'H3K27ME3': {
            'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneGm12878H3k27me3StdSigV2.bigWig',
            'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'h3k27me3.bigwig'),
        },
        'H3K36ME3': {
            'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneGm12878H3k36me3StdSig.bigWig',
            'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'h3k36me3.bigwig'),
        },
        'H3K4ME1': {
            'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneGm12878H3k4me1StdSig.bigWig',
            'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'h3k4me1.bigwig'),
        },
        'H3K4ME2': {
            'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneGm12878H3k4me2StdSig.bigWig',
            'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'h3k4me2.bigwig'),
        },
        'H3K4ME3': {
            'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneGm12878H3k4me3StdSig.bigWig',
            'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'h3k4me3.bigwig'),
        },
        'H3K79ME2': {
            'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneGm12878H3k79me2StdSig.bigWig',
            'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'h3k79me2.bigwig'),
        },
        'H3K9AC': {
            'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneGm12878H3k9acStdSig.bigWig',
            'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'h3k9ac.bigwig'),
        },
        'H4K20ME1': {
            'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneGm12878H4k20me1StdSig.bigWig',
            'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'h4k20me1.bigwig'),
        },
        'H3K9ME3': {
            'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneGm12878H3k9me3StdSig.bigWig',
            'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'h3k9me3.bigwig'),
        },
        'DNASE-Seq':{
            'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeOpenChromDnase/wgEncodeOpenChromDnaseGm12878Sig.bigWig',
            'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'dnase.bigwig'),
        },
        'ATAC-Seq': {
            'remote_path': 'https://drive.google.com/file/d/1dfvGmomovO6TLJRPhG13OXL6fdRk5GTQ/view?usp=sharing',
            'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'atacseq.bigwig')
        }, 
        'CTCF': {
            'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneGm12878CtcfStdSig.bigWig',
            'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'ctcf.bigwig')
        },
        'RNA-Pol2': {
            'remote_path': 'https://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeOpenChromChip/wgEncodeOpenChromChipGm12878Pol2Sig.bigWig',
            'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'rnapol2.bigwig')
        }, 
        'RAD-21': {
            'remote_path': 'https://encode-public.s3.amazonaws.com/2012/07/01/bb401e4f-91f5-4ddc-ac2b-2b36a56ec114/ENCFF000WCT.bigWig',
            'local_path': os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'rad21.bigwig')
        },
    }
}




# Step 1: Download and parse the HiC files
for hic_file_key in hic_file_paths.keys():
    if not os.path.exists(hic_file_paths[hic_file_key]['local_path']):
        download_file(hic_file_paths[hic_file_key])
    
    parse_hic_file(
        hic_file_paths[hic_file_key]['local_path'], 
        os.path.join(PARSED_HIC_FILES_DIRECTORY, hic_file_key),
        hic_data_resolution
    )

node_encoding_files = []

# Download and parse the Epigenetic files
for cell_line in epigenetic_factor_paths.keys():
    for histone_mark in epigenetic_factor_paths[cell_line].keys():
        if not os.path.exists(epigenetic_factor_paths[cell_line][histone_mark]['local_path']):
            download_file(epigenetic_factor_paths[cell_line][histone_mark])
        parse_node_encoding_file(
            epigenetic_factor_paths[cell_line][histone_mark]['local_path'],
            os.path.join(PARSED_EPIGENETIC_FILES_DIRECTORY, cell_line, histone_mark),
            hic_data_resolution
        )
        # If this histone mark we are considering for node features, append it to the paths list
        if histone_mark in epi_features_list:
            node_encoding_files.append(os.path.join(PARSED_EPIGENETIC_FILES_DIRECTORY, cell_line, histone_mark))




# Step 2: Create Dataset (Currently I am only creating dataset with base:encode-0 and target:rao-et-al)

# First we construct a string that is the dataset path
base = 'GM12878-encode-0'
target = 'GM12878-geo-raoetal'


dataset_name = 'base:{}_target:{}_c:{}_s:{}_b:{}_n:{}_rz:{}_sdz:{}_p:{}_r:{}_enc:{}_noi:{}_nirrm:{}_epi:{}_nconcat:{}/'.format(
    base,
    target,
    cropping_params['sub_mat'],
    cropping_params['stride'],    
    cropping_params['bounds'],
    normalization_params['norm'],
    normalization_params['remove_zeros'],
    normalization_params['set_diagonal_zero'],
    normalization_params['percentile'],
    normalization_params['rescale'],
    positional_encoding_method,
    noise_augmentation_method,
    non_informative_row_resolution_method,
    '+'.join(list(map(lambda x: x.split('/')[-1], node_encoding_files))),
    node_embedding_concat_method
)




print(dataset_name)

# We create a hash because the path length is too long, we remember the hash to name correspondances in an external JSON. 

dataset_name_hash = hashlib.sha1(dataset_name.encode('ascii')).hexdigest()

# save hash in the external json for later references
datasets_database = json.load(open('datasets.json'))

if not datasets_database.get(dataset_name_hash):
    datasets_database[dataset_name_hash] = dataset_name
    with open("datasets.json", "w") as outfile:
        json.dump(datasets_database, outfile, indent=4)

else:
    print('Entry already exists in the database')




dataset_path = os.path.join(DATASET_DIRECTORY, dataset_name_hash)


if not os.path.exists(os.path.join(dataset_path, 'train.npz')):
    create_dataset_from_hic_files(
        os.path.join(PARSED_HIC_FILES_DIRECTORY, base ,'resolution_{}'.format(hic_data_resolution)),
        os.path.join(PARSED_HIC_FILES_DIRECTORY, target ,'resolution_{}'.format(hic_data_resolution)),
        dataset_path,
        positional_encoding_method,
        node_encoding_files,
        node_embedding_concat_method,
        cropping_params,
        normalization_params,
        noise_augmentation_method,
        non_informative_row_resolution_method,
        ['test', 'valid', 'train'] 
    )
else:
   print('Dataset already exists!')

############################################## TRAIN AND RUN GRAPHIC MODEL ###############################################################
#Step3: Create the Model

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


model_name = 'graphic(embd-64)_l1loss+tvloss(e-04)_dhash:{}/'.format(
    dataset_name_hash
)

input_embedding_size = len(node_encoding_files) if node_embedding_concat_method != 'concat' else 2*len(node_encoding_files)

graphic_model = GrapHiC(
    HYPERPARAMETERS, device, model_name, 
    input_embedding_size=input_embedding_size
)

# Step 4: Run the main training and evaluation loop
run(
    graphic_model,
    os.path.join(dataset_path, 'train.npz'),
    os.path.join(dataset_path, 'valid.npz'),
    os.path.join(dataset_path, 'test.npz')
)
############################################################################################################################################

############################################## TRAIN AND RUN HiCPLUS MODEL ###############################################################
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")


# model_name = 'hicplus_dhash:{}/'.format(
#     dataset_name_hash
# )
# hicplus_model = HiCPlus(HYPERPARAMETERS, device, model_name)

# run(
#     hicplus_model,
#     os.path.join(dataset_path, 'train.npz'),
#     os.path.join(dataset_path, 'valid.npz'),
#     os.path.join(dataset_path, 'test.npz')
# )

#############################################################################################################################################















