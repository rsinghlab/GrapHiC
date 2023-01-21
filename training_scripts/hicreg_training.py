import sys
sys.path.append('../GrapHiC/')

import os
from src.utils import hic_file_paths, epigenetic_factor_paths
from src.epigentic_encodings import parse_node_encoding_file
from src.parse_hic_files import parse_hic_file, download_file
from src.utils import PARSED_EPIGENETIC_FILES_DIRECTORY, PARSED_HIC_FILES_DIRECTORY
from related_work.hic_reg import run

# Constants 
hic_data_resolution = 10000



# Step 1: Download and parse the HiC files
for hic_file_key in hic_file_paths.keys():
    if not os.path.exists(hic_file_paths[hic_file_key]['local_path']):
        download_file(hic_file_paths[hic_file_key])
    
    parse_hic_file(
        hic_file_paths[hic_file_key]['local_path'], 
        os.path.join(PARSED_HIC_FILES_DIRECTORY, hic_file_key),
        hic_data_resolution
    )

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


# Step 2: Run the script and generate the results 
run(
    'GM12878',
    10000,
    train=True
)