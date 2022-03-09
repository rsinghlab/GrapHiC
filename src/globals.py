
NUM_CHROMOSOMES = 23
RESOLUTION = 10000
NORMALIZATION = 'KR'
PERCENTILE = 99.95



PATH_TO_HIC_RAW_FILES = '/media/murtaza/ubuntu/updated_hic_data/data/hic_datasets/'
PATH_TO_GM12878_HIC_RAW_FILES = '/media/murtaza/ubuntu/updated_hic_data/data/hic_datasets/'
PATH_TO_HIC_IMR90_RAW_FILES = '/media/murtaza/ubuntu/updated_hic_data/data/hic_datasets/'
PATH_TO_HIC_K562_RAW_FILES = '/media/murtaza/ubuntu/updated_hic_data/data/hic_datasets/'


PATH_TO_GRAPHIC_DATASETS = '/media/murtaza/ubuntu/GrapHiC_datasets/'

PATH_TO_DATASET_STATISTICS_FILE= 'dataset_stats.csv'

SUB_MATRIX_SIZE = 1000


DATASETS = {
    'LOW_RES': {
        'GM12878': ['encode-0', 'encode-1', 'encode-2', 'encode-3', 'encode-4', 'hic010', 'hic026', 'hic031', 'hic033', 'hic046'],
        'IMR90': ['hic050', 'hic052', 'hic056', 'hic057'],
        'K562': ['hic070', 'hic071', 'hic073', 'hic074']
    },
    'HIGH_RES': {
        'GM12878': ['primary'],
        'IMR90': ['primary'],
        'K562': ['primary']
    }
}

CHROMOSOME_SIZES = {
    '1': 249250621, '2': 243199373, '3': 198022430, '4': 191154276, '5': 180915260, 
    '6': 171115067, '7': 159138663, '8': 146364022, '9': 141213431, '10': 135534747, 
    '11': 135006516, '12': 133851895, '13': 115169878, '14': 107349540, '15': 102531392, 
    '16': 90354753, '17': 81195210, '18': 78077248, '19': 59128983, '20': 63025520, 
    '21': 48129895, '22': 51304566 
}











DATASET_REQUIREMENTS = {
    'cell_lines' : ['GM12878'],
    'base_datasets': ['encode-0', 'hic010'],
    'noise_type': 'NONE',
    'chroms': list(range(1,23))
}










