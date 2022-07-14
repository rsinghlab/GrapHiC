import numpy as np
import pandas as pd
import os

from src.utils import load_hic_file



def read_bed_file(path_to_bed_file, filter_chrom=None):
    '''
        Read the difficult to read regions bed file and returns the positions based on chrom provided
        @params:path_to_bed_file <string>, path to the bed file
        @params: filter_chrom <string> chromosome to filter out of the file
        @returns: <dict> a dictionary that contains chromosomes as keys and difficult to map regions as values 
    '''
    # read the data
    file_data = open(path_to_bed_file).read().split('\n')
    print(len(file_data))
    
    file_data = list(map(lambda x: x.split('\t'), file_data))
        
    # convert it into a dataframe and return 
    df = pd.DataFrame(file_data, columns = ['chromosome', 'start', 'end'])

    df = df.loc[df['chromosome'] == filter_chrom]


    return df






def get_non_informative_regions_from_hic(path_to_hic_file):
    data = load_hic_file(path_to_hic_file)
    compact = data['compact']
    data = data['hic']
    all_idxs = np.array(list(range(data.shape[0])))

    non_informative = np.setdiff1d(all_idxs, compact, assume_unique=True)
    

    return non_informative






def compare_regions(hic_files_path, bed_file, chromosome='1', hic_resolution=10000):
    hic_non_informative = get_non_informative_regions_from_hic(
        os.path.join(hic_files_path, 'chr{}.npz'.format(chromosome))
    )
    hic_non_informative = hic_non_informative*hic_resolution

    bed_difficult_regions = read_bed_file(bed_file, chromosome)
    count = 0
    total = 0

    for index, row in bed_difficult_regions.iterrows():
        total += 1
        start = int(row['start'])
        end = int(row['end'])

        print(start, end)

        for hic_region in hic_non_informative:
            print('HiC Region:', hic_region, hic_region+hic_resolution) 
            
            if start >= hic_region and start <= (hic_region + hic_resolution):
                count += 1
            elif end >= hic_region and end <= (hic_region + hic_resolution):
                count += 1
            else:
                continue
    
    overlap_percentage = ((count*1.0)/(total*1.0)) * 100

    return overlap_percentage


    #print(hic_non_informative)
































































