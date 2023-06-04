import os, math
import numpy as np
import pandas as pd
import cooler
from wrapt_timeout_decorator import * 

def distance(c1, c2):
    return math.sqrt((float(c1[0]) - float(c2[0]))**2 + (float(c1[1]) - float(c2[1]))**2)


def is_overlapping(coordinate, target_coordinates, rp=0):
    relaxation_parameter = math.sqrt(2*(rp**2))
    try:
        closest = sorted(list(map(lambda x: (x, distance(coordinate, x)), target_coordinates)), key = lambda y: y[1])[0]
    except:
        return ''
    
    coor, d = closest

    if d <= relaxation_parameter:
       return np.array2string(coor, separator=',')
    else:
        return ''


def read_chromosight_tsv_file(file_path):
    if os.path.exists(file_path):
        data = open(file_path).read().split('\n')[1:-1]
        data = np.array(list(map(lambda x: [x.split('\t')[i] for i in [6, 7]], data))).astype(np.int64)
        return data
    else: 
        return []



def create_genomic_bins(
        chromosome_name,
        resolution,
        size
    ):
    """
        The only currently supported type is 'bed' format which is chromosome_id, start, end
        So the function requires input of 'chromosome_name' chromosome name and 'resolution' resolution of of the file. 
        This function also requires size of the chromosome to estimate the maximum number of bins
    """
    chr_names = np.array([chromosome_name]*size)
    starts = (np.arange(0, size, 1, dtype=int))*resolution
    ends = (np.arange(1, size+1, 1, dtype=int))*resolution
    bins = {
        'chrom': chr_names,
        'start': starts,
        'end': ends
    }
    bins = pd.DataFrame(data=bins)
    return bins

def create_genomic_pixels(dense_matrix, upscale=255):
    """
        Converts a dense matrix into a .bed style sparse matrix file
        @params: dense_matrix <np.array>, input dense matrix
        @params: output_type <string>, output type, currently only supported style is bed style
    """
    
    lower_triangular_matrix_coordinates = np.tril_indices(dense_matrix.shape[0], k=-1)
    dense_matrix[lower_triangular_matrix_coordinates] = 0
    
    non_zero_indexes = np.nonzero(dense_matrix)
    bin_ones = non_zero_indexes[0]
    bin_twos = non_zero_indexes[1]
    counts = dense_matrix[np.nonzero(dense_matrix)]*upscale    
    pixels = {
        'bin1_id': bin_ones,
        'bin2_id': bin_twos,
        'count': counts
    }

    pixels = pd.DataFrame(data=pixels)

    return pixels

def create_cooler_file(sample, output_path, chr_name, resolution=10000):
    h, w = sample.shape
    dense_hic_file_genomic_bins = create_genomic_bins(chr_name, resolution, h)
    dense_hic_file_pixels_in_bins = create_genomic_pixels(sample)
    # This generates a cooler file in the provided output file path
    cooler.create_cooler(
        output_path, 
        dense_hic_file_genomic_bins, 
        dense_hic_file_pixels_in_bins,
        dtypes={"count":"int"},
        assembly="hg19"
    )


@timeout(60)
def run_chromosight(cooler_file):
    folder = '/'.join(cooler_file.split('/')[:-1])
    file_name = cooler_file.split('/')[-1].split('.')[0]
    
    loops_output_path = os.path.join(
        folder,
        '{}_loops'.format(file_name)
    )
    borders_output_path = os.path.join(
        folder,
        '{}_borders'.format(file_name)
    )
    hairpins_output_path = os.path.join(
        folder,
        '{}_hairpins'.format(file_name)
    )
    
    if not os.path.exists('{}.tsv'.format(borders_output_path)):
        cmd_path = 'chromosight detect --pattern=borders --pearson=0.3 --threads 1 {} {};'.format(
            cooler_file,
            borders_output_path
        )
        os.system(cmd_path)

    if not os.path.exists('{}.tsv'.format(hairpins_output_path)):
        cmd_path = 'chromosight detect --pattern=hairpins --pearson=0.4 --threads 1 {} {};'.format(
            cooler_file,
            hairpins_output_path
        )
        os.system(cmd_path)

    if not os.path.exists('{}.tsv'.format(loops_output_path)):
        cmd_path = 'chromosight detect --pattern=loops --threads 1 --min-dist 2000 --max-dist 200000 {} {};'.format(
            cooler_file, 
            loops_output_path,
        )
        os.system(cmd_path)
        
    return loops_output_path+'.tsv', borders_output_path+'.tsv', hairpins_output_path+'.tsv'



# create cooler files for all the available samples
def extract_features(
    samples,
    indexes,
    output_path
):
    all_loops_path = []
    all_borders_path = []
    all_hairpins_path = []
    # Create cooler files for all samples 
    for idx in range(samples.shape[0]):
        file_path = os.path.join(
            output_path, 
            'chrom:{}-i:{}-j:{}.cool'.format(
                indexes[idx][0],
                indexes[idx][2], 
                indexes[idx][3]
            )
        )
        
        sample = samples[idx, 0, :, :]        
        create_cooler_file(sample, file_path, str(indexes[idx][0]))
        l, b, h = run_chromosight(file_path)
        all_loops_path.append(l)
        all_borders_path.append(b)
        all_hairpins_path.append(h)
        
    return all_loops_path, all_borders_path, all_hairpins_path




def compare_features(base_feature_files, target_feature_files, feature_rp):
    f1_scores = []
    
    for base, target in zip(base_feature_files, target_feature_files):
        base = read_chromosight_tsv_file(base)
        target = read_chromosight_tsv_file(target)
        p, r, f1, acc = overlap_analysis(base, target, feature_rp)
        f1_scores.append(f1)
    
    return {'f1': f1_scores}



def overlap_analysis(base, target, rp):
    multi_map = {}
    for coordinate in base:
        coordinate = is_overlapping(coordinate, target, rp)
        if coordinate not in multi_map.keys(): 
            multi_map[coordinate] = 0
            
        multi_map[coordinate] += 1

    fp = multi_map[''] if '' in multi_map.keys() else 0  # When we couldnt find any mapping in target
    fn = len(target)
    tp = 0.000000001
    mm = -1

    for key in multi_map.keys():
        if multi_map[key] >= 1 and key != '':
            tp += 1
            fn -= 1
        if multi_map[key] >= 2:
            mm += 1


    precision = tp/(tp+fp)
    recall =  tp/(tp+fn)
    
    f1 = (2*precision*recall)/(precision + recall)

    accuracy = tp/(tp+fn+fp)

    return precision, recall, f1, accuracy





























