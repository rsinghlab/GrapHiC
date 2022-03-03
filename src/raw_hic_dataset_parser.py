'''
    This script file does three things:
    1) It parases out all the chromosome files from the raw .hic file at 10KB resolution
    2) It calculates the cutoff value for all the chromosome files and stores them in a .csv file
    3) Generates a file that contains the statistics of all the datasets files we have collected
'''



import os, struct, glob
import hicstraw
from src import globals
import numpy as np
from scipy.sparse import csr_matrix

def readcstr(f):
    buf = ""
    while True:
        b = f.read(1)
        b = b.decode('utf-8', 'backslashreplace')
        if b is None or b == '\0':
            return str(buf)
        else:
            buf = buf + b


'''
    Reads the HiC Header and returns an object that contains the information
'''

def read_hic_header(hicfile):
    if not os.path.exists(hicfile):
        return None  # probably a cool URI

    req = open(hicfile, 'rb')
    magic_string = struct.unpack('<3s', req.read(3))[0]
    req.read(1)
    if (magic_string != b"HIC"):
        return None  # this is not a valid .hic file

    info = {}
    version = struct.unpack('<i', req.read(4))[0]
    info['version'] = str(version)

    masterindex = struct.unpack('<q', req.read(8))[0]
    info['Master index'] = str(masterindex)

    genome = ""
    c = req.read(1).decode("utf-8")
    while (c != '\0'):
        genome += c
        c = req.read(1).decode("utf-8")
    info['Genome ID'] = str(genome)

    nattributes = struct.unpack('<i', req.read(4))[0]
    attrs = {}
    for i in range(nattributes):
        key = readcstr(req)
        value = readcstr(req)
        attrs[key] = value
    info['Attributes'] = attrs

    nChrs = struct.unpack('<i', req.read(4))[0]
    chromsizes = {}
    for i in range(nChrs):
        name = readcstr(req)
        length = struct.unpack('<i', req.read(4))[0]
        if name != 'ALL':
            chromsizes[name] = length

    info['chromsizes'] = chromsizes

    info['Base pair-delimited resolutions'] = []
    nBpRes = struct.unpack('<i', req.read(4))[0]
    for i in range(nBpRes):
        res = struct.unpack('<i', req.read(4))[0]
        info['Base pair-delimited resolutions'].append(res)

    info['Fragment-delimited resolutions'] = []
    nFrag = struct.unpack('<i', req.read(4))[0]
    for i in range(nFrag):
        res = struct.unpack('<i', req.read(4))[0]
        info['Fragment-delimited resolutions'].append(res)

    return info


def calculate_percentile(counts):
    counts = counts[np.where(counts != 0)]
    return np.percentile(counts, globals.PERCENTILE)






'''
    Parses out a dense chromosome from the chromosome object returned by the 
    straw parser
    @params: <obj> chrom straw chromosome object
    @params: <int> chrom_size chromosome size used to construct the dense array
    @returns: <np.array> dense 2D numpy array
'''

def straw_chrom_obj_to_dense_matrix(chrom, chrom_size):
    rows = [r.binX//globals.RESOLUTION for r in chrom]
    cols = [r.binY//globals.RESOLUTION for r in chrom]
    counts = np.array([r.counts for r in chrom])
    counts[np.isnan(counts)] = 0

    percentile = calculate_percentile(counts)
    
    read_counts = np.sum(counts)

    print(percentile, read_counts)

    N = chrom_size//globals.RESOLUTION + 1

    mat = csr_matrix((counts, (rows, cols)), shape=(N,N))
    mat = csr_matrix.todense(mat)
    mat = mat.T
    mat = mat + np.tril(mat, -1).T

    mat = mat.astype(int)

    return mat, percentile, read_counts




'''
    This function parses out all the chromosomes from the .hic file
    @params: <string> file_path path to the .hic file 
    @params: <string> output_path path to the output folder 
    @returns None
'''

def parse_out_chromosomes_from_hic_file(file_path, output_folder_path, noise_type='NONE'):
    print("Parsing file {}!".format(file_path))

    dataset_id = '-'.join(file_path.split('/')[-1].split('.')[0].split('_'))
    cell_line = file_path.split('/')[-3]

    print(dataset_id)

    statistics = {
        'cell_line': cell_line,
        'dataset_id': dataset_id,
        'percentile': 0,
        'read_counts': 0,
    }
    percentiles = []
    read_counts = []

    for chromosome in range(1, globals.NUM_CHROMOSOMES):
        chrom_size = read_hic_header(file_path)['chromsizes'][str(chromosome)]
        chrom = hicstraw.straw(
            'observed', 'KR', file_path,
            str(chromosome), str(chromosome), 'BP', globals.RESOLUTION
        )
        chrom, percentile, read_count = straw_chrom_obj_to_dense_matrix(chrom, chrom_size)

        percentiles.append(percentile)
        read_counts.append(read_count)

        output_path = os.path.join(output_folder_path, 
                              '{}_{}_{}_chr{}.npz'.format(noise_type, cell_line ,dataset_id, chromosome))
        
        print('Saving file:', output_path, ' of shape {}'.format(chrom.shape))
        np.savez_compressed(output_path, hic=chrom)


    statistics['percentile'] = np.mean(percentiles)
    statistics['read_counts'] = np.sum(read_counts)

    print("Done with file {}!".format(file_path))
    
    return statistics
    


'''
    Just a wrapper function to run the parse hic function on all the .hic files in
    my storage directory
    @params: path_to_base_directory <string> path to the base directory 
    @returns: None
'''
def parse_out_all_hic_files(path_to_base_directory):
    hic_files = glob.glob(os.path.join(path_to_base_directory, '**/*.hic'), recursive=True)
    #hic_files = list(filter(lambda x: 'K562' not in x and 'IMR90' not in x, hic_files))

    for hic_file in hic_files:
        stats = parse_out_chromosomes_from_hic_file(hic_file, os.path.join(globals.PATH_TO_GRAPHIC_DATASETS, 'raw'))
        with open('dataset_stats.csv', 'a+') as f:
            f.write('{},{},{},{}\n'.format(
                stats['cell_line'],
                stats['dataset_id'],
                stats['percentile'],
                stats['read_counts']
            ))











