'''
    This file contains the matrix operations we have used

'''
import numpy as np


except_chr = {'hsa': {'X': 23, 23: 'X'}, 'mouse': {'X': 20, 20: 'X'}}

def compactM(matrix, compact_idx, verbose=False):
    """
        Compacting matrix according to the index list.
        @params: matrix <np.array> Full sized matrix, that needs to be compressed
        @params: compact_idx <list> Indexes of rows that contain data
        @params: verbose <boolean> Debugging print statements
        @returns: <np.array> Condesed matrix with zero arrays pruned 
    """

    compact_size = len(compact_idx)
    
    result = np.zeros((compact_size, compact_size)).astype(matrix.dtype)
    
    if verbose: print('Compacting a', matrix.shape, 'shaped matrix to', result.shape, 'shaped!')
    
    for i, idx in enumerate(compact_idx):
        result[i, :] = matrix[idx][compact_idx]
    
    return result



def spreadM(c_mat, compact_idx, full_size, convert_int=True, verbose=False):
    """spreading matrix according to the index list (a reversed operation to compactM)."""
    result = np.zeros((full_size, full_size)).astype(c_mat.dtype)
    if convert_int: result = result.astype(np.int)
    if verbose: print('Spreading a', c_mat.shape, 'shaped matrix to', result.shape, 'shaped!' )
    for i, s_idx in enumerate(compact_idx):
        result[s_idx, compact_idx] = c_mat[i]
    return result







def together(matlist, indices, corp=0, species='hsa', tag='HiC'):
    chr_nums = sorted(list(np.unique(indices[:,0])))
    # convert last element to str 'X'
    if chr_nums[-1] in except_chr[species]: chr_nums[-1] = except_chr[species][chr_nums[-1]]
    print(f'{tag} data contain {chr_nums} chromosomes')
    _, h, w = matlist[0].shape
    results = dict.fromkeys(chr_nums)
    for n in chr_nums:
        # convert str 'X' to 23
        num = except_chr[species][n] if isinstance(n, str) else n
        loci = np.where(indices[:,0] == num)[0]
        sub_mats = matlist[loci]
        index = indices[loci]
        width = index[0,1]
        full_mat = np.zeros((width, width))
        for sub, pos in zip(sub_mats, index):
            i, j = pos[-2], pos[-1]
            if corp > 0:
                sub = sub[:, corp:-corp, corp:-corp]
                _, h, w = sub.shape
            full_mat[i:i+h, j:j+w] = sub
        results[n] = full_mat
    return results

def divide(mat, chr_num, cropping_params, verbose=False):
    """
        @params: mat <np.array> HiC matrix that needs to be chunked up
        @params: chr_num <string> Chromosome number of the input HiC matrix
        @params: cropping_params <dict> contains the required parameters to crop the matrix
        @params: verbose <boolean> Debugging print statements
        @returns: list<np.array>, list first return is chunked up matrices in a list and second 
                return contains the positions 
    """
    chr_str = str(chr_num)
    result = []
    index = []
    size = mat.shape[0]
    
    stride = cropping_params['stride']
    chunk_size = cropping_params['chunk_size']
    bound = cropping_params['bounds']
    padding = cropping_params['padding']

    if (stride < chunk_size and padding):
        pad_len = (chunk_size - stride) // 2
        mat = np.pad(mat, ((pad_len,pad_len), (pad_len,pad_len)), 'constant')
    # mat's shape changed, update!
    height, width = mat.shape
    assert height == width, 'Now, we just assumed matrix is squared!'
    
    for i in range(0, height, stride):
        for j in range(0, width, stride):
            if abs(i-j)<=bound and i+chunk_size<height and j+chunk_size<width:
                subImage = mat[i:i+chunk_size, j:j+chunk_size]
                result.append([subImage])
                index.append((chr_num, size, i, j))
    result = np.array(result)
    if verbose: print(f'[Chr{chr_str}] Dividing HiC matrix ({size}x{size}) into {len(result)} samples with chunk={chunk_size}, stride={stride}, bound={bound}')
    index = np.array(index)
    return result, index
