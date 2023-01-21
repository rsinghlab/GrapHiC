import os, shutil, gzip
import numpy as np
import wget

hic_data_resolution = 10000


BASE_DIRECTORY = '/users/gmurtaza/GrapHiC/'
HIC_FILES_DIRECTORY = '/users/gmurtaza/data/gmurtaza/hic_datasets/' # Recommended that this path is on some larger storage device
PARSED_HIC_FILES_DIRECTORY = '/users/gmurtaza/data/gmurtaza/parsed_hic_datasets/'
EPIGENETIC_FILES_DIRECTORY = '/users/gmurtaza/data/gmurtaza/epigenetic_datasets/' # Recommended that this path is on some larger storage device
PARSED_EPIGENETIC_FILES_DIRECTORY = '/users/gmurtaza/data/gmurtaza/parsed_epigenetic_datasets/' # Recommended that this path is on some larger storage device
DATASET_DIRECTORY = '/users/gmurtaza/data/gmurtaza/generated_datasets'

PREDICTED_FILES_DIRECTORY = '/users/gmurtaza/data/gmurtaza/predicted_data/'
GENERATED_RESULTS_DIRECTORY = '/users/gmurtaza/data/gmurtaza/results/'

WEIGHTS_DIRECTORY = os.path.join(BASE_DIRECTORY, 'weights') # Recommended to keep weights on the same directory
GENERATED_DATA_DIRECTORY = os.path.join(BASE_DIRECTORY, 'outputs')

JAR_LOCATION = os.path.join(BASE_DIRECTORY, 'other_tools', '3DMax.jar')

# Dataset file paths

# These files should exist, (currently not using all of them but would at some point)
hic_file_paths = {
    # 'GM12878-geo-raoetal': {
    #         'local_path' : os.path.join(HIC_FILES_DIRECTORY, 'GM12878', 'geo-raoetal.hic'),
    #         'remote_path': 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63525&format=file&file=GSE63525%5FGM12878%5Finsitu%5Fprimary%2Breplicate%5Fcombined%5F30%2Ehic'
    # },
    # 'HMEC-geo-raoetal': {
    #         'local_path' : os.path.join(HIC_FILES_DIRECTORY, 'HMEC', 'geo-raoetal.hic'),
    #         'remote_path': 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63525&format=file&file=GSE63525%5FHMEC%5Fcombined%5F30%2Ehic'
    # },
    # 'HUVEC-geo-raoetal': {
    #         'local_path' : os.path.join(HIC_FILES_DIRECTORY, 'HUVEC', 'geo-raoetal.hic'),
    #         'remote_path': 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63525&format=file&file=GSE63525%5FHUVEC%5Fcombined%5F30%2Ehic'
    # },
    # 'IMR90-geo-raoetal': {
    #         'local_path' : os.path.join(HIC_FILES_DIRECTORY, 'IMR90', 'geo-raoetal.hic'),
    #         'remote_path': 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63525&format=file&file=GSE63525%5FIMR90%5Fcombined%5F30%2Ehic'
    # },
    # 'K562-geo-raoetal': {
    #         'local_path' : os.path.join(HIC_FILES_DIRECTORY, 'K562', 'geo-raoetal.hic'),
    #         'remote_path': 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63525&format=file&file=GSE63525%5FK562%5Fcombined%5F30%2Ehic'
    # },
    # 'KBM7-geo-raoetal': {
    #         'local_path' : os.path.join(HIC_FILES_DIRECTORY, 'KBM7', 'geo-raoetal.hic'),
    #         'remote_path': 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63525&format=file&file=GSE63525%5FKBM7%5Fcombined%5F30%2Ehic'
    # },
    # 'NHEK-geo-raoetal': {
    #         'local_path' : os.path.join(HIC_FILES_DIRECTORY, 'NHEK', 'geo-raoetal.hic'),
    #         'remote_path': 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63525&format=file&file=GSE63525%5FNHEK%5Fcombined%5F30%2Ehic'
    # },
    # 'GM12878-encode-0': {
    #         'local_path' : os.path.join(HIC_FILES_DIRECTORY, 'GM12878', 'encode-0.hic'),
    #         'remote_path': 'https://www.encodeproject.org/files/ENCFF799QGA/@@download/ENCFF799QGA.hic'
    # },
    # 'GM12878-encode-1': {
    #         'local_path' : os.path.join(HIC_FILES_DIRECTORY, 'GM12878', 'encode-1.hic'),
    #         'remote_path': 'https://www.encodeproject.org/files/ENCFF473CAA/@@download/ENCFF473CAA.hic'
    # },
    # 'GM12878-encode-2': {
    #         'local_path' : os.path.join(HIC_FILES_DIRECTORY, 'GM12878', 'encode-2.hic'),
    #         'remote_path': 'https://www.encodeproject.org/files/ENCFF227XJZ/@@download/ENCFF227XJZ.hic'
    # },
    # 'GM12878-geo-026': {
    #         'local_path' : os.path.join(HIC_FILES_DIRECTORY, 'GM12878', 'geo-026.hic'),
    #         'remote_path': 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM1551575&format=file&file=GSM1551575%5FHIC026%5F30%2Ehic'
    # },
    # 'GM12878-geo-033': {
    #         'local_path' : os.path.join(HIC_FILES_DIRECTORY, 'GM12878', 'geo-033.hic'),
    #         'remote_path': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1551nnn/GSM1551582/suppl/GSM1551582_HIC033_30.hic'
    # },
    # 'K562-geo-073': {
    #         'local_path' : os.path.join(HIC_FILES_DIRECTORY, 'K562', 'geo-073.hic'),
    #         'remote_path': 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM1551622&format=file&file=GSM1551622%5FHIC073%5F30%2Ehic'
    # },
    # 'HUVEC-geo-059': {
    #         'local_path' : os.path.join(HIC_FILES_DIRECTORY, 'HUVEC', 'geo-068.hic'),
    #         'remote_path': 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM1551608&format=file&file=GSM1551608%5FHIC059%5F30%2Ehic'
    # }
    'IMR90-geo-057': {
            'local_path' : os.path.join(HIC_FILES_DIRECTORY, 'IMR90', 'geo-057.hic'),
            'remote_path': 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM1551606&format=file&file=GSM1551606%5FHIC057%5F30%2Ehic'
    },    
}


epigenetic_factor_paths = {
    # 'GM12878': {
    #     'H3K27AC': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneGm12878H3k27acStdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'h3k27ac.bigwig'),
    #     },
    #     'H3K27ME3': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneGm12878H3k27me3StdSigV2.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'h3k27me3.bigwig'),
    #     },
    #     'H3K36ME3': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneGm12878H3k36me3StdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'h3k36me3.bigwig'),
    #     },
    #     'H3K4ME1': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneGm12878H3k4me1StdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'h3k4me1.bigwig'),
    #     },
    #     'H3K4ME2': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneGm12878H3k4me2StdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'h3k4me2.bigwig'),
    #     },
    #     'H3K4ME3': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneGm12878H3k4me3StdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'h3k4me3.bigwig'),
    #     },
    #     'H3K79ME2': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneGm12878H3k79me2StdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'h3k79me2.bigwig'),
    #     },
    #     'H3K9AC': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneGm12878H3k9acStdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'h3k9ac.bigwig'),
    #     },
    #     'H4K20ME1': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneGm12878H4k20me1StdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'h4k20me1.bigwig'),
    #     },
    #     'H3K9ME3': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneGm12878H3k9me3StdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'h3k9me3.bigwig'),
    #     },
    #     'DNASE-Seq':{
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeOpenChromDnase/wgEncodeOpenChromDnaseGm12878Sig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'dnase.bigwig'),
    #     }, 
    #     'CTCF': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneGm12878CtcfStdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'ctcf.bigwig')
    #     },
    #     'RNA-Pol2': {
    #         'remote_path': 'https://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeOpenChromChip/wgEncodeOpenChromChipGm12878Pol2Sig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'rnapol2.bigwig')
    #     }, 
    #     'RAD-21': {
    #         'remote_path': 'https://encode-public.s3.amazonaws.com/2012/07/01/bb401e4f-91f5-4ddc-ac2b-2b36a56ec114/ENCFF000WCT.bigWig',
    #         'local_path': os.path.join(EPIGENETIC_FILES_DIRECTORY, 'GM12878', 'rad21.bigwig')
    #     },
    # },
    # 'K562':{
    #     'H3K27AC': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneK562H3k27acStdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'K562', 'h3k27ac.bigwig'),
    #     },
    #     'H3K27ME3': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneK562H3k27me3StdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'K562', 'h3k27me3.bigwig'),
    #     },
    #     'H3K36ME3': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneK562H3k36me3StdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'K562', 'h3k36me3.bigwig'),
    #     },
    #     'H3K4ME1': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneK562H3k4me1StdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'K562', 'h3k4me1.bigwig'),
    #     },
    #     'H3K4ME2': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneK562H3k4me2StdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'K562', 'h3k4me2.bigwig'),
    #     },
    #     'H3K4ME3': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneK562H3k4me3StdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'K562', 'h3k4me3.bigwig'),
    #     },
    #     'H3K79ME2': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneK562H3k79me2StdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'K562', 'h3k79me2.bigwig'),
    #     },
    #     'H3K9AC': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneK562H3k9acStdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'K562', 'h3k9ac.bigwig'),
    #     },
    #     'H4K20ME1': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneK562H4k20me1StdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'K562', 'h4k20me1.bigwig'),
    #     },
    #     'H3K9ME3': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneK562H3k9me3StdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'K562', 'h3k9me3.bigwig'),
    #     },
    #     'DNASE-Seq':{
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeOpenChromDnase/wgEncodeOpenChromDnaseK562Sig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'K562', 'dnase.bigwig'),
    #     }, 
    #     'CTCF': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneK562CtcfStdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'K562', 'ctcf.bigwig')
    #     },
    #     'RNA-Pol2': {
    #         'remote_path': 'https://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeOpenChromChip/wgEncodeOpenChromChipK562Pol2Sig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'K562', 'rnapol2.bigwig')
    #     }, 
    #     'RAD-21': {
    #         'remote_path': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM935nnn/GSM935319/suppl/GSM935319_hg19_wgEncodeSydhTfbsK562Rad21StdSig.bigWig',
    #         'local_path': os.path.join(EPIGENETIC_FILES_DIRECTORY, 'K562', 'rad21.bigwig')
    #     },
    # },
    # 'HUVEC':{
    #     'H3K27AC': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneHuvecH3k27acStdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'HUVEC', 'h3k27ac.bigwig'),
    #     },
    #     'H3K27ME3': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneHuvecH3k27me3StdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'HUVEC', 'h3k27me3.bigwig'),
    #     },
    #     'H3K36ME3': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneHuvecH3k36me3StdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'HUVEC', 'h3k36me3.bigwig'),
    #     },
    #     'H3K4ME1': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneHuvecH3k4me1StdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'HUVEC', 'h3k4me1.bigwig'),
    #     },
    #     'H3K4ME2': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneHuvecH3k4me2StdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'HUVEC', 'h3k4me2.bigwig'),
    #     },
    #     'H3K4ME3': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneHuvecH3k4me3StdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'HUVEC', 'h3k4me3.bigwig'),
    #     },
    #     'H3K79ME2': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneHuvecH3k79me2Sig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'HUVEC', 'h3k79me2.bigwig'),
    #     },
    #     'H3K9AC': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneHuvecH3k9acStdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'HUVEC', 'h3k9ac.bigwig'),
    #     },
    #     'H4K20ME1': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneHuvecH4k20me1StdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'HUVEC', 'h4k20me1.bigwig'),
    #     },
    #     'H3K9ME3': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneHuvecH3k9me1StdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'HUVEC', 'h3k9me3.bigwig'),
    #     },
    #     'DNASE-Seq':{
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeOpenChromDnase/wgEncodeOpenChromDnaseHuvecSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'HUVEC', 'dnase.bigwig'),
    #     },
    #     'CTCF': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneHuvecCtcfStdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'HUVEC', 'ctcf.bigwig')
    #     },
    #     'RNA-Pol2': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/wgEncodeBroadHistoneHuvecPol2bStdSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'HUVEC', 'rnapol2.bigwig')
    #     }, 
    #     'RAD-21': {
    #         'remote_path': 'https://dbarchive.biosciencedbc.jp/kyushu-u/hg19/eachData/bw/SRX2559013.bw',
    #         'local_path': os.path.join(EPIGENETIC_FILES_DIRECTORY, 'HUVEC', 'rad21.bigwig')
    #     },
    # },
    # 'IMR90':{
    #     'H3K27AC': {
    #         'remote_path': 'https://www.encodeproject.org/files/ENCFF801HPN/@@download/ENCFF801HPN.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'IMR90', 'h3k27ac.bigwig'),
    #     },
    #     'H3K27ME3': {
    #         'remote_path': 'https://www.encodeproject.org/files/ENCFF328UMQ/@@download/ENCFF328UMQ.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'IMR90', 'h3k27me3.bigwig'),
    #     },
    #     'H3K36ME3': {
    #         'remote_path': 'https://www.encodeproject.org/files/ENCFF187AAC/@@download/ENCFF187AAC.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'IMR90', 'h3k36me3.bigwig'),
    #     },
    #     'H3K4ME1': {
    #         'remote_path': 'https://www.encodeproject.org/files/ENCFF159UXL/@@download/ENCFF159UXL.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'IMR90', 'h3k4me1.bigwig'),
    #     },
    #     'H3K4ME2': {
    #         'remote_path': 'https://www.encodeproject.org/files/ENCFF502IRW/@@download/ENCFF502IRW.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'IMR90', 'h3k4me2.bigwig'),
    #     },
    #     'H3K4ME3': {
    #         'remote_path': 'https://www.encodeproject.org/files/ENCFF254FBR/@@download/ENCFF254FBR.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'IMR90', 'h3k4me3.bigwig'),
    #     },
    #     'H3K79ME2': {
    #         'remote_path': 'https://www.encodeproject.org/files/ENCFF373KFT/@@download/ENCFF373KFT.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'IMR90', 'h3k79me2.bigwig'),
    #     },
    #     'H3K9AC': {
    #         'remote_path': 'https://www.encodeproject.org/files/ENCFF606CWZ/@@download/ENCFF606CWZ.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'IMR90', 'h3k9ac.bigwig'),
    #     },
    #     'H4K20ME1': {
    #         'remote_path': 'https://www.encodeproject.org/files/ENCFF025PLZ/@@download/ENCFF025PLZ.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'IMR90', 'h4k20me1.bigwig'),
    #     },
    #     'H3K9ME3': {
    #         'remote_path': 'https://www.encodeproject.org/files/ENCFF545BBZ/@@download/ENCFF545BBZ.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'IMR90', 'h3k9me3.bigwig'),
    #     },
    #     'DNASE-Seq':{
    #         'remote_path': 'https://www.encodeproject.org/files/ENCFF291DOH/@@download/ENCFF291DOH.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'IMR90', 'dnase.bigwig'),
    #     },
    #     'CTCF': {
    #         'remote_path': 'https://www.encodeproject.org/files/ENCFF583IZF/@@download/ENCFF583IZF.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'IMR90', 'ctcf.bigwig')
    #     },
    #     'RNA-Pol2': {
    #         'remote_path': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeSydhTfbs/wgEncodeSydhTfbsImr90Pol2IggrabSig.bigWig',
    #         'local_path' : os.path.join(EPIGENETIC_FILES_DIRECTORY, 'IMR90', 'rnapol2.bigwig')
    #     }, 
    #     'RAD-21': {
    #         'remote_path': 'https://www.encodeproject.org/files/ENCFF115RIK/@@download/ENCFF115RIK.bigWig',
    #         'local_path': os.path.join(EPIGENETIC_FILES_DIRECTORY, 'IMR90', 'rad21.bigwig')
    #     },
    # }
}












def compress_file(file_path, clean=False):
    '''
        This function compresses the file found on the provided path
        @params: file_path <string>, path to the file that needs to be compressed
        @params: clean <bool>, to keep or remove the original file
        @returns: return the path of the generated .gzip file
    '''
    compressed_file_path = file_path+'.gz'
    with open(file_path, 'rb') as f_in, gzip.open(compressed_file_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    if clean:
        if os.path.exists(file_path):
            os.remove(file_path)
    return compressed_file_path



def load_hic_file(path, format='.npz'):
    '''
        A wrapper that reads the HiC file and returns it, it essentially is created to 
        later support multiple HiC file formats, currently the only utilized file format is 
        .npz. 

        @params: 
    '''
    if format == '.npz':
        return np.load(path, allow_pickle=True)
    else:
        print('Provided file format is invalid')
        exit(1)



def delete_files(folder_path):
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)




def create_entire_path_directory(path):
    """
        Given a path, creates all the missing directories on the path and 
        ensures that the given path is always a valid path
        @params: <string> path, full absolute path, this function can act wonkily if given a relative path
        @returns: None
    """
    
    path = path.split('/')
    curr_path = '/'
    for dir in path:
        curr_path = os.path.join(curr_path, dir)
        if os.path.exists(curr_path):
            continue
        else:
            os.mkdir(curr_path)



def download_file(file_paths):
    print('Downloading file from {}'.format(file_paths['remote_path']))
    
    create_entire_path_directory('/'.join(file_paths['local_path'].split('/')[:-1]))
    wget.download(file_paths['remote_path'], file_paths['local_path'])   

    print('File downloaded at {}'.format(file_paths['local_path']))
    



def get_required_node_encoding_files_paths(cell_line, histone_marks):
    node_encoding_files = []
    for histone_mark in histone_marks:
        node_encoding_files.append(
            os.path.join(
                PARSED_EPIGENETIC_FILES_DIRECTORY,
                cell_line,
                histone_mark
            )
        )
    return node_encoding_files


create_entire_path_directory(WEIGHTS_DIRECTORY)
create_entire_path_directory(GENERATED_DATA_DIRECTORY)
create_entire_path_directory(PREDICTED_FILES_DIRECTORY)






