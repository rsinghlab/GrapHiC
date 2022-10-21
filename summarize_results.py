import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


for results_for in ['GenomeDISCO', 'HiCRep', 'HiC-Spector', 'QuASAR-Rep']:
	cell_line_to_sparsity = {
		'GM12878-geo-026': '1/10',
		'GM12878-encode-2': '1/24',
		'GM12878-encode-0': '1/44',
		'GM12878-encode-1': '1/50',
		'GM12878-geo-033': '1/100',
  
	}

	parsed_data = {
		'graphic_best':{
			'GM12878-encode-0': [],
			'GM12878-encode-1': [],
			'GM12878-encode-2': [],
			'GM12878-geo-026': [],
			'GM12878-geo-033': [],
			'Name': 'GrapHiC',
			'Marker': "o",
			'Color': '#845EC2',
		},
		'graphic_positional':{
			'GM12878-encode-0': [],
			'GM12878-encode-1': [],
			'GM12878-encode-2': [],
			'GM12878-geo-026': [],
			'GM12878-geo-033': [],
			'Name': 'GrapHiC-Positional',
			'Marker': ">",
			'Color': '#008F7A',
		},
		'graphic_rad21':{
			'GM12878-encode-0': [],
			'GM12878-encode-1': [],
			'GM12878-encode-2': [],
			'GM12878-geo-026': [],
			'GM12878-geo-033': [],
			'Name': 'GrapHiC-RAD21',
			'Marker': "8",
			'Color': '#0089BA',
		},
		'graphic_baseline':{
			'GM12878-encode-0': [],
			'GM12878-encode-1': [],
			'GM12878-encode-2': [],
			'GM12878-geo-026': [],
			'GM12878-geo-033': [],
			'Name': 'GrapHiC-Baseline',
			'Marker': "s",
			'Color': '#B0A8B9',
		},
		# 'hicreg':{
		# 	'GM12878-encode-0': [],
		# 	'GM12878-encode-1': [],
		# 	'GM12878-encode-2': [],
		# 	'GM12878-geo-026': [],
		# 	'GM12878-geo-033': [],
		# 	'Name': 'HiCReg',
		# 	'Marker': "P",
		# 	'Color': '#F9F871',
		# },
		# 'hicnn':{
		# 	'GM12878-encode-0': [],
		# 	'GM12878-encode-1': [],
		# 	'GM12878-encode-2': [],
		# 	'GM12878-geo-026': [],
		# 	'GM12878-geo-033': [],
		# 	'Name': 'HiCNN',
		# 	'Marker': "X",
		# 	'Color': '#00C9A7',
		# }
	}


	base_path = '/users/gmurtaza/data/gmurtaza/results/hic_similarity_results'
	all_folders = list(map(lambda x: os.path.join(base_path, x), os.listdir(base_path)))

	for method in parsed_data.keys():
		for chrom_path in all_folders:
			if 'chr9' in chrom_path: 
				continue

			for cell_line in os.listdir(chrom_path):
				genome_disco_path = os.path.join(chrom_path, cell_line, 'results/reproducibility/{}'.format(results_for))
				if not os.path.exists(genome_disco_path):
					continue
				for result_file in os.listdir(genome_disco_path):
					result_file_path = os.path.join(genome_disco_path, result_file)
					if method in result_file_path:
						data = open(result_file_path).read().split('\n')[0].split('\t')
						try:
							parsed_data[method][cell_line].append(float(data[-1]))
						except:
							continue

	# Parsed out all results, plotting them now 
	print(results_for)


	fig = plt.figure()

	#fig.set_size_inches(18.5, 10.5)

	x = list(cell_line_to_sparsity.values())

	for method in parsed_data.keys():
		line_label = parsed_data[method]['Name']
		y = []
		ystd = []
		for cell_line in cell_line_to_sparsity.keys():
			cleaned_results = [x for x in parsed_data[method][cell_line] if (math.isnan(x) != True)]
			#print(line_label, len(cleaned_results))

			y.append(np.mean(cleaned_results))
			
		print(x, y, line_label, np.mean(y))
		plt.plot(x, y, label=line_label, linewidth=4.0, marker=parsed_data[method]['Marker'], markersize=15)


	plt.xlabel('Sparsity wrt High Read Count', size=14)
	plt.xticks(fontsize= 15)
	plt.ylabel(results_for, size=14)
	plt.yticks(fontsize= 15)
	#plt.ylim(0.7, 1)

	plt.legend(fontsize=12)

	plt.savefig('{}_ablation_results.png'.format(results_for))
	plt.cla()
	plt.clf()
	plt.close()
