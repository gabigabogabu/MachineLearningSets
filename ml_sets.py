import numpy as np
import pandas as pd

def shuffle_set(s):
	"""shuffles a set"""
	return s.sample(frac=1)

def fill_NaN_with_random(all_sets):
	"""replace NaNs in a dataset with random values from the same column"""
	nan_sum = pd.concat([s.isnull().sum() for s in all_sets], axis=1).sum(axis=1)
	columns = (nan_sum[nan_sum > 0] / pd.concat([s for s in all_sets]).shape[0]).index.values
	print('fill nan with random', columns)
	for i in range(len(all_sets)):
		all_sets[i] = all_sets[i].apply(lambda x: x.fillna(np.random.choice(x.dropna())))
	return all_sets

def add_is_NaN(all_sets, columns=None):
	"""adds a feature that describes whether a feature has a NaN at that position
	columns: columns to be handled. Defaults to None
	if columns is None all columns that have at least one NaN value will be handled"""
	if columns == None:
		nan_sum = pd.concat([s.isnull().sum() for s in all_sets], axis=1).sum(axis=1)
		columns = (nan_sum[nan_sum > 0] / pd.concat([s for s in all_sets]).shape[0]).index.values
	print('add is nan', columns)
	for i in range(len(all_sets)):
		for c in columns:
			all_sets[i][str(c)+'_nan'] = all_sets[i][c].isnull().astype(int)
	return all_sets

def add_string_contains(all_sets, string, columns=None):
	"""UNTESTED"""
	# UNTESTED
	if columns == None:
		dtypes = pd.concat([s for s in full_set], axis=0).dtypes
		columns = dtypes[(dtypes != 'float') & (dtypes != 'int')].index.values
	print('add contains:', columns)
	for i in range(len(all_sets)):
		for c in columns:
			all_sets[i][str(c) +'_has_'+ string] = all_sets[i][c].apply(lambda x: string in x)

	return all_sets

def add_string_length(all_sets, columns=None):
	"""adds a feature that describes the length of a string in a feature
	columns: columns to be handled. Defaults to None
	if columns is None all columns that are not numbers will be handled"""
	if columns == None:
		dtypes = pd.concat([s for s in all_sets], axis=0).dtypes
		columns = dtypes[(dtypes != 'float') & (dtypes != 'int')].index.values
	print('add string length', columns)
	for i in range(len(all_sets)):
		for c in columns:
			all_sets[i][str(c) +'_#letters'] = all_sets[i][c].apply(lambda x: len(str(x)) if x is not np.nan else 0)
	return all_sets

def add_word_count(all_sets, columns=None):
	"""adds a feature that describes the number of words in a string in a feature
	columns: columns to be handled. Defaults to None
	if columns is None all columns that are not numbers will be handled"""
	if columns == None:
		dtypes = pd.concat([s for s in all_sets], axis=0).dtypes
		columns = dtypes[(dtypes != 'float') & (dtypes != 'int')].index.values
	print('add word count', columns)
	for i in range(len(all_sets)):
		for c in columns:
			all_sets[i][str(c) +'_#words'] = all_sets[i][c].apply(lambda x: len(str(x).split()) if x is not np.nan else 0)
	return all_sets

def hot_encode_classes(all_sets, columns=None, unique_frac=.01):
	"""adds a feature for each unique value of a column that describes if that value is in that feature
	columns: columns to be handled. Defaults to None. If None columns to be handled will be that meet unique_frac criteria
	unique_frac: the relative number of unique values to length of a column to be hot encoded. Defaults to .01. Only used when columns is None"""
	if columns == None:
		unique_count = pd.concat([s for s in all_sets], axis=0).nunique() / pd.concat([s for s in all_sets]).shape[0]
		columns = unique_count[unique_count < unique_frac].index.values
	print('hot encoding:', columns)
	for c in columns:
		# enumerate
		str_dict = {k:v for v,k in dict(enumerate(pd.unique(pd.concat([s for s in all_sets], axis=0)[c].values))).items()}
		# hot encode enumeration
		str_dict_h = {k:np.zeros(len(str_dict), dtype=int) for k,v in str_dict.items()}
		for k,v in str_dict_h.items():
			v[str_dict[k]] = 1
		# insert into dataframe
		for i in range(len(all_sets)):
			if c not in all_sets[i].columns:
				continue
			# convert nested arrays into multiple top level columns
			for j,subfeature in enumerate(str_dict_h):
				all_sets[i][str(c)+'_'+str(subfeature)] = all_sets[i][c].apply(lambda x: str_dict_h[x][j])
	return all_sets

def delete_non_numbers(all_sets):
	"""deletes columns with non numbers"""
	dtypes = pd.concat([s for s in all_sets], axis=0).dtypes
	columns = dtypes[(dtypes != 'float') & (dtypes != 'int')].index.values
	# print(columns)
	for i in range(len(all_sets)):
		all_sets[i] = all_sets[i].drop(columns=columns)
	return all_sets

def delete_single_valued_columns(all_sets):
	"""deletes columns that only contain a single unique value"""
	std = pd.concat([s for s in all_sets], axis=0).std()
	columns = std[std == 0].index.values
	print('deleting redundant columns: ', columns)
	for i in range(len(all_sets)):
		all_sets[i] = all_sets[i].drop(columns=columns, errors='ignore')
	return all_sets

def normalize(all_sets):
	"""normalizes (subtract mean, then divide by standart deviation)"""
	mean = pd.concat([s for s in all_sets], axis=0).mean(axis=0)
	std = pd.concat([s for s in all_sets], axis=0).std(axis=0)
	print('normalizing')
	for i in range(len(all_sets)):
		all_sets[i] = (all_sets[i] - mean) / std
	return all_sets, mean, std

def denormalize(all_sets, mean, std):
	"""reverses normalization"""
	for i in range(len(all_sets)):
		all_sets[i] = (all_sets[i] * std) + mean
	return all_sets
