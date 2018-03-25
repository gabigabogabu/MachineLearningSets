#!/usr/bin/python3

"""Convert csv files into numpy arrays usable for machine learning applications"""

import csv
import numpy as np

__author__ = "github.com/gaschu95"
__license__ = "GPLv3"
__version__ = "0.1"

FEATURETYPES = ['continuous', 'class', 'ignore', 'result_continuous', 'result_class']

class MachineLearningSet(object):
	def __init__(self, csvFileName, features, classDict={}, norm={}):
		"""
		* csvFilename 
			* CSV file in which data is stored
		* features 
			* A dict with the features' name as key and the type of feature as value
			* The name has to be the same in the csv header.
			* possible types of features are:
				* 'continuous': this feature is on a continuous scale as opposed to classes. Will be represented in the input set
				* 'class': this feature has discrete values. Will be represented in the input set
				* 'result_continous': contiuous but a value that should be predicted by your ML algorithm. Will be represented in the result set
				* 'result_class': contiuous but a value that should be predicted by your ML algorithm. Will be represented in the result class
				* 'ignore': feature will be ignores and not represented in both the input and result set
		* classDict (optional)
			* A dict that specifies how classes are to be encoded 
				* The keys are the names of individual features as named in the csv and features-dict
				* The values are lists that contain all possible values/classes for that feature
				* The index for that value/class in the list of that feature represents that class
				* The numbers chosen to represent a class will be hot-encoded in the constructed sets
		* norm (optional) (i need to find a better name for this)
			* norm is a dictionary that stores standard deviation and mean of each feature:
				* The keys are the name of the individual features such as in the dicts above
				* The values are dicts with 'mean' and 'std' as keys that store the mean and standard deviation of that feature

		Note: If classDict or norm are not specified it will be calculated by this class.
		If you are converting your first/only csv file for your ML problem you can leave it empty and the will be computed automatically.
		However if you are converting a additional file for the same problem (such as a test set to a previous training set) you should give the 
		first conversion's classDict and norm to the new MachineLearningSet-instance in order make sure the class representations and scales remain the same. 
		If you don't do this it is very likely that you will get wonky results."""
		self.csvFileName = csvFileName
		self.features = {f:features[f] for f in features if features[f] != 'ignore'}
		self.input_features = {f:features[f] for f in features if features[f] != 'ignore' and 'result' not in features[f]}
		self.result_features = {f:features[f] for f in features if features[f] != 'ignore' and 'result' in features[f]}
		self.classDict = classDict
		self.norm = norm
		self.input_set = None
		self.result_set = None
		self._construct_matrices()
		self.input_vector_length = self.input_set.shape[1]
		self.result_vector_length = self.result_set.shape[1]

	def _construct_matrices(self):
		"""method that constructs the numpy arrays out of the data in csv file"""
		
		# get number of rows in the csv
		rowCount = self._get_csv_rowcount(self.csvFileName)

		# build np arrays
		self.input_set = np.ndarray((rowCount, len(self.input_features)), dtype=object)
		self.result_set = np.ndarray((rowCount, len(self.result_features)), dtype=object)

		# fill array
		with open(self.csvFileName) as csvFile:
			csvDict = csv.DictReader(csvFile)
			for r,row in enumerate(csvDict):
				for feature in self.features:
					if feature not in row:
						# skip over features that are not in the file
						# Note: does not ignore features that are in the file and not mentioned in the features dict
						continue
					elif self.features[feature] == 'continuous':
						# convert to float
						c = list(self.input_features.keys()).index(feature) # index of current collumn
						if row[feature] == '':
							# NaN if no value in that cell
							self.input_set[r, c] = np.nan
						else:
							self.input_set[r, c] = float(row[feature])
					elif self.features[feature] == 'result_continuous':
						c = list(self.result_features.keys()).index(feature) # index of current collumn
						# convert to float
						if row[feature] == '':
							# NaN if no value in that cell
							self.result_set[r, c] = np.nan
						else:
							self.result_set[r, c] = float(row[feature])
					elif self.features[feature] == 'class':
						# map class to a number
						if feature not in self.classDict.keys():
							# add feature to classDict if not already seen
							self.classDict[feature] = []
						if row[feature] not in self.classDict[feature]:
							# add class to feature in classDict if not already seen
							self.classDict[feature].append(row[feature])
						c = list(self.input_features.keys()).index(feature) # index of current collumn
						self.input_set[r, c] = self.classDict[feature].index(row[feature])
					elif self.features[feature] == 'result_class':
						# map class to a number
						if feature not in self.classDict.keys():
							# add feature to classDict if not already seen
							self.classDict[feature] = []
						if row[feature] not in self.classDict[feature]:
							# add class to feature in classDict if not already seen
							self.classDict[feature].append(row[feature])
						c = list(self.result_features.keys()).index(feature) # index of current collumn
						self.result_set[r, c] = self.classDict[feature].index(row[feature])
		self._normalize()
		self._encode_classes()

	def _encode_classes(self):
		"""Hot-Encode features that are classes"""
		c_offset = 0 # c_offset counts how many collumns have been added in total
		for feature in self.input_features:
			if self.input_features[feature] == 'class':
				# c is the index of the current collumn
				c = list(self.input_features.keys()).index(feature) + c_offset
				# max_i is how many classes this feature has
				classes_count = len(self.classDict[feature])
				# hot-encode the feature
				encoded = self._hot_encode(self.input_set[:,c].astype(int, copy=False), max_i=classes_count)
				# encoded.shape[1] is the number of collumns in the encoded array
				# we subtract 1 because the encoded array is going to replace the unencoded collumn
				c_offset += encoded.shape[1] - 1
				# replace unencoded collumn with encoded collumns
				self.input_set = np.hstack([ self.input_set[:,:c], encoded, self.input_set[:,(c+1):] ])

		# encode result classes, same as above but with self.result_set and self.result_features
		c_offset = 0
		for feature in self.result_features:
			if self.result_features[feature] == 'result_class':
				c = list(self.result_features.keys()).index(feature) + c_offset
				max_i = len(self.classDict[feature])
				encoded = self._hot_encode(self.result_set[:,c].astype(int, copy=False), max_i=max_i)
				c_offset += encoded.shape[1] - 1
				self.result_set = np.hstack([ self.result_set[:,:c], encoded, self.result_set[:,(c+1):] ])

	def _normalize(self):
		"""Normalize features that are continuous such that the normalized features have a mean of 0 and a standard of 1"""
		for feature in self.features:
			# normalize input featrues
			if feature in self.input_features and self.input_features[feature] == 'continuous':
				# c is the index of the current collumn
				c = list(self.input_features.keys()).index(feature)
				# add feature to norm dict if not already present
				if feature not in self.norm.keys():
					self.norm[feature] = {
							'mean': np.nanmean(self.input_set[:,c].astype(float, copy=False)),
							'std': np.nanstd(self.input_set[:,c].astype(float, copy=False))
					}
				# normalize feature by subtracting mean and dividing by standard deviation
				self.input_set[:,c] = (self.input_set[:,c] - self.norm[feature]['mean']) / self.norm[feature]['std']

			# normalize result features. same as above but with self.result_features and self.result_set
			elif feature in self.result_features and self.result_features[feature] == 'result_continuous':
				c = list(self.result_features.keys()).index(feature)
				if feature not in self.norm.keys():
					self.norm[feature] = {
							'mean': np.nanmean(self.input_set[:,c].astype(float, copy=False)),
							'std': np.nanstd(self.input_set[:,c].astype(float, copy=False))
					}
				self.result_set[:,c] = (self.result_set[:,c] - self.norm[feature]['mean']) / self.norm[feature]['std']

	@staticmethod
	def _get_csv_rowcount(csvFileName):
		"""returns the rows in a csv file"""
		rowCount = 0
		with open(csvFileName) as csvFile:
			rowCount = sum(1 for row in csv.DictReader(csvFile))
		return rowCount

	@staticmethod
	def _hot_encode(i, max_i=None):
		"""hot encodes a vector. If i is a N-d vector a N x max(N) matrix is returned"""
		if len(i.shape) != 1: 
			raise ValueError('i is not a vector: i.shape = ' + str(i.shape))
		if max_i == None: max_i = max(i)+1
		v = np.zeros((len(i), max_i))
		v[range(len(i)), i] = 1
		# 2 classes can be represented by only one value (class and !class)
		# so only the first collumn is used
		# it has to be reshaped into an collumn vector to keep the "1 row = 1 example"-format
		if max_i <= 2:
			v = v[:,0].reshape(v.shape[0], 1)
		return v


def main():
	features = {'PassengerId': 'ignore', 
			'Survived': 'result_class', 
			'Pclass': 'class', 
			'Name': 'ignore', 
			'Sex': 'class', 
			'Age': 'continuous',
			'SibSp': 'continuous',
			'Parch': 'continuous',
			'Ticket': 'ignore',
			'Fare': 'continuous',
			'Cabin': 'ignore',
			'Embarked': 'class',
			}
	mls = MachineLearningSet('train.csv', features)
	print('\ninput set\n', mls.input_set)
	# print('\nresult set\n', mls.result_set)
	print('\nfeatures\n', mls.features)
	print('\ninput features\n', mls.input_features)
	print('\nresult features\n', mls.result_features)
	print('\nclassDict\n', mls.classDict)
	print('\nnorm\n', mls.norm)


if __name__ == '__main__':
	main()