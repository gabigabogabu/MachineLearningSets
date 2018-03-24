import csv
import numpy as np

FEATURETYPES = ['id', 'continuous', 'class', 'ignore', 'result_continuous', 'result_class']

class MachineLearningSet(object):
	def __init__(self, csvFileName, features, classMap={}, norm={}):
		"""csvFileName: File containing a learning set
		features: Dict in which the key is a collumn in the csv and the value is either:
			'ignore': collumn is ignored and not put into the learning sets
			'class': collumn is a class (such as male/female/other)
			'continuous': collumn is a continuous value (such as price)
			'result_class': collumn is used to calc cost of ml-model and is a class
			'result_continuous': collumn is used to calc cost of ml-model and is continuous
		classMap: Dict that specifies how classes are to be encoded:
			keys are the names of individual features
			values are lists that contain all possible values/classes for that feature
			the index for that value/class in that list represents that class
		norm is a dictionary that stores variance and mean of each feature:
			keys are the name of the individual features
			values are the mean and standart devation
		Note: If classMap or norm are not specified it will be calculated by this class.
		It is adivsed to specify classMap and norm if the csv to be converted is a continuation of a previous set
		- such as a test set is to a training or crossvalidation set."""
		self.csvFileName = csvFileName
		self.features = {f:features[f] for f in features if features[f] != 'ignore'}
		self.input_features = {f:features[f] for f in features if features[f] != 'ignore' and 'result' not in features[f]}
		self.result_features = {f:features[f] for f in features if features[f] != 'ignore' and 'result' in features[f]}
		self.classMap = classMap
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
						if feature not in self.classMap.keys():
							# add feature to classMap if not already seen
							self.classMap[feature] = []
						if row[feature] not in self.classMap[feature]:
							# add class to feature in classMap if not already seen
							self.classMap[feature].append(row[feature])
						c = list(self.input_features.keys()).index(feature) # index of current collumn
						self.input_set[r, c] = self.classMap[feature].index(row[feature])
					elif self.features[feature] == 'result_class':
						# map class to a number
						if feature not in self.classMap.keys():
							# add feature to classMap if not already seen
							self.classMap[feature] = []
						if row[feature] not in self.classMap[feature]:
							# add class to feature in classMap if not already seen
							self.classMap[feature].append(row[feature])
						c = list(self.result_features.keys()).index(feature) # index of current collumn
						self.result_set[r, c] = self.classMap[feature].index(row[feature])
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
				classes_count = len(self.classMap[feature])
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
				max_i = len(self.classMap[feature])
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
	print('\nclassMap\n', mls.classMap)
	print('\nnorm\n', mls.norm)


if __name__ == '__main__':
	main()