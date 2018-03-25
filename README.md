# MachineLearningSets
Prepares data from csv files for machine learning applications

# Construction Parameters

* `csvFilename`
	* CSV file in which data is stored
* `features` 
	* A dict with the features' name as key and the type of feature as value
	* The name has to be the same in the csv header.
	* possible types of features are:
		* `'continuous'`: this feature is on a continuous scale as opposed to classes. Will be represented in the input set
		* `'class'`: this feature has discrete values. Will be represented in the input set
		* `'result_continous'`: contiuous but a value that should be predicted by your ML algorithm. Will be represented in the result set
		* `'result_class'`: contiuous but a value that should be predicted by your ML algorithm. Will be represented in the result class
		* `'ignore'`: feature will be ignores and not represented in both the input and result set
* `classDict` (optional)
	* A dict that specifies how classes are to be encoded 
		* The keys are the names of individual features as named in the csv and features-dict
		* The values are lists that contain all possible values/classes for that feature
		* The index for that value/class in the list of that feature represents that class
		* The numbers chosen to represent a class will be hot-encoded in the constructed sets
* `norm` (optional) (i need to find a better name for this)
	* A dictionary that stores standard deviation and mean of each feature:
		* The keys are the name of the individual features such as in the dicts above
		* The values are dicts with `'mean'` and `'std'` as keys that store the mean and standard deviation of that feature

Note: If classDict or norm are not specified it will be calculated by this class.
If you are converting your first/only csv file for your ML problem you can leave it empty and the will be computed automatically.
However if you are converting a additional file for the same problem (such as a test set to a previous training set) you should give the first conversion's classDict and norm to the new MachineLearningSet-instance in order make sure the class representations and scales remain the same. If you don't do this it is very likely that you will get wonky results.

# Useful class members

* All the parameters given on construction will be saved
* `input_features`: A dict with the same structure as the features parameter but filtered for features that serve as input
* `result_features`: Similar to input_features but filtered for features that will be output from your ML algorithm
* `classDict`: Same as parameter but updated if new classes/features found
* `norm`: Same as parameter but updated if new features found
* `input_set`: The fully encoded set ready to be fed into your ML algorithm
* `result_set`: The fully encoded set ready to be compared to your ML alg's output
* `input_vector_length` and `result_vector_length` useful to set the size of your input and output layer of ML alg

# Example

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
	print('\nfeatures\n', mls.features)
	print('\ninput features\n', mls.input_features)
	print('\nresult features\n', mls.result_features)
	print('\nclassMap\n', mls.classMap)
	print('\nnorm\n', mls.norm)

With `train.csv` being:

	PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
	1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
	2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C
	3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
	4,1,1,"Futrelle, Mrs. Jacques Heath (Lily May Peel)",female,35,1,0,113803,53.1,C123,S
	5,0,3,"Allen, Mr. William Henry",male,35,0,0,373450,8.05,,S
	6,0,3,"Moran, Mr. James",male,,0,0,330877,8.4583,,Q
	...

Will return:

	input set
	[[1.0 0.0 0.0 ... 0.0 0.0 0.0]
	[0.0 1.0 0.0 ... 1.0 0.0 0.0]
	[1.0 0.0 0.0 ... 0.0 0.0 0.0]
	...
	[1.0 0.0 0.0 ... 0.0 0.0 0.0]
	[0.0 1.0 0.0 ... 1.0 0.0 0.0]
	[1.0 0.0 0.0 ... 0.0 1.0 0.0]]

	features
	{'Survived': 'result_class', 'Pclass': 'class', 'Sex': 'class', 'Age': 'continuous', 'SibSp': 'continuous', 'Parch':'continuous', 'Fare': 'continuous', 'Embarked': 'class'}

	input features
	{'Pclass': 'class', 'Sex': 'class', 'Age': 'continuous', 'SibSp': 'continuous', 'Parch': 'continuous', 'Fare': 'continuous', 'Embarked': 'class'}

	result features
	{'Survived': 'result_class'}

	classMap
	{'Survived': ['0', '1'], 'Pclass': ['3', '1', '2'], 'Sex': ['male', 'female'], 'Embarked': ['S', 'C', 'Q', '']}

	norm
	{'Age': {'mean': 29.69911764705882, 'std': 14.516321150817316}, 'SibSp': {'mean': 0.5230078563411896, 'std': 1.1021244350892878}, 'Parch': {'mean': 0.38159371492704824, 'std': 0.8056047612452208}, 'Fare': {'mean': 32.204207968574636, 'std': 49.6655344447741}}