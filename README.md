# MachineLearningSets
Makes csv Files numpy accessible. Useful for ML applications.

# Usage

Example:

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
	print('\nfeatures\n', mls.features)
	print('\ninput features\n', mls.input_features)
	print('\nresult features\n', mls.result_features)
	print('\nclassMap\n', mls.classMap)
	print('\nnorm\n', mls.norm)

Will return:

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