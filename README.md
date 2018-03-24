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