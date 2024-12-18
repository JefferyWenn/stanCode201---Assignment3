"""
File: titanic_level2.py
Name: 
----------------------------------
This file builds a machine learning algorithm by pandas and sklearn libraries.
We'll be using pandas to read in dataset, store data into a DataFrame,
standardize the data by sklearn, and finally train the model and
test it on kaggle website. Hyper-parameters tuning are not required due to its
high level of abstraction, which makes it easier to use but less flexible.
You should find a good model that surpasses 77% test accuracy on kaggle.
"""

import math
import pandas as pd
from sklearn import preprocessing, linear_model

TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be read into pandas
	:param mode: str, indicating the mode we are using (either Train or Test)
	:param training_data: DataFrame, a 2D data structure that looks like an excel worksheet
						  (You will only use this when mode == 'Test')
	:return: Tuple(data, labels), if the mode is 'Train'; or return data, if the mode is 'Test'
	"""
	data = pd.read_csv(filename)
	labels = None
	if mode == 'Train':
		data = data.dropna(subset=['Age', 'Embarked'])
		labels = data['Survived']
		data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

	elif mode == 'Test':
		data['Age'].fillna(training_data['Age'].mean(), inplace=True)
		data['Fare'].fillna(training_data['Fare'].mean().round(3), inplace=True)
		data['Embarked'].fillna(training_data['Embarked'].mode()[0], inplace=True)
		data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]


	# Changing 'male' to 1, 'female' to 0
	data.loc[data.Sex == 'male', 'Sex'] = 1
	data.loc[data.Sex == 'female', 'Sex'] = 0
	# Changing 'S' to 0, 'C' to 1, 'Q' to 2
	data['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)
	if mode == 'Train':
		return data, labels
	elif mode == 'Test':
		return data


def one_hot_encoding(data, feature):
	"""
	:param data: DataFrame, key is the column name, value is its data
    :param feature: str, the column name of interest
    :return data: DataFrame, remove the feature column and add its one-hot encoding features
	"""
	if feature == 'Sex':
		data['Sex_0'] = 0  # Female
		data.loc[data.Sex == 0, 'Sex_0'] = 1
		data['Sex_1'] = 0  # Male
		data.loc[data.Sex == 1, 'Sex_1'] = 1
		data.pop('Sex')
	elif feature == 'Pclass':
		data['Pclass_0'] = 0  # FirstClass
		data.loc[data.Pclass == 1, 'Pclass_0'] = 1
		data['Pclass_1'] = 0  # SecondClass
		data.loc[data.Pclass == 2, 'Pclass_1'] = 1
		data['Pclass_2'] = 0  # ThirdClass
		data.loc[data.Pclass == 3, 'Pclass_2'] = 1
		data.pop('Pclass')

	return data


def standardization(data, mode='Train'):
	"""
    :param data: DataFrame, key is the column name, value is its data
	:param mode: str, indicating the mode we are using (either Train or Test)
    :return data: DataFrame, standardized features
	"""
	features_to_standardize = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
	standardizer = preprocessing.StandardScaler()
	if mode == 'Train':
		data[features_to_standardize] = standardizer.fit_transform(data[features_to_standardize])
	elif mode == 'Test':
		data[features_to_standardize] = standardizer.transform(data[features_to_standardize])

	return data[features_to_standardize].values


def main():
	"""
	You should call data_preprocess(), one_hot_encoding(), and
	standardization() on your training data. You should see ~80% accuracy on degree1;
	~83% on degree2; ~87% on degree3.
	Please write down the accuracy for degree1, 2, and 3 respectively below
	(rounding accuracies to 8 decimal places)
	TODO: real accuracy on degree1 -> 0.7949438202247191
	TODO: real accuracy on degree2 -> 0.824438202247191
	TODO: real accuracy on degree3 -> 0.827247191011236
	"""
	# Data cleaning
	train_data, labels = data_preprocess(TRAIN_FILE, mode='Train')
	test_data = data_preprocess(TEST_FILE, mode='Test', training_data=train_data)

	# Normalization / Standardization
	normalizer = preprocessing.MinMaxScaler()
	X_train = normalizer.fit_transform(train_data)

	#############################
	# Degree 1 Polynomial Model #
	#############################
	h = linear_model.LogisticRegression()
	classifier = h.fit(X_train, labels)
	training_acc = classifier.score(X_train, labels)
	print(f"Degree 1 training accuracy: {training_acc}.")

	#############################
	# Degree 2 Polynomial Model #
	#############################

	poly_phi_extractor = preprocessing.PolynomialFeatures(degree=2)
	X_train_poly = poly_phi_extractor.fit_transform(X_train)
	classifier_poly = h.fit(X_train_poly, labels)
	training_acc = classifier_poly.score(X_train_poly, labels)
	print(f"Degree 2 training accuracy: {training_acc}.")

	#############################
	# Degree 3 Polynomial Model #
	#############################

	poly_phi_extractor = preprocessing.PolynomialFeatures(degree=3)
	X_train_poly2 = poly_phi_extractor.fit_transform(X_train)
	classifier_poly2 = h.fit(X_train_poly2, labels)
	training_acc = classifier_poly2.score(X_train_poly2, labels)
	print(f"Degree 3 training accuracy: {training_acc}.")


if __name__ == '__main__':
	main()
