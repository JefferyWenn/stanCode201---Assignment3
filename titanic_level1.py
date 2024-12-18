"""
File: titanic_level1.py
Name: 
----------------------------------
This file builds a machine learning algorithm from scratch 
by Python. We'll be using 'with open' to read in dataset,
store data into a Python dict, and finally train the model and 
test it on kaggle website. This model is the most flexible among all
levels. You should do hyper-parameter tuning to find the best model.
"""

import math
import numpy as np
TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename: str, data: dict, mode='Train', training_data=None):
    """
    :param filename: str, the filename to be processed
    :param data: an empty Python dictionary
    :param mode: str, indicating if it is training mode or testing mode
    :param training_data: dict[str: list], key is the column name, value is its data
                          (You will only use this when mode == 'Test')
    :return data: dict[str: list], key is the column name, value is its data
    """
    with open(filename, 'r') as f:
        first = True
        for line in f:
            if first:
                first = False
                headers = line.strip().split(',')
                # Initialize data dictionary with empty lists for each header
                if mode == 'Train':
                    data = {header: [] for header in headers if header not in ['PassengerId', 'Name', 'Ticket', 'Cabin']}
                else:
                    data = {header: [] for header in headers if header not in ['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin']}
            else:
                row = line.strip().split(',')
                row_dict = {}
                # Map each value to its corresponding header
                for i in range(len(headers) + 1):
                    if i < 3:
                        row_dict[headers[i]] = row[i]
                    elif i > 3:
                        row_dict[headers[i - 1]] = row[i]

                # Check for missing values and convert types
                if mode == 'Train':
                    if row_dict["Age"] != '' and row_dict["Embarked"] != '':
                        row_dict["Age"] = float(row_dict["Age"])
                        row_dict["Embarked"] = 0 if row_dict["Embarked"] == 'S' else 1 if row_dict["Embarked"] == 'C' else 2
                        row_dict["Sex"] = 1 if row_dict['Sex'] == "male" else 0

                        row_dict["Survived"] = float(row_dict["Survived"])
                        row_dict["Pclass"] = float(row_dict["Pclass"])
                        row_dict["SibSp"] = float(row_dict["SibSp"])
                        row_dict["Parch"] = float(row_dict["Parch"])
                        row_dict["Fare"] = float(row_dict["Fare"])

                        for header in headers:
                            if header not in ['PassengerId', 'Name', 'Ticket', 'Cabin']:
                                data[header].append(row_dict[header])
                else:
                    row_dict["Age"] = round(np.mean(training_data['Age']), 3) if row_dict["Age"] == '' else float(row_dict["Age"])
                    row_dict["Fare"] = round(np.mean(training_data['Fare']), 3) if row_dict["Fare"] == '' else float(row_dict["Fare"])
                    row_dict["Embarked"] = round(np.mean(training_data["Embarked"])) if row_dict["Embarked"] == '' else \
                    (0 if row_dict["Embarked"] == 'S' else 1 if row_dict["Embarked"] == 'C' else 2)
                    row_dict["Sex"] = 1 if row_dict['Sex'] == "male" else 0
                    row_dict["Pclass"] = float(row_dict["Pclass"])
                    row_dict["SibSp"] = float(row_dict["SibSp"])
                    row_dict["Parch"] = float(row_dict["Parch"])
                    # Add processed row to data dictionary
                    for header in headers:
                        if header not in ['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin']:
                            data[header].append(row_dict[header])

    return data


def one_hot_encoding(data: dict, feature: str):
    """
    :param data: dict[str, list], key is the column name, value is its data
    :param feature: str, the column name of interest
    :return data: dict[str, list], remove the feature column and add its one-hot encoding features
    """
    # Get unique values for the feature
    unique_values = []
    for value in data[feature]:
        if value not in unique_values:
            unique_values.append(value)

    # Initialize new columns for one-hot encoding
    new_data = {key: data[key][:] for key in data.keys() if key != feature}
    for value in unique_values:
        new_col_name = f"{feature}_{int(value)-1}" if feature == 'Pclass' else f"{feature}_{value}"
        new_data[new_col_name] = [0] * len(data[feature])

    # Fill in the one-hot encoded columns
    for i in range(len(data[feature])):
        value = data[feature][i]
        new_col_name = f"{feature}_{int(value)-1}" if feature == 'Pclass' else f"{feature}_{value}"
        new_data[new_col_name][i] = 1

    return new_data


def normalize(data: dict):
    """
    :param data: dict[str, list], key is the column name, value is its data
	:return data: dict[str, list], key is the column name, value is its normalized data
    """
    normalized_data = {}
    for key, values in data.items():
        min_val = min(values)
        max_val = max(values)
        normalized_values = [(value - min_val) / (max_val - min_val) for value in values]
        normalized_data[key] = normalized_values
    return normalized_data


def learnPredictor(inputs: dict, labels: list, degree: int, num_epochs: int, alpha: float):
    """
    :param inputs: dict[str, list], key is the column name, value is its data
    :param labels: list[int], indicating the true label for each data
    :param degree: int, degree of polynomial features
    :param num_epochs: int, the number of epochs for training
    :param alpha: float, step size or learning rate
    :return weights: dict[str, float], feature name and its weight
    """
    from util import increment
    from util import dotProduct

    def sigmoid(k):
        """
        :param k: float, linear function value
        :return: float, probability of the linear function value
        """
        return 1 / (1 + math.exp(-k))

    # Initialize weights
    weights = {}
    keys = list(inputs.keys())
    if degree == 1:
        for i in range(len(keys)):
            weights[keys[i]] = 0
    elif degree == 2:
        for i in range(len(keys)):
            weights[keys[i]] = 0
        for i in range(len(keys)):
            for j in range(i, len(keys)):
                weights[keys[i] + keys[j]] = 0

    # Feature Extraction
    def feature_extractor(instance, degree):
        x = {}
        if degree == 1:
            for key in keys:
                x[key] = instance[key]
        elif degree == 2:
            for key in keys:
                x[key] = instance[key]
            for i in range(len(keys)):
                for j in range(i, len(keys)):
                    x[keys[i] + keys[j]] = instance[keys[i]] * instance[keys[j]]
        return x

    # Training loop
    for epoch in range(num_epochs):
        for i in range(len(labels)):
            instance = {key: inputs[key][i] for key in keys}
            y = labels[i]
            x = feature_extractor(instance, degree)
            # Compute prediction
            prediction = sigmoid(dotProduct(weights, x))
            # Compute error
            error = prediction - y
            # Update weights
            increment(weights, -alpha * error, x)

    return weights
