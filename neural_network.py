import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Read in Data
def read_in_data():
    d = pd.read_csv('data_titanic/train.csv')
    pd.set_option('display.max_columns', None)
    return d


# Clean Data
def clean_data(d):
    d.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], inplace=True)
    d['Sex'].replace(['female', 'male'], [0, 1], inplace=True)
    d.dropna(inplace=True)
    return d


data_matrix = read_in_data()
cleaned_data = clean_data(data_matrix)

print(data_matrix.describe())
