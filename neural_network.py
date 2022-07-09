import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
X = data_matrix.drop(columns=['Survived'])
y = data_matrix['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

scaler = StandardScaler()

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(5, activation='relu'),
                                    tf.keras.layers.Dense(5, activation='relu'),
                                    tf.keras.layers.Dense(5, activation='relu'),
                                    tf.keras.layers.Dense(5, activation='relu'),
                                    tf.keras.layers.Dense(1, activation='sigmoid')
                                    ])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=200, batch_size=50, verbose=1)

print(history.epoch, history.history['accuracy'][-1])

print(data_matrix.describe())
