"""
Here i am going to split the whole dataset into train and test set for training and after that testing the model.

                   |---->   train_set.csv
raw_dataset.csv ---|
                   |---->   test_set.csv
"""

# Import required packages
import numpy as np
import pandas as pd

# Load dataset
x = pd.read_csv('A_Z Handwritten Data.csv', header=None)

print(x.head(3))

# use list y as a dummy variable in the function train_test_split to split x into train and test set
y = list(map(lambda i: i, range(len(x))))
print(f" Before conversion Number of row in y: {len(y)} and its type is {type(y)}")

# Convert list to numpy array
y = np.asarray(y)
print(f" After conversion Number of row in y: {len(y)} and its type is {type(y)}")

print(f"number of row in x : {len(x)} and its type is {type(x)}")

# Import train_test_split method from sklearn
from sklearn.model_selection import train_test_split

# Split row of dataset in train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# save this data frame as CSV and discard index
x_train.to_csv('train.csv', index=False)

print('-' * 60)

# save this data frame as CSV and discard index
x_test.to_csv('test.csv', index=False)
