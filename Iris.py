# Project Iris

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('iris.csv')
dataset.head()

# Plot Variables
dataset.Species.value_counts().plot(kind="pie", autopct='%.2f%%')
sns.catplot(x="Species", y='sepal length in cm', kind="violin", data=dataset)
sns.swarmplot(x="Species", y='sepal length in cm', color="k", size=3, data=dataset)
#sns.scatterplot(x = 'sepal length in cm', y = 'sepal width in cm', data = dataset, hue = 'Species')
#sns.scatterplot(x = 'petal length in cm', y = 'petal width in cm', data = dataset, hue = 'Species')
#sns.pairplot(dataset.drop("ID", axis=1), hue="Species", size=2) 

# Classify dependent and independent variables
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Correlations
#corr = dataset.corr()

# Label Encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
y = LE.fit_transform(y)

# Splitting the dataset into the Trainin set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform (X_test)

# Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

# Feed the training data to the classifier
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Accuracy Score
from sklearn.metrics import accuracy_score
print("Accuracy:  {:.2f} %".format(accuracy_score(y_test,y_pred)*100))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

# Classification report
from sklearn.metrics import classification_report
target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
print(classification_report(y_test,y_pred,target_names=target_names))
