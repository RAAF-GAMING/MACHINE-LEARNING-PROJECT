#(Data Modelling)Qui andiamo ad implementare il nostro machine learner
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
#as rename 
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

#load our data
datapath = os.path.join("dataset", "")

training=pd.read_csv(datapath + "covid_data_training.csv")
test=pd.read_csv(datapath + "covid_data_test.csv")
#nice output if you want to see your data
#print(covid.head())
#split data: dividiamo le variabili indipendenti da quella dipendente
x_train=training.iloc[ : , 0:9]
y_train=training.iloc[ : , 9]
x_test=test.iloc[ : , 0:9]
y_test=test.iloc[ : , 9]
#Settiamo l'algoritmo di decision tree
tree_model = DecisionTreeClassifier(random_state=42)
#Alleniamo il nostro modello
print("Machine is learning...")
tree_model.fit(x_train, y_train)
#Machine is predicting
print("Machine is predicting...")
y_pred = tree_model.predict(x_test)
labels = np.unique(y_test)
#Stampiamo la matrice di confusione
print("\n\n")
print(confusion_matrix(y_test, y_pred, labels=labels))
print(classification_report(y_test, y_pred))

#Overall, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))