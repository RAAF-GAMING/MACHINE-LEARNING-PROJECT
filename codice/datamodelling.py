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
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay



def splitTrainigAndTest(training,test,selection):
    #split data: dividiamo le variabili indipendenti da quella dipendente
    if selection==1:
        x_train=training.iloc[ : , 0:8]
        y_train=training.iloc[ : , 8]
        x_test=test.iloc[ : , 0:8]
        y_test=test.iloc[ : , 8]
    else:
        x_train=training.iloc[ : , 0:9]
        y_train=training.iloc[ : , 9]
        x_test=test.iloc[ : , 0:9]
        y_test=test.iloc[ : , 9]
    return x_train,y_train,x_test,y_test

def naiveBayes(x_train,y_train,x_test,y_test):
    #Inizializziamo NaiveBayes
    NB = CategoricalNB()
    #Addestriamo Naive Bayes
    print("Machine is learning...")
    NB.fit(x_train, y_train)
    #Facciamo fare le predizioni sul testset
    print("Machine is predicting...")
    predictions = NB.predict(x_test)
    #Stampiamo la matrice di confusione
    #TP-FP
    #FN-TN
    print("Matrice di confusione:")
    cm = confusion_matrix(y_test, predictions,labels=NB.classes_)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=NB.classes_)
    disp.plot()
    plt.show()
    #recall, precision...
    print(classification_report(y_test, predictions))
    #accuracy
    print("Accuracy:",metrics.accuracy_score(y_test, predictions))

def decisionTree(x_train,y_train,x_test,y_test):
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
    #TP-FP
    #FN-TN
    print("\n\n")
    print("Matrice di confusione:")
    cm = confusion_matrix(y_test, y_pred,labels=labels)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()
    #recall, precision...
    print(classification_report(y_test, y_pred))
    #accuracy
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
