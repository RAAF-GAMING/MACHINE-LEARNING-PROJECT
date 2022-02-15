import os
import pandas as pd
from datamodelling import decisionTree, splitTrainigAndTest,naiveBayes
from dataexploration import exploration
from datapreparation import *


print("Inizio dell'analisi dei dati...")
#load our data
datapath = os.path.join("dataset", "")

covid = pd.read_csv(datapath + "covid_data_2020-2021.csv")
#analizziamo i dati
exploration(covid)

#splittiamo i dati in test set e training set
x,y=splitData(covid)

#mostriamo la distrubuzione dei dati per la variabile dipendente
showDistribution(covid)
balance=int(input("Vuoi fare data balancing? (1->si,0->no)"))

if balance==1:
    #bilanciamo il dataset
    x_train_res,y_train_res,x_test,y_test=balancing(x,y)
else:
    x_train_res, x_test, y_train_res, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#normalizziamo i dati
x_train_res,y_train_res,x_test,y_test=featureScaling(x_train_res, y_train_res,x_test, y_test)
selection=int(input("Vuoi fare feature selection? (1->si,0->no)"))
if selection==1:
    #effettuiamo la feature selection
    x_train_res,x_test=featureSelection(x_train_res,y_train_res,x_test,y_test,x)

#salviamo i dati di training e di test
training=pd.concat([x_train_res, y_train_res.reindex(x_train_res.index)], axis=1)
test=pd.concat([x_test, y_test.reindex(x_test.index)], axis=1)
datapath = os.path.join("dataset", "")
training.to_csv(datapath + "covid_data_training.csv",index= False)
test.to_csv(datapath + "covid_data_test.csv",index= False)

#splittiamo il dataset
x_train,y_train,x_test,y_test=splitTrainigAndTest(training,test,selection)

#scegliamo il modello da utilizzare
modello=int(input("Che classificatore vuoi utilizzare (1->Naive Bayes,0->Decision Tree)"))
if modello==1:
    naiveBayes(x_train,y_train,x_test,y_test)
else:
    decisionTree(x_train,y_train,x_test,y_test)




   







