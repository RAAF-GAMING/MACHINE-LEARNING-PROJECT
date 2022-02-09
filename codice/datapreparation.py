#(Data Preparation)Qui andiamo ad attuare la pipeline di data preparation necessarie per preparare il dataset
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
#as rename 
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

#load our data
datapath = os.path.join("dataset", "")

covid = pd.read_csv(datapath + "covid_data_2020-2021.csv")

#nice output if you want to see your data
#print(covid.head())
#split data: dividiamo le variabili indipendenti da quella dipendente
x= covid.iloc[ : , 0:6]
y= covid.iloc[ : , 6]
z= covid.iloc[ : , 7:10]
x = pd.concat([x,z.reindex(x.index)], axis=1)
print(x.head())
print("\n\n")
print(y.head())
print("\n\n")

#mostriamo la distribuzione delle istanze data la variabile target, rispetto la loro frequenza
pd.value_counts(covid['corona_result']).plot.bar()
plt.xlabel('corona_result')
plt.ylabel('Frequency')
print("Dati prima del Balancing")
print(covid['corona_result'].value_counts())
plt.show()
print("\n\n")
#splitto i dati in dati di train(x indipendenti, y dipendente) e i dati di test(x indipendenti, y dipendente)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
#restituisce il numero di righe
print("Number transactions X_train dataset: ", x_train.shape) 
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", x_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)
print("\n\n")
print("Prima UnderSampling, counts of label 'Positive': {}".format(sum(y_train == "Positive")))
print("Prima UnderSampling, counts of label 'Negative': {} ".format(sum(y_train == "Negative")))
print("\n\n")
#prepariamo l'undersampler sulla classe di magioranza e che generera le stesse istanze
un = RandomUnderSampler(sampling_strategy="majority",random_state=42)
#otteniamo il dataset bilanciato (solo training)
x_train_res, y_train_res = un.fit_resample(x_train, y_train)

print("Dopo UnderSampling, counts of label 'Positive': {}".format(sum(y_train_res == "Positive")))
print("Dopo UnderSampling, counts of label 'Negative': {}".format(sum(y_train_res == "Negative")))
print("\n\n")

#rimostriamo il grafico a barre dopo il bilanciamento
pd.value_counts(y_train_res).plot.bar()
plt.xlabel('corona_result')
plt.ylabel('Frequency')
plt.show()

#infine salviamo i dati in un nuovo file
#print("\n\nSalvataggio dati in nuovo file CVS")
#covid_bilanciato.to_csv(datapath + "covid_data_bilanciato.csv",index= False)