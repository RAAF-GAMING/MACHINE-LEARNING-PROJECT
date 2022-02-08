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

#creo un insieme di istanze formate da sole persone positive
data_positive= covid[covid["corona_result"]=="Positive"]

count_positive= data_positive["corona_result"].count()
print("Numero di positivi= ",count_positive)

#creo un insieme di istanze formate da sole persone negative e faccio l'undersampling
data_negative= covid[covid["corona_result"]=="Negative"].sample(count_positive,random_state=16)

print("\n\n")


#creo un dataset bilanciato

covid_bilanciato= data_negative.append(data_positive)
print("Ora il dataset e'")
print(covid_bilanciato.describe())
print("\n\n")
print("Dati dopo il Balancing")
print(covid_bilanciato["corona_result"].value_counts())

#rimostriamo il grafico a barre dopo il bilanciamento
pd.value_counts(covid_bilanciato['corona_result']).plot.bar()
plt.xlabel('corona_result')
plt.ylabel('Frequency')
plt.show()

#infine salviamo i dati in un nuovo file
print("\n\nSalvataggio dati in nuovo file CVS")
covid_bilanciato.to_csv(datapath + "covid_data_bilanciato.csv",index= False)