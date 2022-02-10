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
print("Prima dell'UnderSampling, counts of label 'Positive': {}".format(sum(y_train == "Positive")))
print("Prima dell'UnderSampling, counts of label 'Negative': {} ".format(sum(y_train == "Negative")))
print("\n\n")
#prepariamo l'undersampler sulla classe di magioranza e che generera le stesse istanze
un = RandomUnderSampler(sampling_strategy="majority",random_state=42)
#otteniamo il dataset bilanciato (solo training)
x_train_res, y_train_res = un.fit_resample(x_train, y_train)

print("Dopo l'UnderSampling, counts of label 'Positive': {}".format(sum(y_train_res == "Positive")))
print("Dopo l'UnderSampling, counts of label 'Negative': {}".format(sum(y_train_res == "Negative")))
print("\n\n")

#rimostriamo il grafico a barre dopo il bilanciamento
pd.value_counts(y_train_res).plot.bar()
plt.xlabel('corona_result')
plt.ylabel('Frequency')
plt.show()

#infine salviamo i dati in un nuovo file
#print("\n\nSalvataggio dati in nuovo file CVS")
#covid_bilanciato.to_csv(datapath + "covid_data_bilanciato.csv",index= False)

#FEATURE SCALING
#andiamo a normalizzare la distribuzione delle seguenti feature: test_date, test_indication, gender, age_60_and_above
print("Inizio Feature Scaling")
#normalizziamo gender in questo modo: male->1 , female->0
print("Normalizzo gender...")
x_train_res["gender"]= x_train_res["gender"].replace({"male":1})
x_train_res["gender"]= x_train_res["gender"].replace({"female":0})

#lo applichiamo anche sul test set
x_test["gender"]= x_test["gender"].replace({"male":1})
x_test["gender"]= x_test["gender"].replace({"female":0})

#normalizziamo age_60_and_above in questo modo: yes->1 , no->0
print("Normalizzo age_60_above...")
x_train_res["age_60_and_above"]= x_train_res["age_60_and_above"].replace({"Yes":1})
x_train_res["age_60_and_above"]= x_train_res["age_60_and_above"].replace({"No":0})

x_test["age_60_and_above"]= x_test["age_60_and_above"].replace({"Yes":1})
x_test["age_60_and_above"]= x_test["age_60_and_above"].replace({"No":0})

#normalizziamo test_indication andando ad unire i seguenti 2 valori in un unico valore: Other, Abroad
#la normalizzazione verrà fatta nel seguente modo: other,abroad->0 , Contact with confirmed->1
print("Normalizzo test_indication...")
x_train_res["test_indication"]= x_train_res["test_indication"].replace({"Other":0})
x_train_res["test_indication"]= x_train_res["test_indication"].replace({"Abroad":0})
x_train_res["test_indication"]= x_train_res["test_indication"].replace({"Contact with confirmed":1})

x_test["test_indication"]= x_test["test_indication"].replace({"Other":0})
x_test["test_indication"]= x_test["test_indication"].replace({"Abroad":0})
x_test["test_indication"]= x_test["test_indication"].replace({"Contact with confirmed":1})


#normalizziamo test_date in base alla curva dei contagi
#consideriamo i mesi in cui la curva dei contagi e' alta con valore 1 ovvero: gennaio-maggio e da ottobre-dicembre
#consideriamo i mesi in cui la curva dei contagi e' bassa con valore 0 ovvero: giugno-settembre
print("Normalizzo test_date...")
x_train_res= x_train_res.reset_index()
y_train_res= y_train_res.reset_index()
i=0
#normalizziamo le date del training set
for row in x_train_res.itertuples():
    if row.test_date>="2020-01-01" and row.test_date<="2020-05-31":
        x_train_res.at[i,"test_date"]= 1
    elif row.test_date>="2020-10-01" and row.test_date<="2020-12-31":
        x_train_res.at[i,"test_date"]= 1
    elif row.test_date>="2021-01-01" and row.test_date<="2021-05-31":
        x_train_res.at[i,"test_date"]= 1
    elif row.test_date>="2021-10-01" and row.test_date<="2021-12-31":
        x_train_res.at[i,"test_date"]= 1
    elif row.test_date>="2020-06-01" and row.test_date<="2020-09-30":
        x_train_res.at[i,"test_date"]= 0
    elif row.test_date>="2021-06-01" and row.test_date<="2021-09-30":
        x_train_res.at[i,"test_date"]= 0
    i= i+1

x_test= x_test.reset_index()
y_test= y_test.reset_index()
i=0
#normalizziamo le date del test set
for row in x_test.itertuples():
    if row.test_date>="2020-01-01" and row.test_date<="2020-05-31":
        x_test.at[i,"test_date"]= 1
    elif row.test_date>="2020-10-01" and row.test_date<="2020-12-31":
        x_test.at[i,"test_date"]= 1
    elif row.test_date>="2021-01-01" and row.test_date<="2021-05-31":
        x_test.at[i,"test_date"]= 1
    elif row.test_date>="2021-10-01" and row.test_date<="2021-12-31":
        x_test.at[i,"test_date"]= 1
    elif row.test_date>="2020-06-01" and row.test_date<="2020-09-30":
        x_test.at[i,"test_date"]= 0
    elif row.test_date>="2021-06-01" and row.test_date<="2021-09-30":
        x_test.at[i,"test_date"]= 0
    i= i+1

#riportiamo i dataset agli indici originali altrimenti avremo il campo index
x_train_res= x_train_res.set_index("index")
y_train_res= y_train_res.set_index("index")
x_test= x_test.set_index("index")
y_test= y_test.set_index("index")

print("Fine Feature Scaling!")
