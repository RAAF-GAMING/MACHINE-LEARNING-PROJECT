import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

datapath = os.path.join("dataset", "")
training = pd.read_csv(datapath + "covid_data_training.csv")

#split
x_train=training.iloc[ : , 0:8]
y_train=training.iloc[ : , 8]

#decisionTree
tree_model = DecisionTreeClassifier(random_state=42)
#Alleniamo il nostro modello
tree_model.fit(x_train, y_train)
scelta=1
while scelta==1:
    data=input('Ti trovi in un mese tra ottobre-maggio?==1, oppure giugno-settembre?==0\n')
    tosse=int(input('Hai la tosse?, 1=SI 0=NO\n'))
    febbre=int(input('Hai la febbre?, 1=SI 0=NO\n'))
    mal_di_gola=int(input('Hai mal di gola?, 1=SI 0=NO\n'))
    fiato=int(input('Hai il fiato corto?, 1=SI 0=NO\n'))
    mal_di_testa=int(input('Hai mal di testa?, 1=SI 0=NO\n'))
    età=int(input('Hai più di 60 anni?, 1=SI 0=NO\n'))
    contatti_con_positivi=int(input('Hai avuto contatti con positivi?, 1=SI 0=NON LO SO\n'))
    sintomi={"test_date": [data],"cough":[tosse], "fever":[febbre],"sore_throat":[mal_di_gola],"shortness_of_breath":[fiato],"head_ache":[mal_di_testa],"age_60_and_above":[età],
    "test_indication":[contatti_con_positivi]}
    istanza=pd.DataFrame(sintomi)
    y_pred = tree_model.predict(istanza)
    print("la predizione è: ",y_pred)
    scelta=int(input('vuoi continuare?, 1=SI 0=NO\n'))
