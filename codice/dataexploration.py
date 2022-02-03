# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
#as rename 
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

#load our data
datapath = os.path.join("dataset", "")

covid = pd.read_csv(datapath + "covid_data_2020-2021.csv")

#distribution of data
print(covid.describe())

print("\n\nanalisi test date")
print(covid["test_date"].describe())

print("\n\nanalisi gender")
print(covid["gender"].describe())

print("\n\nanalisi test_indication")
print(covid["test_indication"].describe())

print("\n\nVisualizzazione di possibili valori NULL")
print(covid.isnull().sum())