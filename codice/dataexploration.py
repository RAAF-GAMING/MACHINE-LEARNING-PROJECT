#(Data Understanding)Qui andiamo a fare un esplorazione dei dati: valori null, relazioni, e descrizione dei dati.
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
#as rename 
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


def exploration(covid):
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