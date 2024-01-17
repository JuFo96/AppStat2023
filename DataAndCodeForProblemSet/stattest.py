# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 20:35:58 2024

@author: Julius
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataframe = pd.read_csv('data_AnorocDisease.csv', header=0)
PatientID, Temp, BloodP, Age, Status = dataframe.values.T
df = dataframe

healthy_data = df[Status == 0]
sick_data = df[Status == 1]

plt.hist(healthy_data.iloc[:, 1], bins=20,
         edgecolor='black', alpha=0.7, label='Healthy')

plt.hist(sick_data.iloc[:, 1], bins=20,
         edgecolor='black', alpha=0.7, label='Sick')
plt.legend()
plt.title("Temperatures of healthy and sick population")
plt.ylabel("Count")
plt.xlabel("Temperature [C]")
#plt.savefig("plots/Temperature.png", bbox_inches="tight")
plt.show()

plt.hist(healthy_data.iloc[:, 2], bins=20,
         edgecolor='black', alpha=0.7, label='Healthy')

plt.hist(sick_data.iloc[:, 2], bins=20,
         edgecolor='black', alpha=0.7, label='Sick')
plt.legend()
plt.title("Blood Pressure of healthy and sick population")
plt.ylabel("Count")
plt.xlabel("Blood Pressure")
#plt.savefig("plots/Blood_Pressure.png", bbox_inches="tight")
plt.show()

plt.hist(healthy_data.iloc[:, 3], bins=20,
         edgecolor='black', alpha=0.7, label='Healthy')

plt.hist(sick_data.iloc[:, 3], bins=20,
         edgecolor='black', alpha=0.7, label='Sick')
plt.legend()
plt.title("Age of healthy and sick population")
plt.ylabel("Count")
plt.xlabel("Age [years]")
#plt.savefig("plots/Age.png", bbox_inches="tight")
plt.show()
