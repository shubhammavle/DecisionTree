# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 08:31:04 2024

@author: HP
"""

import pandas as pd
import numpy as np

df = pd.read_csv('e:/datasets/Company_Data.csv')
df.head()

df.columns
df.dtypes
df.info()
df.describe

df.ShelveLoc
df.Urban
df.US

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

df.ShelveLoc = lb.fit_transform(df.ShelveLoc)
df.Urban = lb.fit_transform(df.Urban)
df.US = lb.fit_transform(df.US)

df.Sales = df.Sales.apply(lambda x: int(x))

df.head()

predictors = list(df.columns)[1:7]
target = list(df.columns)[0]
print(predictors, target)

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy')
model.fit(train[predictors], train[target])
pred = model.predict(train[predictors])
print(np.mean(pred == train[target])) #overfit model
