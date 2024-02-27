# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 08:37:27 2024

@author: HP
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('E:/datasets/credit.csv')

data.isnull().sum()
data.dropna()
data.columns
data.drop(['phone'], axis=1)

lb = LabelEncoder()
data['checking_balance'] = lb.fit_transform(data['checking_balance'])
data['credit_history'] = lb.fit_transform(data['credit_history'])
data['purpose'] = lb.fit_transform(data['purpose'])
data['savings_balance'] = lb.fit_transform(data['savings_balance'])
data['employment_duration'] = lb.fit_transform(data['employment_duration'])
data['other_credit'] = lb.fit_transform(data['other_credit'])
data['housing'] = lb.fit_transform(data['housing'])
data['job'] = lb.fit_transform(data['job'])

data['default'].unique()
data['default'].value_counts()
colnames = list(data.columns)

predictors = colnames[:15]
target = colnames[15]

from sklearn.model_selection import train_test_split
train,test = train_test_split(data, test_size=0.2)

from sklearn.tree import DecisionTreeClassifier as DT

model = DT(criterion='entropy')
model.fit(train[predictors], train[target])
preds = model.predict(test[predictors])
preds

pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['predictions'])
print(np.mean(preds==test[target]))

#overfit model
preds_train = model.predict(train[predictors])
pd.crosstab(train[target], preds_train, rownames=['Actual'], colnames=['predictions'])
print(np.mean(preds_train==train[target]) * 100)




























