# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 09:17:45 2024

@author: HP
"""

import pandas as pd
import numpy as np

df = pd.read_csv('E:/datasets/salaries.csv')
df.head()

df.columns

inputs = df.drop('salary_more_then_100k', axis='columns')
target = df['salary_more_then_100k']

from sklearn.preprocessing import LabelEncoder
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])
inputs_n = inputs.drop(columns=['company','job','degree'])

from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(inputs_n, target)
model.predict([[2,1,0]])