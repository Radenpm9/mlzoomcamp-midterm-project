#!/usr/bin/env python
# coding: utf-8

# This file is for a midterm project of Machine Learning Zoomcamp.  
# 
# The project is using Titanic - Machine Learning from Disaster data with target to predict survival of passenger in 'Survived' column.  
# You can find the data from [here](https://www.kaggle.com/competitions/titanic/data?select=train.csv)  
# The original project on Kaggle has 3 separate data: train, test, and gender submission. This project use slightly different approach.  
# Only using train data for training and test, because the test data from Kaggle doesn't have target variable we're interested in. It is hidden so the model will be uploaded to Kaggle for verification.  
# We're not doing it for this project as we're building model and deploying it on cloud.  
# ***The priority of this project is model development (without refining the accuracy) and deployment.***  
# Refining model accuracy will be the next stage if we have chance to do it.

# In[24]:

import pickle
import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import export_text


from sklearn.tree import DecisionTreeClassifier

def load_data():
    data_url = 'https://raw.githubusercontent.com/Radenpm9/mlzoomcamp-midterm-project/6a065554b099e485250fdb7008aa6f061e2b5f0f/train.csv'
    titanic_data = pd.read_csv(data_url)
    #fill missing values with zeros
    titanic_data = titanic_data.fillna(0)
    return titanic_data


# Exploring the pattern, whether gender is a strong survival indicator or not.
# Filtering female passengers from titanic data. Then, we select just the 'Survived' column from those filtered rows.
women = titanic_data.loc[titanic_data.Sex == 'female']['Survived']

# sum(women): Since the women Series contains 0s (for not survived) and 1s (for survived), summing this Series will give you 
# the total number of female passengers who survived. (Each '1' adds one to the sum).
# len(women): This calculates the total number of elements in the women Series, 
# which corresponds to the total number of female passengers in the train_data DataFrame.
rate_women = sum(women)/len(women)

print("percentage of women who survived", rate_women*100,"%")

# Calculating the men percentage with the same logic as above.
men = titanic_data.loc[titanic_data.Sex == 'male']['Survived']
rate_men = sum(men)/len(men)

print("percentage of men who survived", rate_men*100,"%")


# Gender is a strong indication to predict passenger survival.  
# Now, we're going to build model using Random Forest. The model will take a closer look to four different columns ('Pclass', 'Sex', 'SibSp', and 'Parch') of the data.  
# The model will be using 100 of n_estimators, 5 of max_depth, and 1 of random_state.


def train_model():
    #do train/validation/test split with 60/20/20 distribution
    #create titanic data (td) full train and td test data set
    td_full_train, td_test = train_test_split(titanic_data, test_size = 0.2, random_state = 1)

    #create titanic data (td) train and td validation data set
    td_train, td_val = train_test_split(td_full_train, test_size = 0.25, random_state = 1)

    #extract target variable
    y_train = td_train['Survived'].values
    y_val = td_val['Survived'].values
    y_test = td_test['Survived'].values

    #delete all target variable from all partition to avoid mistakes (human error)
    del td_train['Survived']
    del td_val['Survived']
    del td_test['Survived']

    features = ['Pclass', 'Sex', 'SibSp', 'Parch']

    #create dictionary of td_train to apply one-hot encoding for categorical and numerical features we're interested in
    td_train_dict = td_train[features].to_dict(orient = 'records')

    #create feature matrix for numerical and one-hot encoding for categorical features
    dv = DictVectorizer(sparse = True)
    X_train = dv.fit_transform(td_train_dict)

    #train to predict target variable
    model = DecisionTreeClassifier(max_depth = 3, random_state = 1)
    dtreg = model.fit(X_train, y_train)

    # Create feature matrix of validation partition 
    td_val_dict = td_val[features].to_dict(orient='records')
    X_val = dv.fit_transform(td_val_dict)
    return dtreg, X_val


#creating rmse function
def rmse(y, y_pred):
    error = y_pred - y
    mse = (error**2).mean()
    return np.sqrt(mse)

def save_model(dv, model), f_out):
    with open('model.bin', 'wb') as f_out:
        pickle.dump((dv, model), f_out)



df = load_data()
(dv, model) = train_model(df)
save_model((dv, model), 'model.bin')

print('Model saved to model.bin')
