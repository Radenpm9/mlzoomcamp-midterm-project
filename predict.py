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
from typing import Literal
from pydantic import BaseModel, Field


from fastapi import FastAPI
import uvicorn


class victim(BaseModel):
    {'Pclass': 3, 'Sex': 'female', 'SibSp': 0, 'Parch': 0}


class predict(BaseModel):
    y_pred_1: float

app = FastAPI(title="victim-survival-prediction")

#load the model f_in is file input
with open('model.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

def predict_single(victim):
    y_pred_1 = dtreg.predict(X_val)
    return float (y_pred_1)

@app.post("/predict")
def predict(victim: victim) -> PredictResponse:
    prob = predict_single(victim.model_dump())

    return PredictResponse(
        churn_probability=y_pred_1
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
