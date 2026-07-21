import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

df=pd.read_csv("cancer_ml_prediction/years/train.csv",index_col=0)

x=df.drop("Survival_Years",axis=1)
y=df["Survival_Years"]

regressor=RandomForestRegressor()
regressor.fit(x,y)

joblib.dump(regressor,"cancer_ml_prediction/years/randomforest/train_model.dat")