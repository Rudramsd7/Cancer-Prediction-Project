import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib

df=pd.read_csv("cancer_ml_prediction/cost/train.csv",index_col=0)

x=df.drop("Treatment_Cost_USD",axis=1)
y=df["Treatment_Cost_USD"]

regressor=GradientBoostingRegressor(max_depth=15)
regressor.fit(x,y)

joblib.dump(regressor,"cancer_ml_prediction/cost/GradientBoosting/train_model.dat")