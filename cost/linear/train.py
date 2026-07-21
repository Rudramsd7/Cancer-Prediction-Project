import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df=pd.read_csv("cancer_ml_prediction/cost/train.csv",index_col=0)

x=df.drop("Treatment_Cost_USD",axis=1)
y=df["Treatment_Cost_USD"]

regressor=LinearRegression()
regressor.fit(x,y)

joblib.dump(regressor,"cancer_ml_prediction/cost/linear/train_model.dat")