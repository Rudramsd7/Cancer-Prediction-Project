import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df=pd.read_csv("cancer_ml_prediction/years/train.csv",index_col=0)

x=df.drop("Survival_Years",axis=1)
y=df["Survival_Years"]

regressor=LinearRegression()
regressor.fit(x,y)

joblib.dump(regressor,"cancer_ml_prediction/years/linear/train_model.dat")