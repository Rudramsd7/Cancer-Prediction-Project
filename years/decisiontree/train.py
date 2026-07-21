import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import joblib

df=pd.read_csv("cancer_ml_prediction/years/train.csv",index_col=0)

x=df.drop("Survival_Years",axis=1)
y=df["Survival_Years"]

regressor=DecisionTreeRegressor()
regressor.fit(x,y)

joblib.dump(regressor,"cancer_ml_prediction/years/decisiontree/train_model.dat")