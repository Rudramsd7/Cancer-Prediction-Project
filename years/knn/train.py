import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import joblib

df=pd.read_csv("cancer_ml_prediction/years/train.csv",index_col=0)

x=df.drop("Survival_Years",axis=1)
y=df["Survival_Years"]

regressor=KNeighborsRegressor()
regressor.fit(x,y)

joblib.dump(regressor,"cancer_ml_prediction/years/knn/train_model.dat")