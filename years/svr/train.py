import pandas as pd
from sklearn.svm import SVR
import joblib

df=pd.read_csv("cancer_ml_prediction/years/train.csv",index_col=0)

x=df.drop("Survival_Years",axis=1)
y=df["Survival_Years"]

regressor=SVR()
regressor.fit(x,y)

joblib.dump(regressor,"cancer_ml_prediction/years/svr/train_model.dat")