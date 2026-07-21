import pandas as pd
from sklearn.svm import SVR
import joblib

df=pd.read_csv("cancer_ml_prediction/severity/train.csv",index_col=0)

x=df.drop("Target_Severity_Score",axis=1)
y=df["Target_Severity_Score"]

regressor=SVR()
regressor.fit(x,y)

joblib.dump(regressor,"cancer_ml_prediction/severity/knn/train_model.dat")