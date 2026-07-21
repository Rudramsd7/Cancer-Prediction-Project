import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error,r2_score


df=pd.read_csv("cancer_ml_prediction/years/test.csv",index_col=0)

x=df.drop("Survival_Years",axis=1)
y=df["Survival_Years"]

regressor=joblib.load("cancer_ml_prediction/years/randomforest/train_model.dat")
y_pred=regressor.predict(x)

accuracy=r2_score(y,y_pred)
print("accuracy=",accuracy*100)

r2=r2_score(y,y_pred)
print("r2_score=",r2)

mse=mean_squared_error(y,y_pred)
print("mean_squared_error=",mse)