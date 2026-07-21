import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error,r2_score


df=pd.read_csv("cancer_ml_prediction/cost/test.csv",index_col=0)

x=df.drop("Treatment_Cost_USD",axis=1)
y=df["Treatment_Cost_USD"]

regressor=joblib.load("cancer_ml_prediction/cost/linear/train_model.dat")
y_pred=regressor.predict(x)

accuracy=r2_score(y,y_pred)
print("accuracy=",accuracy)

r2=r2_score(y,y_pred)
print("r2_score=",r2)

mse=mean_squared_error(y,y_pred)
print("mean_squared_error=",mse)