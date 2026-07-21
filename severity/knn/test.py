import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error,r2_score


df=pd.read_csv("cancer_ml_prediction/severity/test.csv",index_col=0)

x=df.drop("Target_Severity_Score",axis=1)
y=df["Target_Severity_Score"]

regressor=joblib.load("cancer_ml_prediction/severity/knn/train_model.dat")
y_pred=regressor.predict(x)

accuracy=r2_score(y,y_pred)
print(accuracy)