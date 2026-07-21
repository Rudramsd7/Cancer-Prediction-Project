import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import joblib

df=pd.read_csv("cancer_ml_prediction/cost/train.csv",index_col=0)

x=df.drop("Treatment_Cost_USD",axis=1)
y=df["Treatment_Cost_USD"]

regressor=KNeighborsRegressor()
regressor.fit(x,y)

joblib.dump(regressor,"cancer_ml_prediction/cost/knn/train_model.dat")