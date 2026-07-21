import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Read data from dataset
df=pd.read_csv("cancer_ml_prediction/cancer.csv")

# Drop null or empty or nan records
df=df.dropna()

# Drop unwanted columns
# df=df.drop(["Patient_ID"],axis=1)
df=df.drop(["Patient_ID",'Age','Gender', 'Country_Region', 'Year', 'Genetic_Risk',
       'Air_Pollution', 'Alcohol_Use', 'Smoking', 'Obesity_Level','Treatment_Cost_USD'],axis=1)

# Perform Label Encoding

encoder=LabelEncoder()
# df["Gender"]=encoder.fit_transform(df["Gender"])
# df["Country_Region"]=encoder.fit_transform(df["Country_Region"])
df["Cancer_Type"]=encoder.fit_transform(df["Cancer_Type"])
df["Cancer_Stage"]=encoder.fit_transform(df["Cancer_Stage"])
print(df)

df.to_csv("cancer_ml_prediction/years/preprocessed.csv")


train, test = train_test_split(df, test_size=0.2, random_state=10)

train.to_csv("cancer_ml_prediction/years/train.csv")
test.to_csv("cancer_ml_prediction/years/test.csv")
