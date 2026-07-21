import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
# Read data from dataset
df=pd.read_csv("cancer_ml_prediction/cancer.csv")

# Drop null or empty or nan records
df=df.dropna()

# Drop unwanted columns
df=df.drop(["Patient_ID","Treatment_Cost_USD","Survival_Years"],axis=1)

# Perform Label Encoding

encoder=LabelEncoder()
df["Gender"]=encoder.fit_transform(df["Gender"])
joblib.dump(encoder,"cancer_ml_prediction/severity/Gender_encoded.enc")

df["Country_Region"]=encoder.fit_transform(df["Country_Region"])
joblib.dump(encoder,"cancer_ml_prediction/severity/Country_Region_encoded.enc")

df["Cancer_Type"]=encoder.fit_transform(df["Cancer_Type"])
joblib.dump(encoder,"cancer_ml_prediction/severity/Cancer_Type_enooded.enc")

df["Cancer_Stage"]=encoder.fit_transform(df["Cancer_Stage"])
joblib.dump(encoder,"cancer_ml_prediction/severity/Cancer_Stage_encoded.enc")

print(df)

df.to_csv("cancer_ml_prediction/severity/preprocessed.csv")


train, test = train_test_split(df, test_size=0.2, random_state=10)

train.to_csv("cancer_ml_prediction/severity/train.csv")
test.to_csv("cancer_ml_prediction/severity/test.csv")


# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split

# df = pd.read_csv("cancer_ml_prediction/cancer.csv")

# df = df.dropna()
# df = df.drop(["Patient_ID", "Treatment_Cost_USD", "Survival_Years"], axis=1)

# encoder = LabelEncoder()
# encode_columns = ["Gender", "Country_Region", "Cancer_Stage", "Cancer_Type"]

# for column in encode_columns:
#     df[column] = encoder.fit_transform(df[column])


# df.to_csv("cancer_ml_prediction/severity/preprocessed.csv", index=False)

# train, test = train_test_split(df, test_size=0.2, random_state=10)

# train.to_csv("cancer_ml_prediction/severity/train.csv", index=False)
# test.to_csv("cancer_ml_prediction/severity/test.csv", index=False)