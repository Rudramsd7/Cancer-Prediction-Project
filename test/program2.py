from sklearn.preprocessing import LabelEncoder
import joblib

city=["mumbai","pune","nagpur","chennai"]

encoder=LabelEncoder()
encoded=encoder.fit_transform(city)
print(encoded)

joblib.dump(encoder,"city_encoded.enc")
print("City encoder successfully saved")