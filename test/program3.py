import joblib
encoder=joblib.load("cancer_ml_prediction/test/city_encoded.enc")
encoded=encoder.transform(["mumbai"])
print(encoded)