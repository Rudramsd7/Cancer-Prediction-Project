# wap to load the encoder from the file and find the label list
import joblib
encoder=joblib.load("cancer_ml_prediction/test/city_encoded.enc")
print(encoder.classes_)