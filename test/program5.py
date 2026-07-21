import joblib
encoder=joblib.load("cancer_ml_prediction/test/city_encoded.enc")
# print(encoder.classes_)
def encodeCity(city):
    try:
        encoded=encoder.transform([city])
        return encoded[0]
    except:
        return None
    

cityValue=encodeCity("chennai")
if cityValue==None:
    print("City Not found")
else:
    print("City Value=",cityValue)