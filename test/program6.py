import joblib
Gender_encoder=joblib.load("cancer_ml_prediction/severity/gender_encoded.enc")
# print(Gender_encoded.classes_)

def encodeGender(gender):
    try:
        encoded=Gender_encoder.transform([gender])
        return encoded[0]
    except:
        return None
    
genderVal=encodeGender("Male")
if genderVal==None:
    print("Gender not found")
else:
    print("GenderValue=",genderVal)