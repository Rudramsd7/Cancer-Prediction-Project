import joblib
country_encoder=joblib.load("cancer_ml_prediction/severity/Country_Region_encoded.enc")

country_name="Brazil"
print("Country List:",country_encoder.classes_)
def encodeCountry(country):
    try:
        encoded=country_encoder.transform([country])
        return encoded[0]
    except:
        return None
    
countryVal=encodeCountry(country_name)
if countryVal==None:
    print("Country not found")
else:
    print(country_name+"Value=",countryVal)