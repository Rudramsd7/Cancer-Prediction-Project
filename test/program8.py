import joblib
cancer_type_encoder=joblib.load("cancer_ml_prediction/severity/Cancer_Type_encoded.enc")

cancer_type="Skin"
print("Cancer_Type List:",cancer_type_encoder.classes_)
def encodeCancerType(cancertype):
    try:
        encoded=cancer_type_encoder.transform([cancertype])
        return encoded[0]
    except:
        return None
    
cancer_typeVal=encodeCancerType(cancer_type)
if cancer_typeVal==None:
    print("Cancer_Type not found")
else:
    print(cancer_type+"Value=",cancer_typeVal)