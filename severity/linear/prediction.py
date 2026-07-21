import joblib
import sys

Gender_encoder=joblib.load("cancer_ml_prediction/severity/Gender_encoded.enc")
country_encoder=joblib.load("cancer_ml_prediction/severity/Country_Region_encoded.enc")
cancer_type_encoder=joblib.load("cancer_ml_prediction/severity/Cancer_Type_encoded.enc")
cancer_stage_encoder=joblib.load("cancer_ml_prediction/severity/Cancer_Stage_encoded.enc")

def encodeGender(gender):
    try:
        encoded=Gender_encoder.transform([gender])
        return encoded[0]
    except:
        return None
    

def encodeCountry(country):
    try:
        encoded=country_encoder.transform([country])
        return encoded[0]
    except:
        return None
    
def encodeCancerType(cancertype):
    try:
        encoded=cancer_type_encoder.transform([cancertype])
        return encoded[0]
    except:
        return None
    
def encodeCancerStage(cancerstage):
    try:
        encoded=cancer_stage_encoder.transform([cancerstage])
        return encoded[0]
    except:
        return None
    
age_val=int(input("Enter age:"))

gender=input(f"Enter Gender {Gender_encoder.classes_}:")
gender_val=encodeGender(gender)

if gender_val==None:
    print("Incorrect Gender Value!")
    sys.exit()


country=input(f"Enter Country {country_encoder.classes_}:")
country_val=encodeCountry(country)

if country_val==None:
    sys.exit()


year_val=int(input("Enter Year:"))

genetic_risk_val=float(input("Enter Genetic Risk:"))

air_pollution_val=float(input("Enter Air Pollution:"))

alcohol_use_val=float(input("Enter Alcohol Use:"))

smoking_val=float(input("Enter Smoking:"))

obesity_level_val=float(input("Enter Obesity Level:"))

cancer_type=input(f"Enter Cancer Type {cancer_type_encoder.classes_} :")
cancer_type_val=encodeCancerType(cancer_type)

if cancer_type_val==None:
    sys.exit()

cancer_stage=input(f"Enter Cancer Stage {cancer_stage_encoder.classes_} :")
cancer_stage_val=encodeCancerStage(cancer_stage)

if cancer_stage_val==None:
    sys.exit()

regressor=joblib.load("cancer_ml_prediction/severity/linear/train_model.dat")

x=[[age_val,gender_val,country_val,year_val,genetic_risk_val,air_pollution_val,alcohol_use_val,smoking_val,obesity_level_val,cancer_type_val,cancer_stage_val]]

predicted=regressor.predict(x)
cancer_severity=predicted[0]
print("Predicted Cancer Severity=",cancer_severity)
# print("Good Bye!") 

cost_regressor=joblib.load("cancer_ml_prediction/cost/linear/train_model.dat")
cost_x=[[age_val,gender_val,country_val,year_val,genetic_risk_val,air_pollution_val,alcohol_use_val,smoking_val,obesity_level_val,cancer_type_val,cancer_stage_val,cancer_severity]]

predicted_cost=cost_regressor.predict(cost_x)
cancer_cost=predicted_cost[0]
print("Cost=",cancer_cost,"USD")



# import joblib
# import sys

# Gender_encoder=joblib.load("cancer_ml_prediction/severity/Gender_encoded.enc")
# country_encoder=joblib.load("cancer_ml_prediction/severity/Country_Region_encoded.enc")
# cancer_type_encoder=joblib.load("cancer_ml_prediction/severity/Cancer_Type_encoded.enc")
# cancer_stage_encoder=joblib.load("cancer_ml_prediction/severity/Cancer_Stage_encoded.enc")

# def encodeGender(gender):
#     try:
#         encoded=Gender_encoder.transform([gender])
#         return encoded[0]
#     except:
#         return None
    

# def encodeCountry(country):
#     try:
#         encoded=country_encoder.transform([country])
#         return encoded[0]
#     except:
#         return None
    
# def encodeCancerType(cancertype):
#     try:
#         encoded=cancer_type_encoder.transform([cancertype])
#         return encoded[0]
#     except:
#         return None
    
# def encodeCancerStage(cancerstage):
#     try:
#         encoded=cancer_stage_encoder.transform([cancerstage])
#         return encoded[0]
#     except:
#         return None
    
# age_val=int(input("Enter age:"))

# gender=input(f"Enter Gender {Gender_encoder.classes_}:")
# gender_val=encodeGender(gender)

# if gender_val==None:
#     print("Incorrect Gender Value!")
#     sys.exit()


# country=input(f"Enter Country {country_encoder.classes_}:")
# country_val=encodeCountry(country)

# if country_val==None:
#     sys.exit()


# year_val=int(input("Enter Year:"))

# genetic_risk_val=float(input("Enter Genetic Risk:"))

# air_pollution_val=float(input("Enter Air Pollution:"))

# alcohol_use_val=float(input("Enter Alcohol Use:"))

# smoking_val=float(input("Enter Smoking:"))

# obesity_level_val=float(input("Enter Obesity Level:"))

# cancer_type=input(f"Enter Cancer Type {cancer_type_encoder.classes_} :")
# cancer_type_val=encodeCancerType(cancer_type)

# if cancer_type_val==None:
#     sys.exit()

# cancer_stage=input(f"Enter Cancer Stage {cancer_stage_encoder.classes_} :")
# cancer_stage_val=encodeCancerStage(cancer_stage)

# if cancer_stage_val==None:
#     sys.exit()

# cost_regressor=joblib.load("cancer_ml_prediction/severity/linear/train_model.dat")

# x=[[age_val,gender_val,country_val,year_val,genetic_risk_val,air_pollution_val,alcohol_use_val,smoking_val,obesity_level_val,cancer_type_val,cancer_stage_val,cancer_severity]]
# predicted=cost_regressor.predict(x)
# predicted_Cost=predicted[0]
# print("Predicted Treatment Cost=",predicted_Cost)