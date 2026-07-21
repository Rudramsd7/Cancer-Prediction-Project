import joblib
cancer_stage_encoder=joblib.load("cancer_ml_prediction/severity/Cancer_Stage_encoded.enc")

cancer_stage="Stage 0"
print("Cancer_Stage List:",cancer_stage_encoder.classes_)
def encodeCancerStage(cancerstage):
    try:
        encoded=cancer_stage_encoder.transform([cancerstage])
        return encoded[0]
    except:
        return None
    
cancer_stageVal=encodeCancerStage(cancer_stage)
if cancer_stageVal==None:
    print("Cancer_Type not found")
else:
    print(cancer_stage+"Value=",cancer_stageVal)