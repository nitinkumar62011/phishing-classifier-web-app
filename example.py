import pandas as pd
predict_data=pd.read_csv("./Prediction_Output_File/Predictions.csv")
predict_data_total=predict_data['Predictions'].value_counts()
print("predict_data_total",predict_data_total)