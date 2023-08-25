from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('model.pkl')

# Load the dataset
data = pd.read_csv('Data/synthetic-data-from-a-financial-payment-system/bs140513_032310.csv')

# Preprocess the data
data_reduced = data.drop(['zipcodeOri','zipMerchant'],axis=1)
col_categorical = data_reduced.select_dtypes(include= ['object']).columns
data_reduced[col_categorical] = data_reduced[col_categorical].apply(lambda x: x.cat.codes)
X = data_reduced.drop(['fraud'],axis=1)

# Function to make a prediction
def is_fraud(request):
    if request.method == 'POST':
        payment = request.POST.dict()
        payment = pd.DataFrame([payment], columns=X.columns)
        prediction = model.predict(payment)[0]
        if prediction == 1:
            result = 'fraudulent'
        else:
            result = 'not fraudulent'
        return JsonResponse({'result': result})
    return render(request, 'payment_form.html')