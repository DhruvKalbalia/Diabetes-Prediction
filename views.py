from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn import svm #uses support vector machine classifier to evaluate the model
from sklearn.metrics import accuracy_score

def index(request):
    return render(request, 'index.html')
def predict(request):
    return render(request, 'model.html')
def result(request):
    data = pd.read_csv(r"C:\Users\ADMIN\Desktop\Diabetes\data.csv") #importing data
    X = data.drop(columns = 'Outcome', axis=1)#stores data without the outcome column
    Y = data['Outcome']#stores the outcome part of the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

    classifier = LogisticRegression() 
    classifier.fit(X_train, Y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    pred = classifier.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

    result1 = ""
    if pred ==[1]:
        result1 = "POSITIVE"
    else:
        result1 = "NEGATIVE"

    return render(request, 'model.html', {"result2":result1, "no1": val1, "no2": val2, "no3": val3, "no4": val4, "no5": val5, "no6": val6, "no7": val7, "no8": val8})