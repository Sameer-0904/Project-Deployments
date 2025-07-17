import numpy as np
import pickle
import streamlit as st
import warnings 
warnings.filterwarnings('ignore')

# Loading saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Creating a function for prediction
def diabetes_prediction(input_data):
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
    
def main():
    
    # Title of WebApp
    st.set_page_config(page_title="Diabetes Prediction Web App")
    st.title("Diabetes Prediction Web App")
    
    # Getting input from user
    Pregnancies = st.text_input("Number of Pregancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure Level")
    SkinThickness = st.text_input("Skin Thikness Level")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedegree Function")
    Age = st.text_input("Age")
    
    # Code for prediction
    diagnosis = ''
    
    # Create a button
    if st.button("Get Your Diabetes report"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)
    

if __name__ == "__main__":
    main()
