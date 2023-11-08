import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load your trained model
model = pickle.load(open('model2.pkl', 'rb'))
fet = pd.read_csv("Train.csv")
# Create a Streamlit web app
st.title('Bank Fraud Detection App')

# Input features from the user

rental30 = st.number_input('Enter Rental 30:', min_value=0.0)
rental90 = st.number_input('Enter Rental 90:', min_value=0.0)
maxamnt_loans90 = st.number_input('Enter Max Amount Loans 90:', min_value=0.0)

# Create a button to make predictions
if st.button('Predict'):

    df=fet[(fet['rental30']==rental30) & (fet['rental90']==rental90) & (fet['maxamnt_loans90']==maxamnt_loans90)]
    f_features=[]
    
    # Create a DataFrame with the user's input
    f_features.append(df['aon'])
    f_features.append(df['daily_decr30'])
    f_features.append(df['daily_decr90'])
    f_features.append(df['rental30'])
    f_features.append(df['rental90'])
    f_features.append(df['cnt_loans30'])
    f_features.append(df['amnt_loans30'])
    f_features.append(df['maxamnt_loans30'])
    f_features.append(df['cnt_loans90'])
    f_features.append(df['amnt_loans90'])
    f_features.append(df['maxamnt_loans90'])
    f_features.append(df['payback30'])
    f_features.append(df['payback30'])
    f_features.append(df['Day'])
    f_features.append(df['Month'])
    f_features.append(df['Year'])
    
        

    # Get the prediction
    final_features = (np.array(f_features).reshape(1,-1))
    final_features = final_features[:, :16]
    prediction = model.predict(final_features)

    # Display the prediction
    if prediction[0] == 0:
        st.write('Prediction: Legitimate Transaction')
    else:
        st.write('Prediction: Fraudulent Transaction')
