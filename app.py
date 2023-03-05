import joblib
import streamlit as st

# Load the saved model, StandardScaler, and LabelEncoder
loaded_model = joblib.load('rf_model.sav')
scaler = joblib.load('features_scaler.sav')
encoder = joblib.load('label_encoder.sav')

# Create a title and subtitle
st.write("""
# Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

# Create input variables for user to input values
col1, col2, col3, col4 = st.columns(4)
sl = col1.slider('Select Sepal Length:', 0.0, 10.0, 5.0)
sw = col2.slider('Select Sepal Width:', 0.0, 10.0, 5.0)
pl = col3.slider('Select Petal Length:', 0.0, 10.0, 5.0)
pw = col4.slider('Select Petal Width:', 0.0, 10.0, 5.0)

# Create new data for prediction
new_data = [[sl, sw, pl, pw]]

# Scale the features using the loaded StandardScaler
new_data_scaled = scaler.transform(new_data)

# Make predictions using the Random Forest model
predictions = loaded_model.predict(new_data_scaled)

# Decode the predicted target variable using the loaded LabelEncoder
decoded_predictions = encoder.inverse_transform(predictions)

st.write("""
## Prediction
The predicted Iris flower type is:
""")
st.write(decoded_predictions[0])