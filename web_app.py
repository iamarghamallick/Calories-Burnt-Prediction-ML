import numpy as np
import pickle
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open('saved model/trained_model.sav', 'rb'))


# Function for prediction
def cal_burnt_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=np.float32)  # Convert input to float32
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)[0]
    return f"You have burnt {prediction:.2f} cal energies"  # Format prediction to two decimal places


def main():
    st.title("Calories Burnt Prediction Web App")

    # Collect input data from user
    Gender = st.text_input('Gender')
    Age = st.text_input('Age')
    Height = st.text_input('Height')
    Weight = st.text_input('Weight')
    Duration = st.text_input('Duration')
    Heart_Rate = st.text_input('Heart_Rate')
    Body_Temp = st.text_input('Body_Temp')

    result = ""

    if st.button('Test Result'):
        input_data_list = [Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp]

        # Check for missing data and convert to numeric values
        if all(data != '' for data in input_data_list):
            input_data = tuple(map(float, input_data_list))
            result = cal_burnt_prediction(input_data)
        else:
            result = "Please fill in all fields."

    st.success(result)


if __name__ == "__main__":
    main()
