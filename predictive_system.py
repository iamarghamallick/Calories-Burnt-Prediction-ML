import numpy as np
import pickle


# loading the saved model
loaded_model = pickle.load(open('saved model/trained_model.sav', 'rb'))

input_data = (1,36,177.0,76.0,1.0,74.0,37.8)

# changing tuple to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)[0]
print(prediction)