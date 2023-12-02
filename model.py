from sklearn.neural_network import MLPClassifier
import numpy as np

file_path = "./archive/diabetes_012_health_indicators_BRFSS2015.txt"

# Initialize empty lists to store variable names and their respective values
target_array = []  # For the first column
data_matrix = []   

# Read the file and process its contents
with open(file_path, 'r') as file:
    lines = file.readlines()

# Read the file and process its contents
with open(file_path, 'r') as file:
    lines = file.readlines()

# Separate the data into array and matrix
for line_num, line in enumerate(lines):
    if line_num == 0:
        variable_names = line.strip().split(',')
    else:
        values = line.strip().split(',')
        target_array.append(values[0])   # Collecting values from the first column (array)
        data_matrix.append(values[1:])  #

X = np.array(data_matrix).astype(float)
y = np.array(target_array).astype(float)

# Initialize and train the MLPClassifier model
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
clf.fit(X, y)

# Make predictions using the trained model
predictions_1 = clf.predict(X[:21])
predictions_2 = clf.predict(X[21:])

# Display the predictions
print("Prediction 1:", predictions_1)
print("Prediction 2:", predictions_2)