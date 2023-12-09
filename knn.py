from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
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


# Determine the split indices for 1/3 and 2/3 portions
one_third = int(len(X) / 3)
two_thirds = one_third * 2

one_thirdY = int(len(y) / 3)
two_thirdsY = one_third * 2

# Splitting the array into 1/3 and 2/3 portions
xTest = X[:one_third]
xTraining = X[one_third:two_thirds]

yTest = y[:one_third]
yTraining = y[one_third:two_thirds]

#################

neigh = KNeighborsClassifier()

neigh.fit(xTraining, yTraining)

yPred = neigh.predict(xTest)

confusion_matrix(yPred, yTest)




