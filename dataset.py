file_path = "./archive/diabetes_012_health_indicators_BRFSS2015.txt"

# Initialize empty lists to store variable names and their respective values
variable_names = []
values_list = []

# Read the file and process its contents
with open(file_path, 'r') as file:
    lines = file.readlines()

# Split the lines to get variable names and values
for line_num, line in enumerate(lines):
    if line_num == 0:
        variable_names = line.strip().split(',')
    else:
        values = [float(val) if val.replace('.', '', 1).isdigit() else val for val in line.strip().split(',')]
        values_list.append(values)

# Display the variable names
print("Variable Names:", variable_names)

# Display the values
print("Values:")
for values in values_list:
    print(values)