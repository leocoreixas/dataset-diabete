file_path = "./archive/diabetes_012_health_indicators_BRFSS2015.txt"

# Initialize empty lists to store variable names and their respective values
variable_names = []
values_dict = {}  # Use a dictionary to store variable names and their values

# Read the file and process its contents
with open(file_path, 'r') as file:
    lines = file.readlines()

# Split the lines to get variable names and values
for line_num, line in enumerate(lines):
    if line_num == 0:
        variable_names = line.strip().split(',')
        for name in variable_names:
            values_dict[name] = []  # Initialize an empty list for each variable name

    else:
        values = [float(val) if val.replace('.', '', 1).isdigit() else val for val in line.strip().split(',')]
        for i, val in enumerate(values):
            values_dict[variable_names[i]].append(val)

# Display all values corresponding to each variable name
for name, values in values_dict.items():
    print(f"{name}: {values}")