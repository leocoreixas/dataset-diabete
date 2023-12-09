from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import time

file_path = "./archive/diabetes_012_health_indicators_BRFSS2015.txt"

target_array = []
data_matrix = []   

with open(file_path, 'r') as file:
    lines = file.readlines()
with open(file_path, 'r') as file:
    lines = file.readlines()

for line_num, line in enumerate(lines):
    if line_num == 0:
        variable_names = line.strip().split(',')
    else:
        values = line.strip().split(',')
        target_array.append(values[0])   
        data_matrix.append(values[1:])

X = np.array(data_matrix).astype(float)
y = np.array(target_array).astype(float)

y_int = y.round().astype(np.int64)

# Encontrar a contagem de cada classe presente no array y_int
contagem_classes = np.bincount(y_int)
print("Contagem de classes:", contagem_classes)

start_time = time.time()

# Utilizando hold out
# Determina em 1/3 e 2/3 
one_third = int(len(X) / 3)
two_thirds = one_third * 2
one_thirdY = int(len(y) / 3)
two_thirdsY = one_thirdY * 2

# Separa na proporção 1/3 e 2/3 para treino e teste
xTest = X[:one_third]
xTraining = X[one_third:two_thirds]
yTest = y[:one_thirdY]
yTraining = y[one_thirdY:two_thirdsY]

# Inicializa a classe do classificador KNN
neigh = KNeighborsClassifier()

# Treina o modelo com os dados de treino e teste
neigh.fit(xTraining, yTraining)

# Faz a predição com os dados de teste
yPred = neigh.predict(xTest)

# Calcula a matriz de confusão 
conf = confusion_matrix(yPred, yTest)
print("Matriz de confusão:")
print(conf)
total = len(yPred)
print("Total de amostras:", total)
# Acurácia
acuracia = (conf[0][0] + conf[1][1] + conf[2][2]) / total
print("Acurácia em porcentagem:", acuracia * 100 , "%")

# Precisão class 0
precisao1 = conf[0][0] / (conf[0][0] + conf[1][0] + conf[2][0])
# Precisão class 1
precisao2 = conf[1][1] / (conf[0][1] + conf[1][1] + conf[2][1])
# Precisão class 2
precisao3 = conf[2][2] / (conf[0][2] + conf[1][2] + conf[2][2]) 
print("Precisão class 0 em porcentagem:", precisao1 * 100 , "%")
print("Precisão class 1 em porcentagem:", precisao2 * 100 , "%")
print("Precisão class 2 em porcentagem:", precisao3 * 100 , "%")

# Recall ou Revocação class 0
recall1 = conf[0][0] / (conf[0][0] + conf[0][1] + conf[0][2])
# Recall ou Revocação class 1
recall2 = conf[1][1] / (conf[1][0] + conf[1][1] + conf[1][2])
# Recall ou Revocação class 2
recall3 = conf[2][2] / (conf[2][0] + conf[2][1] + conf[2][2])
print("Recall class 0 em porcentagem:", recall1 * 100 , "%")
print("Recall class 1 em porcentagem:", recall2 * 100 , "%")
print("Recall class 2 em porcentagem:", recall3 * 100 , "%")

# Especificidade class 0
especificidade1 = (conf[1][1] + conf[1][2] + conf[2][1] + conf[2][2]) / (conf[1][1] + conf[1][2] + conf[2][1] + conf[2][2] + conf[1][0] + conf[2][0])
# Especificidade class 1
especificidade2 = (conf[0][0] + conf[0][2] + conf[2][0] + conf[2][2]) / (conf[0][0] + conf[0][2] + conf[2][0] + conf[2][2] + conf[0][1] + conf[2][1])
# Especificidade class 2
especificidade3 = (conf[0][0] + conf[0][1] + conf[1][0] + conf[1][1]) / (conf[0][0] + conf[0][1] + conf[1][0] + conf[1][1] + conf[0][2] + conf[1][2])
print("Especificidade class 0 em porcentagem:", especificidade1 * 100 , "%")
print("Especificidade class 1 em porcentagem:", especificidade2 * 100 , "%")
print("Especificidade class 2 em porcentagem:", especificidade3 * 100 , "%")


# F1-Score class 0
f1_1 = 2 * (precisao1 * recall1) / (precisao1 + recall1)
# F1-Score class 1
f1_2 = 2 * (precisao2 * recall2) / (precisao2 + recall2)
# F1-Score class 2
f1_3 = 2 * (precisao3 * recall3) / (precisao3 + recall3)
print("F1-Score class 0 em porcentagem:", f1_1 * 100 , "%")
print("F1-Score class 1 em porcentagem:", f1_2 * 100 , "%")
print("F1-Score class 2 em porcentagem:", f1_3 * 100 , "%")

end_time = time.time()
execution_time = end_time - start_time
print("Tempo de execução:", execution_time, "segundos")









