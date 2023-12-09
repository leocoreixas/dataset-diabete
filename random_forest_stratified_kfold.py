from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
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

n_splits = 10
skf = StratifiedKFold(n_splits=n_splits)
skf.get_n_splits(X, y)
# Inicializa a classe do classificador Floresta Aleatória
rf = RandomForestClassifier()

acuracia_aux = []
precisao1_aux = []
precisao2_aux = []
precisao3_aux = []
recall1_aux = []
recall2_aux = []
recall3_aux = []
especificidade1_aux = []
especificidade2_aux = []
especificidade3_aux = []
f1_1_aux = []
f1_2_aux = []
f1_3_aux = []
StratifiedKFold(n_splits=n_splits, random_state=None, shuffle=False)
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    rf.fit(X[train_index], y[train_index])
    y_pred = rf.predict(X[test_index])
    conf = confusion_matrix(y_pred, y[test_index])
    print("Matriz de confusão index:" + str(i))
    print(conf)
    total = len(y_pred)
    # Acurácia
    acuracia = (conf[0][0] + conf[1][1] + conf[2][2]) / total
    # Precisão class 0
    precisao1 = conf[0][0] / (conf[0][0] + conf[1][0] + conf[2][0])
    # Precisão class 1
    precisao2 = conf[1][1] / (conf[0][1] + conf[1][1] + conf[2][1])
    # Precisão class 2
    precisao3 = conf[2][2] / (conf[0][2] + conf[1][2] + conf[2][2]) 

    # Recall ou Revocação class 0
    recall1 = conf[0][0] / (conf[0][0] + conf[0][1] + conf[0][2])
    # Recall ou Revocação class 1
    recall2 = conf[1][1] / (conf[1][0] + conf[1][1] + conf[1][2])
    # Recall ou Revocação class 2
    recall3 = conf[2][2] / (conf[2][0] + conf[2][1] + conf[2][2])

    # Especificidade class 0
    especificidade1 = (conf[1][1] + conf[1][2] + conf[2][1] + conf[2][2]) / (conf[1][1] + conf[1][2] + conf[2][1] + conf[2][2] + conf[1][0] + conf[2][0])
    # Especificidade class 1
    especificidade2 = (conf[0][0] + conf[0][2] + conf[2][0] + conf[2][2]) / (conf[0][0] + conf[0][2] + conf[2][0] + conf[2][2] + conf[0][1] + conf[2][1])
    # Especificidade class 2
    especificidade3 = (conf[0][0] + conf[0][1] + conf[1][0] + conf[1][1]) / (conf[0][0] + conf[0][1] + conf[1][0] + conf[1][1] + conf[0][2] + conf[1][2])

    # F1-Score class 1
    if precisao1 == 0 or recall1 == 0:
        f1_1 = 0.0
    else:
        f1_1 = 2 * (precisao1 * recall1) / (precisao1 + recall1)
    # F1-Score class 1
    if precisao2 == 0 or recall2 == 0:
        f1_2 = 0.0
    else:  
        f1_2 = 2 * (precisao2 * recall2) / (precisao2 + recall2)
    # F1-Score class 2
    if precisao3 == 0 or recall3 == 0:
        f1_3 = 0.0
    else:
        f1_3 = 2 * (precisao3 * recall3) / (precisao3 + recall3)
    
    acuracia_aux.append(acuracia)
    precisao1_aux.append(precisao1)
    precisao2_aux.append(precisao2)
    precisao3_aux.append(precisao3)
    recall1_aux.append(recall1)
    recall2_aux.append(recall2)
    recall3_aux.append(recall3)
    especificidade1_aux.append(especificidade1)
    especificidade2_aux.append(especificidade2)
    especificidade3_aux.append(especificidade3)
    f1_1_aux.append(f1_1)
    f1_2_aux.append(f1_2)
    f1_3_aux.append(f1_3)

print("Acurácia em porcentagem:", np.mean(acuracia_aux) * 100 , "%")
print("Precisão class 0 em porcentagem:", np.mean(precisao1_aux) * 100 , "%")
print("Precisão class 1 em porcentagem:", np.mean(precisao2_aux) * 100 , "%")
print("Precisão class 2 em porcentagem:", np.mean(precisao3_aux) * 100 , "%")
print("Recall class 0 em porcentagem:", np.mean(recall1_aux) * 100 , "%")
print("Recall class 1 em porcentagem:", np.mean(recall2_aux) * 100 , "%")
print("Recall class 2 em porcentagem:", np.mean(recall3_aux) * 100 , "%")
print("Especificidade class 0 em porcentagem:", np.mean(especificidade1_aux) * 100 , "%")
print("Especificidade class 1 em porcentagem:", np.mean(especificidade2_aux) * 100 , "%")
print("Especificidade class 2 em porcentagem:", np.mean(especificidade3_aux) * 100 , "%")
print("F1-Score class 0 em porcentagem:", np.mean(f1_1_aux) * 100 , "%")
print("F1-Score class 1 em porcentagem:", np.mean(f1_2_aux) * 100 , "%")
print("F1-Score class 2 em porcentagem:", np.mean(f1_3_aux) * 100 , "%")



end_time = time.time()
execution_time = end_time - start_time
print("Tempo de execução:", execution_time, "segundos")     
    
    
    