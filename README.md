# Dataset Diabete

Este é um conjunto de dados sobre diabetes que pode ser usado para análise e modelagem preditiva. O arquivo .py contém um script que realiza algumas operações básicas nesse conjunto de dados.

## Funcionamento

O script .py realiza as seguintes etapas:

1. Carrega o conjunto de dados de diabetes.
2. Realiza uma análise exploratória dos dados, exibindo informações estatísticas e visualizações gráficas.
3. Divide o conjunto de dados em conjuntos de treinamento e teste.
4. Treina um modelo de aprendizado de máquina usando o conjunto de treinamento.
5. Avalia o desempenho do modelo usando o conjunto de teste.
6. Gera métricas de avaliação, como precisão, recall e F1-score.
7. Salva o modelo treinado em um arquivo para uso futuro.

## Requisitos

Para executar o script, você precisará ter as seguintes bibliotecas Python instaladas:

- Pandas
- Matplotlib
- Scikit-learn

Certifique-se de ter essas bibliotecas instaladas antes de executar o script.

## Executando o Script

Para executar o script, siga estas etapas:

1. Abra um terminal ou prompt de comando.
2. Navegue até o diretório onde o arquivo .py está localizado.
3. Execute o comando `python nome_do_arquivo.py`, substituindo "nome_do_arquivo.py" pelo nome real do arquivo.

Após a execução do script, você verá as saídas no terminal e os gráficos gerados serão salvos em arquivos PNG.

## Contribuição

Se você tiver alguma sugestão ou melhoria para este projeto, sinta-se à vontade para contribuir. Basta abrir uma issue ou enviar um pull request.

Referências:

- [Dataset Diabete](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- [Scikit-learn](https://scikit-learn.org/stable/)

## Exemplo de saída


Ao executar o caso de random_forest_stratified_kfold.py por exemplo, temos a seguinte saída:

```
Contagem de classes: [213703   4631  35346]
Acurácia em porcentagem: 84.1292179123305 %
Precisão class 0 em porcentagem: 96.59199843257562 %
Precisão class 1 em porcentagem: 0.04319654427645789 %
Precisão class 2 em porcentagem: 19.79570292707175 %
Recall class 0 em porcentagem: 86.42079238782415 %
Recall class 1 em porcentagem: 0.9263157894736843 %
Recall class 2 em porcentagem: 48.05951329652014 %
Especificidade class 0 em porcentagem: 50.90812780925742 %
Especificidade class 1 em porcentagem: 98.17341649478888 %
Especificidade class 2 em porcentagem: 88.14454124945176 %
F1-Score class 0 em porcentagem: 91.22333725713622 %
F1-Score class 1 em porcentagem: 0.08247738249098702 %
F1-Score class 2 em porcentagem: 28.014815971957567 %
Tempo de execução: 202.74721121788025 segundos
```


