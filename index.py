# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
datasetUrl = []

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if(filename == 'bs140513_032310.csv'):
            datasetUrl.append(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# Generate and plot a synthetic imbalanced classification dataset
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
import statistics as stats
import itertools

# Clusters (grupos):
#     - Idade: 1-2, 3-4, 5-6
#     - Gênero: M, F
#     - Valor: <=50, <=200, >200
# lista de categorias para possível novo grupo:
#    "category_'es_barsandrestaurants'", "category_'es_contents'", "category_'es_fashion'", "category_'es_food'", "category_'es_health'", "category_'es_home'",
#    "category_'es_hotelservices'", "category_'es_hyper'", "category_'es_leisure'", "category_'es_otherservices'", "category_'es_sportsandtoys'",
#    "category_'es_tech'", "category_'es_transportation'", "category_'es_travel'", "category_'es_wellnessandbeauty'"
ages = [1, 2, 3]
genders = ['M', 'F']
amounts = [1, 2, 3]
categories = ["'es_transportation'", "'es_food'", "'es_wellnessandbeauty'"]

raw = pd.read_csv(datasetUrl[0])

# limitando registros por questões de consumo de CPU do notebook
features = raw[0:3000]
# features = raw[0:50000]

features = features.drop('merchant', axis = 1)
features = features.drop('zipcodeOri', axis = 1)
features = features.drop('zipMerchant', axis = 1)

features = features[features['category'].isin(categories)]

features.sort_index()

customers_fraud = []
# k = 0
# for i in features['customer']:
#     print(k)
#     print(features['fraud'])
#     if(features['fraud'][k] == 1):
#         customers_fraud.insert(k, i)
#     k+=1

# normalização
features = pd.get_dummies(features)

# Separating the labels
labels = features['fraud']
counter = Counter(labels)

print('Before oversampling: ', counter)

# Remove the labels from the features
features = features.drop('fraud', axis = 1)
# Saving feature list for later use
feature_list = list(features.columns)

# SMOTE Oversampling, deixando o dataset com 75%+ de transações não fraudulentas
over = SMOTE(sampling_strategy=0.75)
X, y = over.fit_resample(features, labels)

counter = Counter(y)
print('After oversampling: ', counter, "\n")

# Convert to np array for the regressor
features = np.array(X)
labels   = np.array(y)

# Split the data into training and testing sets
seed = 42
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.3, random_state = seed)

for j in range(len(features[0])):
    ft = feature_list[j]
    if(ft.startswith("category")):
        categories.append(ft)

counter = Counter(test_labels)
print('Amostra de testes: ', counter)
print('N Amostra de testes: ', len(test_labels), "\n")

#clusteriza transações da amostra de treino
print('\n------------- CLUSTERING -----------------\n')

#inicializa listas auxiliares
grouped_train_features = []
grouped_test_features = []
grouped_train_labels = []
grouped_test_labels = []
for g in itertools.product(ages, genders, amounts, categories):
    grouped_train_features.append([])
    grouped_test_features.append([])
    grouped_train_labels.append([])
    grouped_test_labels.append([])

def cluster(features, labels, feature_list, categories, grouped_features, grouped_labels, ages, genders, amounts):   
    for i in range(len(train_features)):
        register = train_features[i]
        label = train_labels[i]
        amount = 3
        category = ''
        age = 1
        gender = 'M'

        for j in range(len(train_features[i])):
            value = train_features[i][j]
            ft = feature_list[j]

            if(ft == 'amount'):
                if(value <= 50):
                    amount = 1
                elif (value <= 200):
                    amount = 2
            elif(ft == "gender_'F'" and value == 1):
                gender = 'F'
            elif((ft == "age_'3'" or ft == "age_'4'") and value == 1):
                age = 2
            elif((ft == "age_'5'" or ft == "age_'6'") and value == 1):
                age = 3
            elif(ft in categories and value == 1):
                category = ft


        current_list = [age, gender, amount, category]
        k = 0
        for g in itertools.product(ages, genders, amounts, categories):
            g1 = [g[0], g[1], g[2], g[3]]
            if(g1 == current_list):
                grouped_features[k].append(register)
                grouped_labels[k].append(label)
                break
            k+=1

cluster(train_features, train_labels, feature_list, categories, grouped_train_features, grouped_train_labels, ages, genders, amounts)
cluster(test_features, test_labels, feature_list, categories, grouped_test_features, grouped_test_labels, ages, genders, amounts)

print('\n------------- END CLUSTERING -----------------\n')

print('\n------------- CALCULATING THRESHOLDS -----------------\n')

grouped_thresholds = []
#treina RandomForest separadamente por grupo para obter as thresholds
for i in range(len(grouped_train_features)):
    if(len(grouped_train_features[i]) == 0):
        grouped_thresholds.insert(i, [0.0])
        continue

    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 750, random_state = seed)
    # Train the model on training data
    rf.fit(grouped_train_features[i], grouped_train_labels[i])
    # Calcular "threshold" do grupo
    pred = rf.predict(grouped_test_features[i])
    
    grouped_thresholds.insert(i, pred)

#Cálculo das thresholds por grupo
final_thresholds = []
for i in range(len(grouped_thresholds)):
    e = grouped_thresholds[i]
    #T=(mean+max)/2+STD
    t = 0
    if(len(e) > 1):
#         t = (stats.mean(e)+e.max())/(2+stats.stdev(e))
        t = stats.mean(e)/(2+stats.stdev(e))
#         t = stats.mean(e)
    elif(len(e) > 0):
        t = e[0]
    
    final_thresholds.insert(i, t)
    

print('\n------------- FINISH CALCULATING THRESHOLDS -----------------\n')
    

print('\n------------- TRAINING RANDOM FOREST REGRESSOR -----------------\n')

new_ft = np.array(test_features)
new_lb = np.array(test_labels)

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = seed)
# Train the model on training data
rf.fit(train_features, train_labels)
# Calcula os scores das novas transações
new_predictions = rf.predict(new_ft)

print('\n------------- FINISH TRAINING REGRESSORS -----------------\n')

print('\n------------- EVALUATING -----------------\n')

# Cálculo da quantidade de acertos e erros e da matriz de confusão
errors = 0
false_positives = 0
false_negatives = 0
true_positives = 0
true_negatives = 0
for i in range(len(new_predictions)):
    fraud = 0
    multiplier = 1
    register = new_ft[i]
    label = new_lb[i]
    amount = 3
    category = ''
    age = 1
    gender = 'M'

    for j in range(len(new_ft[i])):
        value = new_ft[i][j]
        ft = feature_list[j]

        if(ft == 'amount'):
            if(value <= 50):
                amount = 1
            elif (value <= 200):
                amount = 2
        elif(ft == "gender_'F'" and value == 1):
            gender = 'F'
        elif((ft == "age_'3'" or ft == "age_'4'") and value == 1):
            age = 2
        elif((ft == "age_'5'" or ft == "age_'6'") and value == 1):
            age = 3
        elif(ft in categories and value == 1):
            category = ft
        elif(ft.startswith('category') and ft not in categories):
            print(ft)
        elif(ft.startswith('customer') and ft in customers_fraud and value == 1):
            multiplier = 1.25


    current_list = [age, gender, amount, category]

    index = -1
    k = 0 
    for g in itertools.product(ages, genders, amounts, categories):
        g1 = [g[0], g[1], g[2], g[3]]
        if(g1 == current_list):
            index = k
            break
        k+=1

    if(index < 0):
        continue

    threshold = final_thresholds[index]*multiplier
    
    if(new_predictions[i] > threshold):
        fraud = 1

    cl = new_lb[i]
    if(fraud != cl):
        errors += 1
        
    if(fraud == 1 and cl == 1):
        true_positives += 1
        
    if(fraud == 0 and cl == 0):
        true_negatives += 1
        
    if(fraud == 0 and cl == 1):
        false_positives += 1
        
    if(fraud == 1 and cl == 0):
        false_negatives += 1
        
print('\n------------- FINISH EVALUATING -----------------\n')

print('\n---------------- CONFUSION MATRIX ----------------\n')

print('Total: ', len(new_predictions), "\n")
print('Errors: ', errors)
print('True positives: ', true_positives)
print('True negatives: ', true_negatives)
print('False negatives: ', false_negatives)
print('False positives: ', false_positives)

#Accuracy = TP+TN/TP+FP+FN+TN
#Precision = TP/TP+FP
#Recall = TP/TP+FN
#F1 Score = 2*(Recall * Precision) / (Recall + Precision)

accuracy = (true_positives+true_negatives)/(true_positives+true_negatives+false_positives+false_negatives)
precision = true_positives/(true_positives+false_positives)
recall = true_positives/(true_positives+false_negatives)
f1 = (2*(recall*precision)) / (recall+precision)

print('\n----------------- METRICS ------------------------\n')

print('Acurácia: ', accuracy)
print('Precisão: ', precision)
print('Recall: ', recall)
print('F1 Score: ', f1)
