import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from pickle import dump
from imblearn.over_sampling import SMOTE
from collections import Counter

#Carrega e limpa os dados
dados = pd.read_csv('default_of_credit_card_clients.csv', sep=';')
dados.columns = dados.columns.str.strip()

for col in dados.select_dtypes(include=['object', 'string']).columns:
    dados[col] = dados[col].astype(str).str.strip()

if 'ID' in dados.columns:
    dados = dados.drop(columns=['ID'])

coluna_alvo = 'default payment next month'
atributos_originais = dados.drop(columns=[coluna_alvo])
dados_classe = dados[coluna_alvo]

#Dummificacao
colunas_categoricas = ['SEX', 'EDUCATION', 'MARRIAGE']
dados_atributos = pd.get_dummies(atributos_originais, columns=colunas_categoricas, drop_first=True)
dados_atributos = dados_atributos.astype(float)

#Balanceamento
resampler = SMOTE(random_state=42)
atributos_b, classes_b = resampler.fit_resample(dados_atributos, dados_classe)
print('Distribuicao apos SMOTE:', Counter(classes_b))

#Divisao treino/teste
atributos_train, atributos_teste, classe_train, classe_test = train_test_split(
    atributos_b, classes_b, test_size=0.3, random_state=42
)

#Hiperparametrizacao
rf_grid = {
    'n_estimators': [int(x) for x in np.linspace(10, 100, 10)],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 5, 10],
    'max_depth': [int(x) for x in np.linspace(10, 50, 5)],
    'max_features': ['sqrt', 'log2']
}

print('\nBuscando hiperparametros para RF')
rf_base = RandomForestClassifier(random_state=42)
rf_hyper = RandomizedSearchCV(estimator=rf_base, param_distributions=rf_grid, n_iter=10, cv=3, n_jobs=-1, random_state=42)
rf_hyper.fit(atributos_train, classe_train)

print('Melhores parametros:', rf_hyper.best_params_)

#Treinamento dos modelos finais
model_tree = DecisionTreeClassifier(random_state=42)
model_tree.fit(atributos_train, classe_train)
model_rf = rf_hyper.best_estimator_

#Salva o melhor modelo
dump(model_rf, open('melhor_modelo_default.pkl', 'wb'))

#Avaliacao
def avaliar(modelo, nome):
    predito = modelo.predict(atributos_teste)
    tn, fp, fn, tp = confusion_matrix(classe_test, predito).ravel()
    
    print(f'\n--- {nome} ---')
    print(f'Acuracia: {accuracy_score(classe_test, predito):.4f}')
    print(f'Especificidade: {tn / (tn + fp):.4f}')
    print(f'Sensibilidade: {tp / (tp + fn):.4f}')

avaliar(model_tree, 'Decision Tree')
avaliar(model_rf, 'Random Forest')