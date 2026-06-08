import pickle
import pandas as pd

#Carregar modelo e features
modelo = pickle.load(open('melhor_modelo_default.pkl', 'rb'))
colunas_treinamento = modelo.feature_names_in_

molde_df = pd.DataFrame(columns=colunas_treinamento)

#Dados do novo cliente para teste
colunas_originais = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

novo_cliente = pd.DataFrame([[
    50000, 'F', 'Middle School ', 'Married', 57, 
    -1, 0, -1, 0, 0, 0, 
    8617, 5670, 35835, 20940, 19146, 19131, 
    2000, 36681, 10000, 9000, 689, 679
]], columns=colunas_originais)

#Tratar dados de entrada
novo_cliente_tratado = pd.get_dummies(novo_cliente)

#Concatena com o molde para garantir mesma estrutura de colunas e preenche nulos
novo_cliente_final = pd.concat([molde_df, novo_cliente_tratado]).fillna(0)
novo_cliente_final = novo_cliente_final[colunas_treinamento].astype(float)

#Executa inferencia
predicao = modelo.predict(novo_cliente_final)
probabilidades = modelo.predict_proba(novo_cliente_final)

score_adimplencia = probabilidades[0][0] * 100
score_inadimplencia = probabilidades[0][1] * 100

print('\n--- Resultado da Inferencia ---')
if predicao[0] == 1:
    print('Classificacao: Inadimplente')
else:
    print('Classificacao: Adimplente')

print(f'Score de Inadimplencia (Default): {score_inadimplencia:.2f}%')
print(f'Score de Adimplencia: {score_adimplencia:.2f}%')