#!/usr/bin/env python
# coding: utf-8

# # Contexto
# No Airbnb, qualquer pessoa que tenha um quarto ou um imóvel de qualquer tipo (apartamento, casa, chalé, pousada, etc.) pode ofertar o seu imóvel para ser alugado por diária.
# 
# Você cria o seu perfil de host (pessoa que disponibiliza um imóvel para aluguel por diária) e cria o anúncio do seu imóvel.
# 
# Nesse anúncio, o host deve descrever as características do imóvel da forma mais completa possível, de forma a ajudar os locadores/viajantes a escolherem o melhor imóvel para eles (e de forma a tornar o seu anúncio mais atrativo)
# 
# Existem dezenas de personalizações possíveis no seu anúncio, desde quantidade mínima de diária, preço, quantidade de quartos, até regras de cancelamento, taxa extra para hóspedes extras, exigência de verificação de identidade do locador, etc.

# # Nosso objetivo
# Construir um modelo de previsão de preço que permita uma pessoa comum que possui um imóvel possa saber quanto deve cobrar pela diária do seu imóvel.
# 
# Ou ainda, para o locador comum, dado o imóvel que ele está buscando, ajudar a saber se aquele imóvel está com preço atrativo (abaixo da média para imóveis com as mesmas características) ou não.

# # O que temos disponível, inspirações e créditos
# As bases de dados foram retiradas do site kaggle: https://www.kaggle.com/allanbruno/airbnb-rio-de-janeiro
# -  As bases de dados são os preços dos imóveis obtidos e suas respectivas características em cada mês.
# - Os preços são dados em reais (R$)
# - Temos bases de abril de 2018 a maio de 2020, com exceção de junho de 2018 que não possui base de dados

# # Importando bibliotecas e a base de dados

# In[7]:


import pandas as pd
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split


# In[8]:


#Importando a base de dados
meses = {'jan':1, 'fev':2, 'mar':3, 'abr':4, 'mai':5, 'jun':6, 'jul':7, 'ago':8, 'set':9, 'out':10, 'nov':11, 'dez':12}
caminho_bases = pathlib.Path('dataset_')
base_airbnb = pd.DataFrame()

for arquivo in caminho_bases.iterdir():
    nome_mes = arquivo.name[:3]
    mes = meses[nome_mes]
    
    ano = arquivo.name[-8:]
    ano = int(ano.replace('.csv', ''))
    
    df = pd.read_csv(caminho_bases / arquivo.name, low_memory=False)
    df['ano'] = ano
    df['mes'] = mes 
    base_airbnb = base_airbnb.append(df)
   
display(base_airbnb)
    
    


# #  Começando os tratamentos
# Como o dataset tem muitas colunas, o modelo pode acabar ficando muito lento.
# 
# Além disso, uma análise rápida permite ver que várias colunas não são necessárias para o nosso modelo de previsão, por isso, irei excluir algumas colunas da base
# 
# Tipos de colunas que será excluído:
# 
# IDs, Links e informações não relevantes para o modelo
# Colunas repetidas ou extremamente parecidas com outra (que dão a mesma informação para o modelo. Ex: Data x Ano/Mês
# Colunas preenchidas com texto livre -> Não rodaremos nenhuma análise de palavras ou algo do tipo
# Colunas em que todos ou quase todos os valores são iguais
# Para isso, vou criar um arquivo em excel com os 1.000 primeiros registros e fazer uma análise qualitativa, olhando as colunas e identificando quais são desnecessárias

# In[3]:


print(list(base_airbnb.columns))
base_airbnb.head(1000).to_csv('primeiros_registros.csv', sep=';')


# Após a análise qualitativa, eliminando as colunas com de acordo com os critérios citados acima, restaram as seguintes colunas:

# In[9]:


colunas = ['host_response_time','host_response_rate','host_is_superhost','host_listings_count','latitude','longitude','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','amenities','price','security_deposit','cleaning_fee','guests_included','extra_people','minimum_nights','maximum_nights','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','instant_bookable','is_business_travel_ready','cancellation_policy','ano','mes']

base_airbnb = base_airbnb.loc[:, colunas]
print(list(base_airbnb.columns))
display(base_airbnb)


# # Tratar colunas que estão vazias
# - Visualizando os dados, percebe-se que existe uma grande disparidade em dados faltantes. As colunas com mais de 300.000 valores NaN serão excluídas da análise

# In[10]:


for coluna in base_airbnb:
    if base_airbnb[coluna].isnull().sum() >= 100000:
        base_airbnb = base_airbnb.drop(coluna, axis=1)
print(base_airbnb.isnull().sum())
        


# In[11]:


# Como ainda possui valores vazios, removerei as linhas que contêm valores nulos
base_airbnb = base_airbnb.dropna()
print(base_airbnb.isnull().sum())


# In[6]:


# Verificando os tipos de dados
print(base_airbnb.dtypes)
print('-'*60)
print(base_airbnb.iloc[0])


# In[12]:


#Alterando o tipo de dado da coluna "price" e "extra_people"
# price
base_airbnb['price'] = base_airbnb['price'].str.replace('$', '')
base_airbnb['price'] = base_airbnb['price'].str.replace(',', '')
base_airbnb['price'] = base_airbnb['price'].astype(np.float32, copy=False)
# extra people
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace('$', '')
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace(',', '')
base_airbnb['extra_people'] = base_airbnb['extra_people'].astype(np.float32, copy=False)

#verificando os tipo
print(base_airbnb.dtypes)


# # Verificando a existência de outliers
# Vou basicamente olhar feature por feature para:
# 
# - Ver a correlação entre as features e decidir se manteremos todas as features que temos.
# - Excluir outliers (usarei como regra, valores abaixo de Q1 - 1.5xAmplitude e valores acima de Q3 + 1.5x Amplitude). Amplitude = Q3 - Q1
# - Confirmar se todas as features que temos fazem realmente sentido para o nosso modelo ou se alguma delas não vai nos ajudar e se devemos excluir
# 
# 
# Vou começar pelas colunas de preço (resultado final que queremos) e de extra_people (também valor monetário). Esses são os valores numéricos contínuos.
# 
# Depois irei analisar as colunas de valores numéricos discretos (accomodates, bedrooms, guests_included, etc.)
# 
# Por fim, irei avaliar as colunas de texto e definir quais categorias fazem sentido manter ou não.
# 

# In[9]:


plt.figure(figsize=(15,10))
sns.heatmap(base_airbnb.corr(), annot=True, cmap='Greens')


# # Definição de Funções para Análise de Outliers
# Irei definir algumas funções para ajudar na análise de outliers das colunas

# In[13]:


def limite(coluna):
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = q3 - q1
    return q1 - 1.5 * amplitude, q3 + 1.5 * amplitude
def excluir_outliers(df, nome_coluna):
    qtde_linhas = df.shape[0]
    lim_inf, lim_sup = limite(df[nome_coluna])
    df = df.loc[(df[nome_coluna] >= lim_inf) & (df[nome_coluna] <= lim_sup), :]
    linhas_removidas = qtde_linhas - df.shape[0]
    return df,  linhas_removidas


# In[14]:


def diagrama_caixa(coluna):
    fig, (ax1, ax2) = plt.subplots(1, 2) 
    fig.set_size_inches(15, 5)
    
    sns.boxplot(x=coluna, ax=ax1) 
    ax2.set_xlim(limite(coluna))
    sns.boxplot(x=coluna, ax=ax2)

def histograma(coluna):
    plt.figure(figsize=(15,5))
    sns.distplot(coluna, hist=True)
    
def grafico_de_barras(coluna):
    plt.figure(figsize=(15,5))
    ax = sns.barplot(x=coluna.value_counts().index, y=coluna.value_counts())
    ax.set_xlim(limite(coluna))
    


# # Price

# In[12]:


diagrama_caixa(base_airbnb['price'])
histograma(base_airbnb['price'])


# Como estamos construindo um modelo para imóveis comuns, acredito que os valores acima do limite superior serão apenas de apartamentos de altíssimo luxo, que não é o nosso objetivo principal. Por isso, irei excluir esses outliers.

# In[15]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'price')
print('{}, linhas removidas'.format(linhas_removidas)) 


# In[14]:


histograma(base_airbnb['price'])


# # Extra People

# In[15]:


diagrama_caixa(base_airbnb['extra_people'])
histograma(base_airbnb['extra_people'])


# In[16]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb,'extra_people')
print('{}, linhas removidas'.format(linhas_removidas)) 


# # Host listings count

# In[19]:


diagrama_caixa(base_airbnb['host_listings_count'])
grafico_de_barras(base_airbnb['host_listings_count'])


# Irei excluir os outliers, pois para o objetivo do  projeto, hosts com mais de 6 imóveis no airbnb não é o público alvo do objetivo do projeto (imagino que sejam imobiliários ou profissionais que gerenciam imóveis no airbnb)

# In[17]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'host_listings_count')
print('{}, linhas removidas'.format(linhas_removidas))


# # Accommodates

# In[21]:


diagrama_caixa(base_airbnb['accommodates'])
grafico_de_barras(base_airbnb['accommodates'])


# - Como o meu objetivo é focar em imóveis comuns, irei remover esses outliers, pois imóveis que acomodam mais de 9 pessoas, são de altíssimo nível
# - Caso fosse construir um modelo que também focasse em imóveis de altíssimo nível, seria necessário construir um outro modelo contendo esses outliers

# In[18]:


base_aibrnb, linhas_removidas = excluir_outliers(base_airbnb, 'accommodates')
print('{}, linhas removidas'.format(linhas_removidas))


# # Bathrooms

# In[23]:


diagrama_caixa(base_airbnb['bathrooms'])
plt.figure(figsize=(15, 5))
sns.barplot(x=base_airbnb['bathrooms'].value_counts().index, y=base_airbnb['bathrooms'].value_counts())


# Irei excluir os outliers dos banheiros pelo mesmo motivo anterior

# In[19]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bathrooms')
print('{}, linhas removidas'.format(linhas_removidas))


# # bedrooms

# In[25]:


diagrama_caixa(base_airbnb['bedrooms'])
plt.figure(figsize=(15,5))
sns.barplot(x=base_airbnb['bedrooms'].value_counts().index, y=base_airbnb['bedrooms'].value_counts())


# Também se enquadra nos mesmos requisitos citados acima para remover os outliers

# In[20]:


base_airbnb, remover_linhas = excluir_outliers(base_airbnb, 'bedrooms')
print('{}, linhas removidas'.format(linhas_removidas))


# # Beds

# In[27]:


diagrama_caixa(base_airbnb['beds'])
grafico_de_barras(base_airbnb['beds'])


# # Também se enquadra nos mesmos requisitos citados acima para remover os outliers

# In[21]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'beds')
print('{}, linhas removidas'.format(linhas_removidas))


# # Guests_included

# In[29]:


plt.figure(figsize=(15, 5))
sns.barplot(x=base_airbnb['guests_included'].value_counts().index, y=base_airbnb['guests_included'].value_counts())


# Irei remover essa feature da analise pois, me parece que a Airbnb usa na maioria das vezes o valor 1 como padrão para o guest included. Então para evitar que o modelo considere uma feature que não seja essencial para a construção do preço, talvez excluir essa feature da análise seja o melhor a se fazer 

# In[22]:


base_airbnb = base_airbnb.drop('guests_included', axis=1)


# # Minimum nights

# In[31]:


diagrama_caixa(base_airbnb['minimum_nights'])
grafico_de_barras(base_airbnb['minimum_nights'])


# Aqui tem um motivo talvez até mais forte para excluir esses outliers da análise.
# 
# Estou querendo um modelo que ajude a precificar apartamentos comuns como uma pessoa comum gostaria de disponibilizar. No caso, apartamentos com mais de 8 noites como o "mínimo de noites" podem ser apartamentos de temporada ou ainda apartamentos para morar, em que o host exige pelo menos 1 mês no apartamento.
# 
# Por isso, vou excluir os outliers dessa coluna

# In[23]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'minimum_nights')
print('{}, linhas removidas'.format(linhas_removidas))


# # Maximum nights

# In[33]:


diagrama_caixa(base_airbnb['maximum_nights'])


# Essa coluna não parece que vai ajudar na análise.
# 
# Isso porque parece que quase todos os hosts não preenchem esse campo de maximum nights, então ele não parece que vai ser um fator relevante.

# In[24]:


base_airbnb = base_airbnb.drop('maximum_nights', axis=1)
base_airbnb.shape


# # Number of Reviews

# In[35]:


diagrama_caixa(base_airbnb['number_of_reviews'])
grafico_de_barras(base_airbnb['number_of_reviews'])


# Irei remover esta coluna por uma conclusão que cheguei:
#    -  Não acho que o número de reviews possa impactar no preço de uma casa
#    - Acredito também que esta feature pode prejudicar pessoas novatas no site, já que elas provavelmente não terão tantos reviews quanto as grandes mobiliarias, podendo ter seus preços impactados.
#    - Visando tudo isso, resolvi remover essa feature da análise
# 

# In[25]:


base_airbnb = base_airbnb.drop('number_of_reviews', axis=1)
base_airbnb.shape


# # Tratamento de Colunas do Tipo de Texto

# # Property Type

# In[26]:


print(base_airbnb['property_type'].value_counts())

plt.figure(figsize=(15,5))
grafico = sns.countplot('property_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# - Aqui eu irei colocar os valores que não possuem muita frequência, e junta-los todos em uma mesma coluna de nome 'outros'.
# - Todos os tipos de propriedades que possuem menos frequência na análise menor que 2000, eu irei juntar na nova coluna

# In[27]:


tabela_tipos_casa = base_airbnb['property_type'].value_counts()
colunas_agrupar = []

for tipo in tabela_tipos_casa.index:
    if tabela_tipos_casa[tipo] < 2140:
        colunas_agrupar.append(tipo)
print(colunas_agrupar)

for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['property_type']==tipo, 'property_type'] = 'Outros'

print(base_airbnb['property_type'].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot('property_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# # Room Type

# In[28]:


print(base_airbnb['room_type'].value_counts())

plt.figure(figsize=(15, 5))
grafico = sns.countplot('room_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# Por enquanto, esses dados aparentam estar bem distribuídos, portanto, deixarei como está.

# # Bed Type

# In[29]:


print(base_airbnb['bed_type'].value_counts())

# Agrupando as categorias de bed type
tabela_bed = base_airbnb['bed_type'].value_counts()
agrupar_colunas = []

for tipo in tabela_bed.index:
    if tabela_bed[tipo] < 10000:
        agrupar_colunas.append(tipo)
print(agrupar_colunas)

for tipo in agrupar_colunas:
    base_airbnb.loc[base_airbnb['bed_type']==tipo, 'bed_type'] = 'Outros'
    
print(base_airbnb['bed_type'].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot('bed_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# # Cancellation policy

# In[30]:


print(base_airbnb['cancellation_policy'].value_counts())
plt.figure(figsize=(15,5))
grafico = sns.countplot('cancellation_policy', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)

#Agrupando categoria de cancellation policy
tabela_cancellation = base_airbnb['cancellation_policy'].value_counts()
agrupar_colunas = []

for tipo in tabela_cancellation.index:
    if tabela_cancellation[tipo] < 11000:
        agrupar_colunas.append(tipo)
print(agrupar_colunas)

for tipo in agrupar_colunas:
    base_airbnb.loc[base_airbnb['cancellation_policy']==tipo, 'cancellation_policy'] = 'strict'

print(base_airbnb['cancellation_policy'].value_counts())
plt.figure(figsize=(15,5))
grafico = sns.countplot('cancellation_policy', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# # Amenities
# Como no dataset a coluna amenities possui vários valores diferentes, resolvi avaliar a quantidade de amenities como parâmetro para o modelo

# In[31]:


print(base_airbnb['amenities'].iloc[1].split(','))
print(len(base_airbnb['amenities'].iloc[1].split(',')))

base_airbnb['n_amenities'] = base_airbnb['amenities'].str.split(',').apply(len)


# In[32]:


base_airbnb = base_airbnb.drop('amenities', axis=1)
base_airbnb.shape


# In[64]:


diagrama_caixa(base_airbnb['n_amenities'])
grafico_de_barras(base_airbnb['n_amenities'])


# Feito isso, a coluna virou uma coluna numérica, portanto irei excluir esses outliers assim como fiz com as demais colunas de valores numéricos

# In[33]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb,'n_amenities')
print('{}, linhas removidas'.format(linhas_removidas))


# # Visualização de Mapa das Propriedades
# Irei construir um mapa com uma parcela dos dados (50 000 linhas) para vizualizar a relação entre preço e localização  das casas distribuidas no dataset 

# In[71]:


amostra = base_airbnb.sample(n=50000)
centro_mapa = {'lat':amostra.latitude.mean(), 'lon':amostra.longitude.mean()}
mapa = px.density_mapbox(amostra, lat='latitude', lon='longitude', z='price', radius = 2.5,
                        center = centro_mapa, zoom = 10,
                        mapbox_style='stamen-terrain')
mapa.show()


# # Encoding
# 
# Agora, para ajustar para o modelo, irei ajustar as colunas que possuem valores "True" e "False" e features de categoria(vamos utilizar o método de encoding de variáveis dummies)

# In[34]:


colunas_tf = ['host_is_superhost', 'instant_bookable', 'is_business_travel_ready']
base_airbnb_cod = base_airbnb.copy()
for coluna in colunas_tf:
    base_airbnb_cod.loc[base_airbnb_cod[coluna]=='t', coluna] = 1
    base_airbnb_cod.loc[base_airbnb_cod[coluna]=='f', coluna] = 0
    
    


# In[35]:


colunas_categorias = ['property_type', 'room_type', 'bed_type', 'cancellation_policy']
base_airbnb_cod = pd.get_dummies(data=base_airbnb_cod, columns=colunas_categorias) 
display(base_airbnb_cod.head())


# #  Modelos de previsão
# - Irei usar aqui o R² para dizer o quão bem o modelo consegue explicar o preço. 
# - Vou calcular também o Erro Quadrático Médio, que vai mostrar para gente o quanto o nosso modelo está errando.

# In[36]:


def avaliar_modelo (nome_modelo, y_teste, previsao):
    r2 = r2_score(y_teste, previsao)
    RMSE = np.sqrt(mean_squared_error(y_teste, previsao))
    return f'modelo {nome_modelo}: \nR2:{r2:.2%}\nRMSE:{RMSE:.2f}'
    


# - Modelos que serão testados
#  
#  1- RandomForest
#  
#  2- LinearRegression
#  

# In[33]:


modelo_rf = RandomForestRegressor()
modelo_lr = LinearRegression()


modelos = {'RandomForest': modelo_rf,
          'LinearRegression':modelo_lr,
          }


y = base_airbnb_cod['price']
X = base_airbnb_cod.drop('price', axis=1)


# Separando os dados em treino e teste e o treino dos modelos

# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

for nome_modelo, modelo in modelos.items():
    #treinar
    modelo.fit(X_train, y_train)
    #testar
    previsao = modelo.predict(X_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))


# # Verificando a importância de cada feature

# In[35]:



importancia_features = pd.DataFrame(modelo_rf.feature_importances_, X_train.columns)
importancia_features = importancia_features.sort_values(by=0, ascending=False)
display(importancia_features)
plt.figure(figsize=(15,5))
ax = sns.barplot(x=importancia_features.index, y=importancia_features[0])
ax.tick_params(axis='x', rotation=90)


# # Ajustes finais
# De acordo com a análise que fiz acima, percebe-se que a coluna "is_business_travel_ready não possui impacto algum na realização do modelo, por isso a removerei do modelo 

# In[38]:


base_airbnb_cod = base_airbnb_cod.drop(columns=['is_business_travel_ready', 'cancellation_policy_strict', 'room_type_Hotel room'])
y = base_airbnb_cod['price']
X = base_airbnb_cod.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

modelo_rf.fit(X_train, y_train)
previsao = modelo_rf.predict(X_test)
print(avaliar_modelo('RandomForest', y_test, previsao))


# In[41]:


base_teste = base_airbnb_cod.copy()
for coluna in base_teste:
    if 'bed_type' in coluna:    
        base_teste = base_teste.drop(coluna, axis=1)
print(base_teste.columns)
y = base_teste['price']
X = base_teste.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

modelo_rf.fit(X_train, y_train)
previsao = modelo_rf.predict(X_test)
print(avaliar_modelo('RandomForest', y_test, previsao))


# In[40]:


print(previsao)


# # Persistindo o modelo para o disco
# 
# 
# 

# In[1]:


from sklearn.externals import joblib 


# In[43]:


joblib.dump(modelo_rf, 'previsao.pkl')


# Carregando o modelo a partir da memória

# In[2]:


modelo = joblib.load('previsao.pkl')

