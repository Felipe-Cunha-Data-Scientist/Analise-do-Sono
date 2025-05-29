# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 08:35:24 2025

@author: User
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression

# Carregar os dados
df = pd.read_csv(r"C:\Users\User\OneDrive - Instituto Presbiteriano Mackenzie\Mackenzie\Aulas\5 semestre\Projeto Aplicado\Aplicando conhecimento\dataset\sleep_cycle_productivity.csv")

# Tendencia e Sazonalidade
# Exibir informações gerais
print(df.info())
df.head



# Série Temporal ou Agrupamento por idade
df_group = df.groupby('Age')['Sleep Quality'].mean()

plt.plot(df_group)
plt.title('Qualidade do Sono Média por Idade')
plt.xlabel('Idade')
plt.ylabel('Qualidade do Sono')
plt.show()

# Decomposição da série (se tiver variável temporal)
result = seasonal_decompose(df['Sleep Quality'], model='additive', period=7)  # ou period=12 ou conforme o ciclo esperado
result.plot()


# Variavel temporal, plotar as medias das horas de sono ao longo do tempo para ver tendencia
# Convertendo a coluna de data
df['Date'] = pd.to_datetime(df['Date'])

# Agrupando por data
media_sono = df.groupby('Date')['Total Sleep Hours'].mean()

# Plotando tendência
plt.figure(figsize=(12,6))
media_sono.plot()
plt.title('Tendência de Horas de Sono ao Longo do Tempo')
plt.xlabel('Data')
plt.ylabel('Horas de Sono')
plt.grid(True)
plt.show()


# Criar colunas auxiliares para analise de tendencia e sazonalidade

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.day_name()


# Verificando Sazonalidade
# Extraindo mês e dia da semana
# Supondo que você tenha a coluna Sleep_Quality
df_monthly = df.groupby(['Year', 'Month'])['Sleep Quality'].mean().reset_index()
df_monthly.head()



# Converter para datetime index
df_monthly['Date'] = pd.to_datetime(df_monthly[['Year', 'Month']].assign(Day=1))
df_monthly.set_index('Date', inplace=True)

#Visualização Tendencia e sazonalidade agrupada


plt.figure(figsize=(12, 6))
plt.plot(df_monthly.index, df_monthly['Sleep Quality'], marker='o')
plt.title('Tendência e Sazonalidade - Qualidade do Sono')
plt.xlabel('Ano')
plt.ylabel('Média da Qualidade do Sono')
plt.grid(True)
plt.show()

#Verificação automatica de tendencia e sazonalidade
result = seasonal_decompose(df_monthly['Sleep Quality'], model='additive', period=6)
result.plot()

from statsmodels.tsa.seasonal import seasonal_decompose

# Aplicar o método de decomposição
result = seasonal_decompose(df_monthly['Sleep Quality'], model='additive', period=6)

# Converter os componentes extraídos em um dataframe
df_decomposed = pd.DataFrame({
    'Tendência': result.trend,
    'Sazonalidade': result.seasonal,
    'Resíduos': result.resid,
    'Original': df_monthly['Sleep Quality']
})

# Exibir as primeiras linhas
print(df_decomposed.head())


######Modelo ARIMA

# Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt


# Ajustar índice
df = df.sort_values('Date')
df = df.set_index('Date')

# Exibir amostra
print(df.head(8))
print (df.info)
print (df.columns)


# Criar o gráfico
plt.figure(figsize=(12,6))
sns.lineplot(data=df, x=df.index, y='Sleep Quality', label='Qualidade do Sono')

# Adicionar título e legendas
plt.title('Qualidade do Sono ao Longo do Tempo', fontsize=16)
plt.xlabel('Data', fontsize=12)
plt.ylabel('Qualidade do Sono (%)', fontsize=12)
plt.legend(title='Legenda', loc='upper left')

# Exibir o gráfico
plt.tight_layout()


# Teste de Estacionariedade (Dickey-Fuller)
resultado = adfuller(df['Sleep Quality'].dropna())
print('ADF Statistic:', resultado[0])
print('p-value:', resultado[1])


# Treinamento do Modelo ARIMA(p,d,q)
# Exemplo com (1,1,1) → Ideal ajustar depois com auto_arima ou grid search
modelo = ARIMA(df['Sleep Quality'], order=(5,1,0))
resultado_modelo = modelo.fit()

# Resumo
print(resultado_modelo.summary())




# Previsão
df.index = pd.to_datetime(df.index)
forecast = resultado_modelo.get_forecast(steps=30)
forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
forecast_df = forecast.summary_frame()

# Plotar o histórico e a previsão
plt.figure(figsize=(14,7))
plt.plot(df['Sleep Quality'], label='Histórico', color='blue', linewidth=2)
plt.plot(forecast_index, forecast_df['mean'], label='Previsão', color='red', linestyle='--', marker='o')
plt.fill_between(forecast_index, forecast_df['mean_ci_lower'], forecast_df['mean_ci_upper'], color='pink', alpha=0.2, label='Intervalo de Confiança (95%)')
plt.legend(loc='upper left', fontsize=12)
plt.title('Previsão do Modelo ARIMA', fontsize=16)
plt.xlabel('Data', fontsize=14)
plt.ylabel('Qualidade do Sono', fontsize=14)
plt.grid(alpha=0.5)
plt.show()  # Certifique-se de que esta linha está presente



# Grid search do Arima, procurando melhores hyper parametros

from pmdarima import auto_arima

# Executar o auto_arima
stepwise_fit = auto_arima(df['Sleep Quality'],
                          start_p=0, max_p=5,  # Variação de p
                          start_q=0, max_q=5,  # Variação de q
                          d=1,                # Diferença (fixa ou testável)
                          seasonal=False,     # Sem sazonalidade
                          trace=True,         # Exibir o progresso
                          error_action='ignore',  # Ignorar erros
                          suppress_warnings=True,
                          stepwise=True)      # Busca em grade mais rápida

# Exibir os melhores parâmetros
print(stepwise_fit.summary())


#### Usando LSTM 


from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

def create_lagged_data(series, timesteps):
    X, y = [], []
    for i in range(len(series) - timesteps):
        X.append(series[i:i+timesteps])  # Cria janelas de tamanho "timesteps"
        y.append(series[i+timesteps])   # Valor a ser previsto
    return np.array(X), np.array(y)

# Definir número de timesteps
timesteps = 10

# Reformatar os dados
X, y = create_lagged_data(df['Sleep Quality'].values, timesteps)

# Dividir os dados em treino e teste
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Verificar forma dos dados
print(f"Forma de X_train: {X_train.shape}")  # Exemplo: (n_samples, timesteps, 1)

# Definir número de neurônios na LSTM
neurons = 50

# Criar modelo LSTM
model = Sequential()
model.add(LSTM(neurons, activation='relu', input_shape=(timesteps, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Ajustar o modelo
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Fazer previsões
predictions = model.predict(X_test)

# Visualizar as previsões
plt.figure(figsize=(12,6))
plt.plot(y_test, label='Valores Reais')
plt.plot(predictions, label='Previsões')
plt.legend()
plt.title('Comparação entre Valores Reais e Previsões')
plt.show()


# Grid search do LSTM, procurando melhores hyper parametros

from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from sklearn.metrics import mean_squared_error

# Função para criar o modelo LSTM
def create_model(neurons, learning_rate):
    model = Sequential()
    model.add(LSTM(neurons, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Parâmetros para testar
neurons_list = [50, 100]
epochs_list = [10, 50]
batch_size_list = [16, 32]

# Testar combinações de hiperparâmetros
for neurons in neurons_list:
    for epochs in epochs_list:
        for batch_size in batch_size_list:
            print(f"Testando: Neurons={neurons}, Epochs={epochs}, Batch Size={batch_size}")
            model = create_model(neurons=neurons, learning_rate=0.01)
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            
            # Avaliar desempenho no teste
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            print(f"MSE: {mse}")
 


###verificando residuos

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Resíduos
residuos = forecast_df
residuos_lstm = y_test - predictions

from sklearn.metrics import mean_absolute_error
mae_arima = mean_absolute_error(y_test[:30], forecast_df['mean'])
mae_lstm = mean_absolute_error(y_test, predictions)
print(f"MAE ARIMA: {mae_arima}, MAE LSTM: {mae_lstm}")

from sklearn.metrics import mean_squared_error
mse_arima = mean_squared_error(y_test[:30], forecast_df['mean'])
mse_lstm = mean_squared_error(y_test, predictions)
print(f"MSE ARIMA: {mse_arima}, MSE LSTM: {mse_lstm}")


from sklearn.metrics import r2_score
r2_arima = r2_score(y_test[:30], forecast_df['mean'])
r2_lstm = r2_score(y_test, predictions)
print(f"R² ARIMA: {r2_arima}, R² LSTM: {r2_lstm}")

residuos_lstm = y_test - predictions


import matplotlib.pyplot as plt
import pandas as pd

# Configurar as datas para o histórico (2024) e previsões (2025)
historical_dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
forecast_dates = pd.date_range(start="2025-01-01", end="2025-12-31", freq="M")  # Previsões mensais

# Ajustar dimensões dos dados
forecast_df = forecast_df.iloc[:len(forecast_dates)]  # Garantir que ARIMA tem 12 pontos
predictions = predictions.flatten()[:len(forecast_dates)]  # Garantir que LSTM tem 12 pontos

# Gráfico para comparar histórico e previsões
plt.figure(figsize=(14,7))

# Histórico em azul (linha)
plt.plot(historical_dates, df['Sleep Quality'].iloc[-len(historical_dates):], 
         label='Histórico (Qualidade do Sono)', color='blue', linewidth=2)

# Previsão ARIMA em vermelho (linha)
plt.plot(forecast_dates, forecast_df['mean'], 
         label='Previsão ARIMA', color='red', linestyle='--')

# Intervalo de confiança do ARIMA em rosa
plt.fill_between(forecast_dates, forecast_df['mean_ci_lower'], forecast_df['mean_ci_upper'], 
                 color='pink', alpha=0.3, label='Intervalo de Confiança ARIMA')

# Previsão LSTM em verde (linha)
plt.plot(forecast_dates, predictions, 
         label='Previsão LSTM', color='green', linestyle='-.')

# Customizações do gráfico
plt.title('Comparação entre Histórico e Previsões (ARIMA vs LSTM)', fontsize=16, weight='bold')
plt.xlabel('Data', fontsize=14)
plt.ylabel('Qualidade do Sono', fontsize=14)
plt.legend(loc='upper left', fontsize=12, frameon=True, shadow=False, borderpad=1)
plt.grid(alpha=0.3)

# Melhorar visualização do eixo x
plt.xticks(rotation=45, fontsize=10)

# Exibir o gráfico
plt.show()


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# ===================== ARIMA =====================
# Obter a coluna 'mean' das previsões ARIMA
forecast_arima = forecast.summary_frame()['mean']

# Ajustar o tamanho de y_test para comparar com as previsões ARIMA
y_test_adjusted_arima = y_test[:len(forecast_arima)]

# Tratar possíveis NaN
forecast_arima = forecast_arima.fillna(0)
y_test_adjusted_arima = np.nan_to_num(y_test_adjusted_arima, nan=0)

# Calcular os resíduos para ARIMA
residuos_arima = y_test_adjusted_arima - forecast_arima

# Calcular as métricas para ARIMA
mae_arima = mean_absolute_error(y_test_adjusted_arima, forecast_arima)
mse_arima = mean_squared_error(y_test_adjusted_arima, forecast_arima)
rmse_arima = mse_arima ** 0.5
mape_arima = (abs(residuos_arima / y_test_adjusted_arima).mean()) * 100

# ===================== LSTM =====================
# Supondo que 'predictions' seja a saída do modelo LSTM (ex.: predictions.flatten())
# Ajustar o tamanho de y_test para comparar com as previsões LSTM
forecast_lstm = predictions.flatten()
y_test_adjusted_lstm = y_test[:len(forecast_lstm)]

# Tratar possíveis NaN
forecast_lstm = np.nan_to_num(forecast_lstm, nan=0)
y_test_adjusted_lstm = np.nan_to_num(y_test_adjusted_lstm, nan=0)

# Calcular os resíduos para LSTM
residuos_lstm = y_test_adjusted_lstm - forecast_lstm

# Calcular as métricas para LSTM
mae_lstm = mean_absolute_error(y_test_adjusted_lstm, forecast_lstm)
mse_lstm = mean_squared_error(y_test_adjusted_lstm, forecast_lstm)
rmse_lstm = mse_lstm ** 0.5
mape_lstm = (abs(residuos_lstm / y_test_adjusted_lstm).mean()) * 100

# ===================== Comparação Gráfica =====================
# Métricas e dados para comparação
metrics = ['MAE', 'MSE', 'RMSE', 'MAPE']
arima_metrics = [mae_arima, mse_arima, rmse_arima, mape_arima]
lstm_metrics = [mae_lstm, mse_lstm, rmse_lstm, mape_lstm]

# Gerar gráfico comparativo
x = range(len(metrics))
width = 0.35  # Largura das barras

plt.figure(figsize=(12, 6))
plt.bar(x, arima_metrics, width, label='ARIMA', color='blue')
plt.bar([i + width for i in x], lstm_metrics, width, label='LSTM', color='green')

# Personalizar o gráfico
plt.title('Comparação das Métricas: ARIMA vs LSTM', fontsize=16, weight='bold')
plt.xlabel('Métricas', fontsize=14)
plt.ylabel('Valores', fontsize=14)
plt.xticks([i + width/2 for i in x], metrics, fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()

# Exibir o gráfico
plt.show()

# ===================== Resultados Finais =====================
# Exibir os valores das métricas no console
print("Métricas para o modelo ARIMA:")
print(f"MAE: {mae_arima:.2f}, MSE: {mse_arima:.2f}, RMSE: {rmse_arima:.2f}, MAPE: {mape_arima:.2f}%")

print("\nMétricas para o modelo LSTM:")
print(f"MAE: {mae_lstm:.2f}, MSE: {mse_lstm:.2f}, RMSE: {rmse_lstm:.2f}, MAPE: {mape_lstm:.2f}%")


df.info()

#### Nova Analise@@@@@@@@@@@@@@@@@@@@@@

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_tuner import RandomSearch


# Função para remover outliers com base no intervalo interquartil (IQR)
def remove_outliers_iqr(dataframe, features):
    for feature in features:
        Q1 = dataframe[feature].quantile(0.25)
        Q3 = dataframe[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        dataframe = dataframe[(dataframe[feature] >= lower_bound) & (dataframe[feature] <= upper_bound)]
    return dataframe

# Carregar os dados
df_cleaned = pd.read_csv(r"C:\Users\User\OneDrive - Instituto Presbiteriano Mackenzie\Mackenzie\Aulas\5 semestre\Projeto Aplicado\Aplicando conhecimento\dataset\sleep_cycle_productivity.csv")
df_cleaned.info
# Remover outliers
features_to_check = ['Total Sleep Hours', 'Caffeine Intake (mg)', 'Screen Time Before Bed (mins)', 'Work Hours (hrs/day)']
df_cleaned = remove_outliers_iqr(df_cleaned, features_to_check)

# Seleção de features e variáveis-alvo
X = df_cleaned[['Total Sleep Hours', 'Caffeine Intake (mg)', 'Screen Time Before Bed (mins)', 'Work Hours (hrs/day)']]
y = df_cleaned[['Total Sleep Hours', 'Screen Time Before Bed (mins)']]

# Normalização dos dados
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Redimensionar para formato compatível com LSTM [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Função para criar o modelo Stacked LSTM com ajuste dinâmico de parâmetros
def build_model(hp):
    model = Sequential()
    # Primeira camada LSTM
    model.add(LSTM(
        units=hp.Int('units_layer1', min_value=68, max_value=260, step=36),  # 4 neurônios a mais
        activation='relu',
        return_sequences=True,
        dropout=hp.Float('dropout_layer1', min_value=0.2, max_value=0.5, step=0.1),
        recurrent_dropout=0.2,
        input_shape=(X_train.shape[1], X_train.shape[2])
    ))
    # Segunda camada LSTM
    model.add(LSTM(
        units=hp.Int('units_layer2', min_value=68, max_value=260, step=36),  # 4 neurônios a mais
        activation='relu',
        return_sequences=True,
        dropout=hp.Float('dropout_layer2', min_value=0.2, max_value=0.5, step=0.1),
        recurrent_dropout=0.2
    ))
    # Terceira camada LSTM
    model.add(LSTM(
        units=hp.Int('units_layer3', min_value=68, max_value=260, step=36),  # 4 neurônios a mais
        activation='relu',
        return_sequences=True,
        dropout=hp.Float('dropout_layer3', min_value=0.2, max_value=0.5, step=0.1),
        recurrent_dropout=0.2
    ))
    # Quarta camada LSTM
    model.add(LSTM(
        units=hp.Int('units_layer4', min_value=68, max_value=260, step=36),  # 4 neurônios a mais
        activation='relu',
        return_sequences=True,
        dropout=hp.Float('dropout_layer4', min_value=0.2, max_value=0.5, step=0.1),
        recurrent_dropout=0.2
    ))
    # Quinta camada LSTM
    model.add(LSTM(
        units=hp.Int('units_layer5', min_value=68, max_value=260, step=36),  # 4 neurônios a mais
        activation='relu',
        dropout=hp.Float('dropout_layer5', min_value=0.2, max_value=0.5, step=0.1),
        recurrent_dropout=0.2
    ))
    # Camada de saída
    model.add(Dense(y.shape[1]))
    # Compilar o modelo
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])),
        loss='mse'
    )
    return model

# Configuração do Random Search para encontrar os melhores hiperparâmetros
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,  # Testar 10 combinações diferentes de hiperparâmetros
    executions_per_trial=2,  # Média de 2 execuções por combinação
    directory='grid_search_dir',  # Local para salvar os resultados
    project_name='stacked_lstm_tuning'
)

# Executar o Random Search
tuner.search(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), callbacks=[
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
])

# Recuperar os melhores hiperparâmetros
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"\nMelhores Hiperparâmetros:")
for i in range(1, 6):
    print(f"Camada {i}: {best_hps.get(f'units_layer{i}')}-neurônios, Dropout={best_hps.get(f'dropout_layer{i}')}")
print(f"Learning Rate: {best_hps.get('learning_rate')}")

# Treinar o modelo com os melhores hiperparâmetros
best_model = tuner.hypermodel.build(best_hps)

history = best_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=200,  # Treinar por mais épocas com os melhores parâmetros
    batch_size=64,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
    ]
)

# Avaliar o modelo final
predictions = best_model.predict(X_test)
predictions_inv = scaler_y.inverse_transform(predictions)
y_test_inv = scaler_y.inverse_transform(y_test)

mse = mean_squared_error(y_test_inv, predictions_inv)
mae = mean_absolute_error(y_test_inv, predictions_inv)
r2 = r2_score(y_test_inv, predictions_inv)

print("\nMétricas de Desempenho:")
print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")

# Visualização das Perdas
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Evolução da Perda')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.grid(True)
plt.show()


# Gráfico da previsão e dos valores reais
plt.figure(figsize=(14, 7))

# Linha para o primeiro alvo: Total Sleep Hours
plt.plot(y_test_inv[:, 0], label='Valores Reais - Total Sleep Hours', color='blue')
plt.plot(predictions_inv[:, 0], label='Previsões - Total Sleep Hours', color='red', linestyle='--')

# Linha para o segundo alvo: Screen Time Before Bed
plt.plot(y_test_inv[:, 1], label='Valores Reais - Screen Time Before Bed', color='green')
plt.plot(predictions_inv[:, 1], label='Previsões - Screen Time Before Bed', color='orange', linestyle='--')

# Configuração dos eixos e título
plt.title('Comparação entre Valores Reais e Previsões', fontsize=16, weight='bold')
plt.xlabel('Amostras', fontsize=14)
plt.ylabel('Valores', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()


#Prever novo individuo com base nos dados historcios 


# 1. Criar novos dados aleatórios (exemplo: 5 amostras, 4 features)
np.random.seed(42)  # para reprodutibilidade
novos_dados = np.random.rand(5, 4) * [10, 300, 120, 12]  # escala similar às features originais (exemplo)

print("Novos dados (antes da normalização):")
print(novos_dados)

# 2. Normalizar os dados com scaler_X (assumindo que scaler_X está ajustado)
novos_dados_scaled = scaler_X.transform(novos_dados)

print("\nNovos dados (normalizados):")
print(novos_dados_scaled)

# 3. Redimensionar para o formato que o modelo espera: (n_samples, 4, 1)
novos_dados_scaled_reshaped = novos_dados_scaled.reshape((novos_dados_scaled.shape[0], novos_dados_scaled.shape[1], 1))
print("\nShape após reshape:", novos_dados_scaled_reshaped.shape)

# 4. Prever com o melhor modelo
predictions_scaled = best_model.predict(novos_dados_scaled_reshaped)

print("\nPrevisões normalizadas:")
print(predictions_scaled)

# 5. Inverter a normalização para as previsões
predictions = scaler_y.inverse_transform(predictions_scaled)

print("\nPrevisões em escala original:")
print(predictions)

# 6. Plotar as previsões
plt.figure(figsize=(10,6))

# Supondo que a saída são 2 variáveis (como no seu y: 'Total Sleep Hours' e 'Screen Time Before Bed (mins)')
plt.plot(predictions[:, 0], marker='o', label='Total Sleep Hours (previsto)')
plt.plot(predictions[:, 1], marker='x', label='Screen Time Before Bed (previsto)')

plt.title("Previsões para novos dados")
plt.xlabel("Amostras")
plt.ylabel("Valores previstos")
plt.legend()
plt.grid(True)
plt.show()



