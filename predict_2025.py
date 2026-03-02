#%%
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from train_pipeline import carregar_e_preparar
#%%
def avaliar_desempenho(df,subsistema):
    if df.empty:
        return
    y_real = df['val_cargaenergiamwmed']
    y_pred = df['previsao_modelo']

    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)
    print(f"\n--- Relatório de Precisão 2025 [{subsistema}] ---")
    print(f"Erro Médio Absoluto (MAE): {mae:.2f} MWmed")
    print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.2f} MWmed")
    print(f"R² Score (Variância explicada): {r2:.4f}")
#%%
# Função para fazer previsão
def fazer_previsao(caminho_csv_2025, caminho_inmet_2025, subsistema='SE'):
    # Carregando modelo que treinei
    caminho_modelo = f'modelos/modelo_energia_{subsistema}.joblib'
    caminho_inmet_2025 = 'dataset/2025/INMET_2025'
    modelo = joblib.load(caminho_modelo)

    # Preparando dados de 2025 usando a mesma lógica do treino
    df_2025 = carregar_e_preparar(caminho_csv_2025, caminho_inmet = caminho_inmet_2025)
    df_sub = df_2025[df_2025['id_subsistema'] == subsistema]

    # Selecionando mesmas features que o modelo treinado
    features = ['mes', 'dia_semana', 'trimestre', 'carga_ontem', 'is_feriado', 'is_fds', 'media_7d']
    if 'temp_media' in df_sub.columns:
        features.append('temp_media')

    if 'umidade_media' in df_sub.columns and subsistema != 'NE':
        features.append('umidade_media')

    X_new = df_sub[features]

    # Fazendo previsão
    previsoes = modelo.predict(X_new)

    # Comparando
    resultado = df_sub[['din_instante', 'val_cargaenergiamwmed']].copy()
    resultado['previsao_modelo'] = previsoes
    return resultado
#%%
# Testando com os dados de 2025
if __name__ == "__main__":
    caminho_2025 = 'dataset/2025/CARGA_ENERGIA_2025.csv'
    caminho_inmet_2025 = 'dataset/2025/INMET_2025'
    
    # Garante que a pasta de saída existe
    pasta_saida = 'output/previsao'
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)

    df_temp = pd.read_csv(caminho_2025, sep=';')
    subsistemas_presentes = df_temp['id_subsistema'].unique()

    for subs in subsistemas_presentes:
        try:
            print(f"Gerando previsão para: {subs}...")
            df_final = fazer_previsao(caminho_2025, caminho_inmet_2025, subsistema=subs)
            
            if not df_final.empty:
                avaliar_desempenho(df_final, subs)
                df_final.to_csv(f'{pasta_saida}/previsao_2025_{subs}.csv', index=False)
        except Exception as e:
            print(f"Subsistema {subs} pulado por motivo: {e}")
#%%