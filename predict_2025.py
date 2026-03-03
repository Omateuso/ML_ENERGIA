#%%
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from train_pipeline import carregar_e_preparar, NOMES_REGIOES
#%%
def avaliar_desempenho(df, regiao_sigla):
    if df.empty:
        return
    y_real = df['val_cargaenergiamwmed']
    y_pred = df['previsao_modelo']
    
    # Busca o nome completo da região (Norte, Sul, etc.)
    nome_regiao = NOMES_REGIOES.get(regiao_sigla, regiao_sigla)

    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)
    print(f"\n--- Relatório de Precisão 2025 [Região: {nome_regiao}] ---")
    print(f"Erro Médio Absoluto (MAE): {mae:.2f} MWmed")
    print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.2f} MWmed")
    print(f"R² Score (Variância explicada): {r2:.4f}")
#%%
# Função para fazer previsão
def fazer_previsao(caminho_csv_2025, caminho_inmet_2025, regiao='SE'):
    # Carregando modelo treinado usando a sigla (id_subsistema)
    caminho_modelo = f'modelos/modelo_energia_{regiao}.joblib'
    caminho_inmet_2025 = 'dataset/2025/INMET_2025'
    
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(f"Modelo não encontrado para a região {regiao} em {caminho_modelo}")
        
    modelo = joblib.load(caminho_modelo)

    # Preparando dados de 2025
    df_2025 = carregar_e_preparar(caminho_csv_2025, caminho_inmet = caminho_inmet_2025)
    df_sub = df_2025[df_2025['id_subsistema'] == regiao]

    # Selecionando mesmas features do treinamento
    features = ['mes', 'dia_semana', 'trimestre', 'carga_ontem', 'is_feriado', 'is_fds', 'media_7d']
    if 'temp_media' in df_sub.columns:
        features.append('temp_media')

    if 'umidade_media' in df_sub.columns and regiao != 'NE':
        features.append('umidade_media')

    X_new = df_sub[features]

    # Fazendo previsão
    previsoes_delta = modelo.predict(X_new)

    previsoes_absolutas = df_sub['carga_ontem'] + previsoes_delta

    # Formatando resultado
    resultado = df_sub[['din_instante', 'val_cargaenergiamwmed']].copy()
    resultado['previsao_modelo'] = previsoes_absolutas
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

    if not os.path.exists(caminho_2025):
        print(f"Erro: Arquivo {caminho_2025} não encontrado.")
    else:
        df_temp = pd.read_csv(caminho_2025, sep=';')
        regioes_presentes = df_temp['id_subsistema'].unique()

        for sigla in regioes_presentes:
            nome_regiao = NOMES_REGIOES.get(sigla, sigla)
            try:
                print(f"Gerando previsão para a região: {nome_regiao}...")
                df_final = fazer_previsao(caminho_2025, caminho_inmet_2025, regiao=sigla)
                
                if not df_final.empty:
                    avaliar_desempenho(df_final, sigla)
                    df_final.to_csv(f'{pasta_saida}/previsao_2025_{sigla}.csv', index=False)
            except Exception as e:
                print(f"Região {nome_regiao} pulada por motivo: {e}")
#%%