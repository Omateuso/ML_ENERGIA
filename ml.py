import pandas as pd 
import numpy as np
from datetime import datetime  
from xgboost import XGBRegressor  # Importação direta para evitar NameError
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import holidays
import glob
import os
import time
from codecarbon import EmissionsTracker
import matplotlib.pyplot as plt
import seaborn as sns 

# %% 
# Função para importar dataset de uma pasta
def process_inmet_folder(folder_path='dataset/clima/2025'):
    all_weather_data = []
    search_path = os.path.join(folder_path, "*.CSV")
    files = glob.glob(search_path)

    if not files:
        print(f"Aviso: Nenhum arquivo de clima encontrado em {folder_path}")
        return None

    print(f"Processando {len(files)} arquivos de clima...")

    for file in files:
        try:
            with open(file, 'r', encoding='latin-1') as f:
                first_line = f.readline()
                # Extrai a região (SE, S, NE, N) do cabeçalho
                region = first_line.split(';')[1].strip()

            df_temp = pd.read_csv(file, sep=';', encoding='latin-1', skiprows=8, decimal=',')
            col_data = 'Data'
            col_temp = 'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'

            if col_temp in df_temp.columns:
                df_temp[col_data] = pd.to_datetime(df_temp[col_data].str.replace('/','-'))
                daily_station = df_temp.groupby(col_data)[col_temp].mean().reset_index()
                daily_station['id_subsistema'] = region
                all_weather_data.append(daily_station)

        except Exception as e:
            pass

    if not all_weather_data:
        return None

    full_weather = pd.concat(all_weather_data)
    # Agrupa por Data e Região para ter uma média única por subsistema
    regional_weather = full_weather.groupby(['Data','id_subsistema'])[col_temp].mean().reset_index()
    regional_weather.rename(columns={col_temp: 'temp_media', 'Data': 'data_clima'}, inplace=True)

    return regional_weather

# %%     
# Carregamento e preparação dos dados
def load_and_prep(energy_path, inmet_folder=None):
    if not os.path.exists(energy_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {energy_path}")

    df = pd.read_csv(energy_path, sep=';')
    
    # Convertendo para datetime
    df['din_instante'] = pd.to_datetime(df['din_instante'])
    
    # Separando mes, dia da semana, trimestre e ano
    df['mes'] = df['din_instante'].dt.month
    df['dia_semana'] = df['din_instante'].dt.dayofweek
    df['trimestre'] = df['din_instante'].dt.quarter
    df['ano'] = df['din_instante'].dt.year
    
    # Carga do dia anterior e organizando por região e dia cronológico
    df = df.sort_values(['id_subsistema', 'din_instante'])
    df['carga_ontem'] = df.groupby('id_subsistema')['val_cargaenergiamwmed'].shift(1)

    # Integrando dados do clima
    target_clima = inmet_folder if (inmet_folder and os.path.exists(inmet_folder)) else 'dataset/clima/2025'
    
    if os.path.exists(target_clima):
        df_weather = process_inmet_folder(target_clima)
        if df_weather is not None:
            df = pd.merge(df, df_weather, left_on=['din_instante', 'id_subsistema'],
                          right_on=['data_clima', 'id_subsistema'], how='left')
            print(f"Dados de clima da pasta {target_clima} integrados com sucesso.")
            
    return df.dropna()

# %%
# Treino e Green Coding
def train_model_with_carbon_tracking(df, subsistema='SE'):
    df_sub = df[df['id_subsistema'] == subsistema]

    if df_sub.empty:
        print(f"Aviso: O subsistema {subsistema} não possui dados após a limpeza.")
        return None

    # Define as colunas que o modelo vai usar para prever
    features = ['mes','dia_semana','trimestre','carga_ontem']
    if 'temp_media' in df_sub.columns:
        features.append('temp_media')

    X = df_sub[features]
    y = df_sub['val_cargaenergiamwmed']

    # Divisão treino/teste (shuffle=False para manter a ordem temporal)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Inicia rastreamento de CO2
    tracker = EmissionsTracker(project_name="Previsao_Energia_XGBoost", output_dir="output/carbon") 
    tracker.start()

    start_time = time.time()

    # Treinamento do Modelo XGBoost (Chamada direta corrigida)
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    end_time = time.time()
    emissions: float = tracker.stop()

    # Avaliação das métricas
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds) 
    
    print(f"\n--- Relatório Final [{subsistema}] ---")
    print(f"Erro Médio Absoluto (MAE): {mae:.2f} MWmed")
    print(f"Precisão (R2 Score): {r2:.4f}")
    print(f"Tempo de Treino: {end_time - start_time:.4f}s")
    print(f"Emissões Estimadas: {emissions:.6f} kg de CO2eq")
    print("-" * 30)
    
    return model

# %%
if __name__ == "__main__":
    energy_file = 'dataset/CARGA_ENERGIA_2025.csv'
    inmet_path = 'dataset/clima/2025'

    try:
        # Carregamento e integração
        data = load_and_prep(energy_file, inmet_path)
        
        # Treinamento para o subsistema SE (Sudeste/Centro-Oeste)
        model = train_model_with_carbon_tracking(data, subsistema='SE')

    except Exception as e:
        print(f"Erro Crítico: {e}")
# %%