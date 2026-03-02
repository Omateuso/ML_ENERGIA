#%%
import pandas as pd 
from xgboost import XGBRegressor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import holidays
import glob
import os
import time
from codecarbon import EmissionsTracker
import joblib

# MAPEAMENTO OBRIGATÓRIO PARA AS REGIÕES
NOMES_REGIOES = {
    'N': 'Norte',
    'S': 'Sul',
    'NE': 'Nordeste',
    'SE': 'Sudeste'
}

#%%
# Função para importar dataset de uma pasta
def processar_dados(caminho_pasta = 'dataset/2024/INMET_2024'):
    # Processando dados de clima
    registro_clima = []
    procurar_caminho = os.path.join(caminho_pasta, "*.CSV")
    arquivos = glob.glob(procurar_caminho)
    # Verificando se foram encontrados arquivos de clima
    if not arquivos:
        print(f"Aviso: Nenhum arquivo de clima encontrado em {caminho_pasta}")
        return None
    print(f"Processando {len(arquivos)} arquivos de clima...")
    # Iterando sobre cada arquivo encontrado
    for arquivo in arquivos:
        try:
            with open(arquivo, 'r', encoding='latin-1') as f:
                primeira_linha = f.readline()
                # Extraindo região (SE, S, NE, N) do cabeçalho
                regiao_sigla = primeira_linha.split(';')[1].strip()
            # Lendo o arquivo CSV
            df_temp = pd.read_csv(arquivo, sep=';', encoding='latin-1', skiprows=8, decimal=',')
            # Coluna de data e coluna de temperatura
            coluna_data = 'Data'
            coluna_temp = 'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'
            coluna_umi = 'UMIDADE RELATIVA DO AR, HORARIA (%)'
            # Verificando se a coluna de temperatura existe no DataFrame
            if coluna_temp in df_temp.columns and coluna_umi in df_temp.columns:
                # Convertendo a coluna de data para o formato datetime
                df_temp[coluna_data] = pd.to_datetime(df_temp[coluna_data].str.replace('/','-'))
                # Agrupando por data e calculando a média da temperatura para cada estação
                clima_diario_estacao = df_temp.groupby(coluna_data)[[coluna_temp, coluna_umi]].mean().reset_index()
                # Adicionando a coluna de região ao DataFrame de clima diário
                clima_diario_estacao['id_subsistema'] = regiao_sigla
                # Adicionando o DataFrame de clima diário da estação à lista de registros de clima
                registro_clima.append(clima_diario_estacao)

        except Exception as e:
            pass

    # Verificando se foram processados dados de clima
    if not registro_clima:
        return None
    
    # Concatenando todos os DataFrames de clima em um único DataFrame
    clima_completo = pd.concat(registro_clima)
    # Agrupando por data e região para obter uma média única por região
    clima_regional = clima_completo.groupby(['Data','id_subsistema'])[[coluna_temp, coluna_umi]].mean().reset_index()
    # Renomeando as colunas para facilitar a manipulação posterior
    clima_regional.rename(columns={coluna_temp: 'temp_media', coluna_umi: 'umidade_media', 'Data': 'data_clima'}, inplace=True)
    return clima_regional

#%%
# Função para carregamento e preparação dos dados
def carregar_e_preparar(caminho_energia, caminho_inmet=None):
    # Verificando se o arquivo de energia existe
    if not os.path.exists(caminho_energia):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_energia}")
    
    # Carregando o dataset de energia
    df = pd.read_csv(caminho_energia, sep=';')
    # Convertendo para datetime
    df['din_instante'] = pd.to_datetime(df['din_instante'])

    # Extraindo mes, dia da semana, trimestre e ano a partir da coluna de data
    df['mes'] = df['din_instante'].dt.month
    df['dia_semana'] = df['din_instante'].dt.dayofweek
    df['trimestre'] = df['din_instante'].dt.quarter
    df['ano'] = df['din_instante'].dt.year
    # Feriados e fins de semana
    br_holidays = holidays.Brazil(language='en_US')
    # Criando uma coluna para indicar se o dia é feriado ou fim de semana
    df['is_feriado'] = df['din_instante'].apply(lambda x: 1 if x in br_holidays else 0)
    # Criando uma coluna para indicar se o dia é fim de semana
    df['is_fds'] = df['dia_semana'].apply(lambda x: 1 if x >=5 else 0)

    # Carga do dia anterior, média móvel de 7 dias e organizando por região
    df = df.sort_values(by=['id_subsistema', 'din_instante'])
    df['carga_ontem'] = df.groupby('id_subsistema')['val_cargaenergiamwmed'].shift(1)
    df['media_7d'] = df.groupby('id_subsistema')['val_cargaenergiamwmed'].transform(lambda x: x.rolling(window = 7).mean())

    # Integrando dados de clima
    clima_alvo = caminho_inmet if (caminho_inmet and os.path.exists(caminho_inmet)) else 'dataset/2024/INMET_2024'
    if os.path.exists(clima_alvo):
        df_clima = processar_dados(clima_alvo)
        if df_clima is not None:
            df = pd.merge(df, df_clima, left_on= ['din_instante', 'id_subsistema'], 
                          right_on= ['data_clima', 'id_subsistema'], how='left')
            print(f"Dados de clima da pasta {clima_alvo} integrados com sucesso.")
    return df.dropna()

# %%
# Treinamento do modelo e rastreamento de emissões de carbono
def treinar_modelo_com_rastreamento_carbono(df, regiao='SE'):
    # Filtrando dados para a região específica
    df_sub = df[df['id_subsistema'] == regiao]
    nome_exibicao = NOMES_REGIOES.get(regiao, regiao)

    if df_sub.empty:
        print(f"Aviso: A região {nome_exibicao} não possui dados após a limpeza.")
        return None
        
    features = ['mes','dia_semana','trimestre','carga_ontem','is_feriado','is_fds','media_7d']
    if 'temp_media' in df_sub.columns:
        features.append('temp_media')
    
    if 'umidade_media' in df_sub.columns and regiao != 'NE':
        features.append('umidade_media')

    X = df_sub[features]
    y = df_sub['val_cargaenergiamwmed']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    tracker = EmissionsTracker(project_name="Previsao_Energia_XGBoost", output_dir="output/carbon")
    tracker.start()
    
    start_time = time.time()
    model = XGBRegressor(
        n_estimators = 200,
        learning_rate = 0.1,
        max_depth = 6,
        n_jobs = -1,
        random_state = 42
    )
    model.fit(X_train, y_train)
    
    end_time = time.time()
    emissions: float = tracker.stop()
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n--- Relatório Final [Região: {nome_exibicao}] ---")
    print(f"ERRO MÉDIO ABSOLUTO (MAE): {mae:.2f} MWmed")
    print(f"Precisão (R² Score): {r2:.4f}")
    print(f"Tempo de treinamento: {end_time - start_time:.2f} segundos")
    print(f"Emissões de carbono: {emissions:.2f} kg CO₂")
    return model

# %%
if __name__ == "__main__":
    caminho_energia = 'dataset/2024/CARGA_ENERGIA_2024.csv'
    caminho_inmet = 'dataset/2024/INMET_2024'

    if not os.path.exists('modelos'): os.makedirs('modelos')

    try:
        df_preparado = carregar_e_preparar(caminho_energia, caminho_inmet)
        regioes_siglas = df_preparado['id_subsistema'].unique()
        
        regioes_nomes = [NOMES_REGIOES.get(s, s) for s in regioes_siglas]
        print(f"Regiões encontradas: {regioes_nomes}")
        
        for sigla in regioes_siglas:
            nome_regiao = NOMES_REGIOES.get(sigla, sigla)
            print(f"\nTreinando modelo para a região: {nome_regiao}")
            modelo_regiao = treinar_modelo_com_rastreamento_carbono(df_preparado, regiao = sigla)
            if modelo_regiao is not None:
                nome_arquivo = f'modelos/modelo_energia_{sigla}.joblib'
                joblib.dump(modelo_regiao, nome_arquivo)
                print(f"Modelo para a região {nome_regiao} salvo como {nome_arquivo}")
    except FileNotFoundError as e:
        print(e)
# %%
