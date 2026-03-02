import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
import os
from xgboost import XGBRegressor
import sys
from importlib import import_module
import joblib

# Adicionando o diretório atual ao path para permitir a importação do train_pipeline
sys.path.insert(0, os.path.dirname(__file__))

@pytest.fixture
def sample_energy_data():
    # Criando conjunto de dados de energia fictício para os testes
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'din_instante': dates,
        'id_subsistema': ['SE'] * 100,
        'val_cargaenergiamwmed': np.random.uniform(50000, 60000, 100)
    })

@pytest.fixture
def sample_climate_data():
    # Criando conjunto de dados de clima fictício para os testes
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'data_clima': dates,
        'id_subsistema': ['SE'] * 100,
        'temp_media': np.random.uniform(20, 30, 100),
        'umidade_media': np.random.uniform(50, 80, 100)
    })

def test_processar_dados_invalid_path():
    # Testando se a função lida corretamente com caminhos de arquivos inexistentes
    train_module = import_module('train_pipeline')
    result = train_module.processar_dados('caminho/invalido')
    assert result is None

def test_carregar_e_preparar_file_not_found():
    # Verificando se o sistema levanta erro quando o arquivo principal não é encontrado
    train_module = import_module('train_pipeline')
    with pytest.raises(FileNotFoundError):
        train_module.carregar_e_preparar('arquivo_inexistente.csv')

def test_carregar_e_preparar_with_data(sample_energy_data, tmp_path):
    # Testando a lógica de preparação de dados e engenharia de features
    train_module = import_module('train_pipeline')
    
    # Criando um arquivo CSV temporário para realizar o teste
    csv_file = tmp_path / "test_energy.csv"
    sample_energy_data.to_csv(csv_file, sep=';', index=False)
    
    # Utilizando patches para evitar erros de tradução da biblioteca holidays e processamento de clima
    with patch('os.path.exists', return_value=True):
        with patch('holidays.Brazil', return_value={}):
            with patch.object(train_module, 'processar_dados', return_value=None):
                result = train_module.carregar_e_preparar(str(csv_file))
                # Verificando se as colunas básicas de tempo foram criadas corretamente
                assert result is not None
                assert 'mes' in result.columns
                assert 'dia_semana' in result.columns

def test_treinar_modelo_empty_subsystem():
    # Verificando se o treino retorna None quando não existem dados para um subsistema específico
    train_module = import_module('train_pipeline')
    
    df = pd.DataFrame({'id_subsistema': ['SE'], 'val_cargaenergiamwmed': [100]})
    result = train_module.treinar_modelo_com_rastreamento_carbono(df, subsistema='NE')
    assert result is None

def test_treinar_modelo_with_valid_data(sample_energy_data):
    # Testando o fluxo completo de treinamento do XGBoost com rastreamento de carbono
    train_module = import_module('train_pipeline')
    
    # Preparando dataframe manualmente para simular a entrada do modelo
    df = sample_energy_data.copy()
    df['mes'] = df['din_instante'].dt.month
    df['dia_semana'] = df['din_instante'].dt.dayofweek
    df['trimestre'] = df['din_instante'].dt.quarter
    df['carga_ontem'] = df['val_cargaenergiamwmed'].shift(1)
    df['is_feriado'] = 0
    df['is_fds'] = 0
    df['media_7d'] = df['val_cargaenergiamwmed'].rolling(7).mean()
    df['temp_media'] = np.random.uniform(20, 30, len(df))
    
    # Simulando o rastreador de carbono para não gerar arquivos reais durante o teste
    with patch('codecarbon.EmissionsTracker'):
        result = train_module.treinar_modelo_com_rastreamento_carbono(df, subsistema='SE')
        assert result is not None
        assert isinstance(result, XGBRegressor)

def test_feature_engineering(sample_energy_data):
    # Validando se as transformações temporais estão dentro dos limites lógicos
    df = sample_energy_data.copy()
    df['din_instante'] = pd.to_datetime(df['din_instante'])
    
    df['mes'] = df['din_instante'].dt.month
    df['dia_semana'] = df['din_instante'].dt.dayofweek
    df['trimestre'] = df['din_instante'].dt.quarter
    
    # Verificando limites dos períodos temporais
    assert df['mes'].min() >= 1 and df['mes'].max() <= 12
    assert df['dia_semana'].min() >= 0 and df['dia_semana'].max() <= 6
    assert df['trimestre'].min() >= 1 and df['trimestre'].max() <= 4

def test_model_persistence(sample_energy_data, tmp_path):
    # Garantindo que o modelo pode ser salvo e carregado mantendo sua integridade
    train_module = import_module('train_pipeline')
    
    df = sample_energy_data.copy()
    df['mes'] = df['din_instante'].dt.month
    df['dia_semana'] = df['din_instante'].dt.dayofweek
    df['trimestre'] = df['din_instante'].dt.quarter
    df['carga_ontem'] = df['val_cargaenergiamwmed'].shift(1)
    df['is_feriado'] = 0
    df['is_fds'] = 0
    df['media_7d'] = df['val_cargaenergiamwmed'].rolling(7).mean()
    
    with patch('codecarbon.EmissionsTracker'):
        model = train_module.treinar_modelo_com_rastreamento_carbono(df, subsistema='SE')
    
    if model is not None:
        # Testando o processo de salvar e carregar o arquivo .joblib
        model_path = tmp_path / "test_model.joblib"
        joblib.dump(model, str(model_path))
        loaded_model = joblib.load(str(model_path))
        assert isinstance(loaded_model, XGBRegressor)

