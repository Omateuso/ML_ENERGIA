"""Testes das garantias anti-contaminacao.

Sao as invariantes que sustentam a leitura dos resultados. Se qualquer um
destes quebrar, as metricas do projeto deixam de significar o que dizem.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energia import features as ft
from energia.avaliacao import metricas, prever_recursivo
from energia.config import REGIAO_INMET_PARA_SUBSISTEMA
from energia.features import ALVO


@pytest.fixture
def df_sintetico():
    datas = pd.date_range("2024-01-01", periods=60, freq="D")
    linhas = []
    for sub in ("N", "SE"):
        rng = np.random.default_rng(hash(sub) % 1000)
        linhas.append(pd.DataFrame({
            "din_instante": datas,
            "id_subsistema": sub,
            ALVO: rng.uniform(5000, 6000, len(datas)),
            "temp_mean": rng.uniform(20, 30, len(datas)),
            "temp_max": rng.uniform(28, 36, len(datas)),
            "temp_min": rng.uniform(14, 22, len(datas)),
            "graus_refrig": rng.uniform(0, 6, len(datas)),
            "industrial_mwmed": np.linspace(1000, 1100, len(datas)),
        }))
    return pd.concat(linhas, ignore_index=True)


# --- a garantia central -----------------------------------------------------

def test_modo_climatico_nao_usa_nenhuma_feature_autoregressiva(df_sintetico):
    df = ft._autoregressivo(df_sintetico.copy())
    feats = ft.features_do_modo("climatico", df)
    vazamento = set(feats) & set(ft.AUTOREGRESSIVO)
    assert not vazamento, f"modo climatico expos features de carga: {vazamento}"
    assert ALVO not in feats


def test_modo_d1_declara_as_autoregressivas(df_sintetico):
    df = ft._autoregressivo(df_sintetico.copy())
    feats = ft.features_do_modo("delta_d1", df)
    assert set(ft.AUTOREGRESSIVO).issubset(set(feats))


def test_media_7d_nao_inclui_o_dia_atual():
    """media_7d[t] deve cobrir t-7..t-1. Se incluir t, entrega o alvo."""
    n = 20
    df = pd.DataFrame({
        "din_instante": pd.date_range("2024-01-01", periods=n, freq="D"),
        "id_subsistema": "SE",
        ALVO: np.arange(n, dtype=float),
    })
    r = ft._autoregressivo(df)
    # Em t=14 a janela cobre t-7..t-1, ou seja os indices 7..13.
    assert r["media_7d"].iloc[14] == pytest.approx(np.arange(7, 14).mean())
    # Uma serie estritamente crescente: a media do passado e sempre < valor atual.
    validos = r.dropna(subset=["media_7d"])
    assert (validos["media_7d"] < validos[ALVO]).all()


def test_carga_ontem_e_lag_puro():
    n = 15
    df = pd.DataFrame({
        "din_instante": pd.date_range("2024-01-01", periods=n, freq="D"),
        "id_subsistema": "N",
        ALVO: np.arange(n, dtype=float) * 10,
    })
    r = ft._autoregressivo(df)
    assert r["carga_ontem"].iloc[5] == pytest.approx(40.0)
    assert r["carga_lag7"].iloc[10] == pytest.approx(30.0)
    assert pd.isna(r["carga_ontem"].iloc[0])


def test_lags_nao_atravessam_subsistemas(df_sintetico):
    """O shift precisa ser por regiao; senao o Norte herda a carga do Sudeste."""
    r = ft._autoregressivo(df_sintetico.copy())
    primeiro_de_cada = r.groupby("id_subsistema").head(1)
    assert primeiro_de_cada["carga_ontem"].isna().all()


# --- mapeamento de regiao ---------------------------------------------------

def test_centro_oeste_vai_para_o_subsistema_sudeste():
    """Sem isto as 96 estacoes do CO ficam orfas e somem no merge."""
    assert REGIAO_INMET_PARA_SUBSISTEMA["CO"] == "SE"
    assert set(REGIAO_INMET_PARA_SUBSISTEMA.values()) == {"N", "NE", "S", "SE"}


# --- previsao recursiva -----------------------------------------------------

class ModeloEspiao:
    """Devolve carga_ontem + 1 e registra tudo que leu."""

    def __init__(self):
        self.vistos = []

    def predict(self, df):
        self.vistos.append(df["carga_ontem"].to_numpy()[0])
        return df["carga_ontem"].to_numpy() + 1.0


def test_recursivo_nao_le_carga_observada_apos_a_semente():
    n = 30
    teste = pd.DataFrame({
        "din_instante": pd.date_range("2025-01-01", periods=n, freq="D"),
        "id_subsistema": "SE",
        ALVO: np.full(n, 9999.0),           # valor-sentinela: nao pode vazar
        "carga_ontem": np.full(n, 9999.0),
        "carga_lag7": np.full(n, 9999.0),
        "media_7d": np.full(n, 9999.0),
    })
    teste.loc[:6, ALVO] = 100.0             # semente controlada

    espiao = ModeloEspiao()
    p = prever_recursivo(espiao, teste, dias_semente=7)

    assert 9999.0 not in espiao.vistos, "carga observada do teste vazou para o modelo"
    assert p[:7].tolist() == [100.0] * 7
    assert p[7] == pytest.approx(101.0)     # 100 + 1, realimentado
    assert p[8] == pytest.approx(102.0)


# --- metricas ---------------------------------------------------------------

def test_metricas_previsao_perfeita():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = metricas(y, y)
    assert m["mae"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["n"] == 4


def test_metricas_ignoram_nan():
    y = np.array([1.0, 2.0, np.nan, 4.0])
    p = np.array([1.0, 2.0, 3.0, 4.0])
    assert metricas(y, p)["n"] == 3
