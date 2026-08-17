"""Testes da API de previsao usada pela interface grafica."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from energia import previsao as pv
from energia.config import MODELOS, PROCESSADO

# Estes testes precisam do dataset e dos modelos ja gerados.
precisa_artefatos = pytest.mark.skipif(
    not (PROCESSADO / "dataset_modelo.parquet").exists()
    or not (MODELOS / "climatico_SE.joblib").exists(),
    reason="rode scripts/01..03 antes")


# --- horas acima do limiar (nao depende de artefatos) -----------------------

def test_horas_acima_dentro_dos_limites():
    for t_med, t_max, t_min in [(20, 25, 15), (30, 36, 24), (26, 26.5, 25.5)]:
        h = pv._horas_acima(26.0, t_med, t_max, t_min)
        assert 0.0 <= h <= 24.0


def test_horas_acima_e_monotonica_na_temperatura():
    anterior = -1.0
    for t in range(15, 40):
        h = pv._horas_acima(26.0, t, t + 5, t - 5)
        assert h >= anterior
        anterior = h


def test_horas_acima_casos_extremos():
    assert pv._horas_acima(26.0, 40, 45, 35) == pytest.approx(24.0)
    assert pv._horas_acima(26.0, 10, 14, 6) == pytest.approx(0.0)
    # media exatamente no limiar: metade do dia acima
    assert pv._horas_acima(26.0, 26, 31, 21) == pytest.approx(12.0, abs=0.1)


# --- montagem de features ---------------------------------------------------

@precisa_artefatos
def test_preenche_clima_ausente_com_mediana_da_regiao():
    entrada = pd.DataFrame({"id_subsistema": ["SE"], "din_instante": ["2026-03-10"]})
    f = pv.montar_features(entrada)
    padrao = pv.padroes_regionais()["SE"]
    assert f["temp_mean"].iloc[0] == pytest.approx(padrao["temp_mean"])
    assert f[pv.CAMPOS_CLIMA].notna().all().all()


@precisa_artefatos
def test_respeita_o_valor_informado_pelo_usuario():
    entrada = pd.DataFrame({"id_subsistema": ["SE"], "din_instante": ["2026-03-10"],
                            "temp_mean": [31.5]})
    f = pv.montar_features(entrada)
    assert f["temp_mean"].iloc[0] == pytest.approx(31.5)
    assert f["graus_refrig"].iloc[0] == pytest.approx(31.5 - 22.0)


@precisa_artefatos
def test_calendario_derivado_da_data():
    # 2026-09-07 (Independencia) cai numa segunda: feriado, mas nao e fim de semana
    entrada = pd.DataFrame({"id_subsistema": ["S"], "din_instante": ["2026-09-07"]})
    f = pv.montar_features(entrada).iloc[0]
    assert f["mes"] == 9 and f["trimestre"] == 3
    assert f["dia_semana"] == 0
    assert f["is_feriado"] == 1
    assert f["is_fds"] == 0

    # 2026-09-05 e sabado comum: fim de semana, sem feriado
    sabado = pd.DataFrame({"id_subsistema": ["S"], "din_instante": ["2026-09-05"]})
    s = pv.montar_features(sabado).iloc[0]
    assert s["is_fds"] == 1 and s["is_feriado"] == 0
    # vespera de feriado: 2026-09-06, domingo antes da Independencia
    vespera = pd.DataFrame({"id_subsistema": ["S"], "din_instante": ["2026-09-06"]})
    assert pv.montar_features(vespera).iloc[0]["is_vespera_feriado"] == 1


@precisa_artefatos
def test_temp_max_e_min_sao_coeridas_com_a_media():
    entrada = pd.DataFrame({"id_subsistema": ["N"], "din_instante": ["2026-03-10"],
                            "temp_mean": [33.0], "temp_max": [25.0],
                            "temp_min": [40.0]})
    f = pv.montar_features(entrada).iloc[0]
    assert f["temp_max"] >= f["temp_mean"] >= f["temp_min"]


@precisa_artefatos
def test_subsistema_invalido_gera_erro():
    entrada = pd.DataFrame({"id_subsistema": ["CO"], "din_instante": ["2026-03-10"]})
    with pytest.raises(ValueError, match="desconhecido"):
        pv.montar_features(entrada)


@precisa_artefatos
def test_nao_exige_nenhuma_carga_observada():
    """A interface nao pode pedir carga -- se pedisse, seria contaminavel."""
    entrada = pd.DataFrame({"id_subsistema": ["SE"], "din_instante": ["2030-06-01"]})
    r = pv.prever(entrada)
    assert "val_cargaenergiamwmed" not in entrada.columns
    assert r["previsao_mwmed"].notna().all()


# --- previsao ---------------------------------------------------------------

@precisa_artefatos
def test_decomposicao_soma_a_previsao():
    entrada = pd.DataFrame({"id_subsistema": ["SE", "N", "S", "NE"],
                            "din_instante": ["2026-02-10"] * 4})
    r = pv.prever(entrada)
    soma = r["nivel_industrial"] + r["resposta_climatica"]
    assert np.allclose(soma, r["previsao_mwmed"])


@precisa_artefatos
def test_previsao_em_faixa_plausivel():
    entrada = pd.DataFrame({"id_subsistema": ["N", "NE", "S", "SE"],
                            "din_instante": ["2026-02-10"] * 4})
    r = pv.prever(entrada).set_index("id_subsistema")["previsao_mwmed"]
    assert 4_000 < r["N"] < 15_000
    assert 8_000 < r["NE"] < 22_000
    assert 7_000 < r["S"] < 25_000
    assert 25_000 < r["SE"] < 70_000
    assert r["SE"] > r["NE"] and r["SE"] > r["N"]


@precisa_artefatos
def test_ordem_das_linhas_e_preservada():
    entrada = pd.DataFrame({"id_subsistema": ["SE", "N", "SE", "S"],
                            "din_instante": ["2026-02-10"] * 4})
    r = pv.prever(entrada)
    assert list(r["id_subsistema"]) == ["SE", "N", "SE", "S"]


@precisa_artefatos
def test_temperatura_maior_aumenta_a_carga_no_sudeste():
    entrada = pd.DataFrame({"id_subsistema": ["SE", "SE"],
                            "din_instante": ["2026-02-10"] * 2,
                            "temp_mean": [20.0, 29.0]})
    r = pv.prever(entrada)["previsao_mwmed"].to_numpy()
    assert r[1] > r[0], "calor deveria aumentar a carga no SE"


@precisa_artefatos
def test_curva_temperatura_tem_o_tamanho_pedido():
    c = pv.curva_temperatura("S", "2026-07-15", passos=25)
    assert len(c) == 25
    assert c["temp_mean"].is_monotonic_increasing
    assert c["previsao_mwmed"].notna().all()


@precisa_artefatos
def test_contexto_historico_e_coerente():
    ctx = pv.contexto_historico("SE", "2026-02-10")
    assert ctx["carga_min"] <= ctx["carga_media"] <= ctx["carga_max"]
    assert ctx["n_dias"] > 0
