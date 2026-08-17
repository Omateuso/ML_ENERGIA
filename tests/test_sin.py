"""Testes do agregado nacional (SIN) e do preenchimento por climatologia."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from energia import previsao as pv
from energia.config import MODELOS, NOMES_REGIOES, PROCESSADO, SIN, SUBSISTEMAS
from energia.entrada import normalizar
from energia.features import ALVO, agregar_sin, pesos_do_sin

precisa_artefatos = pytest.mark.skipif(
    not (PROCESSADO / "dataset_modelo.parquet").exists()
    or not (MODELOS / f"climatico_{SIN}.joblib").exists(),
    reason="rode scripts/01..03 antes")


@pytest.fixture
def quatro_subsistemas():
    datas = pd.date_range("2024-01-01", periods=10, freq="D")
    cargas = {"N": 7000.0, "NE": 12000.0, "S": 11000.0, "SE": 45000.0}
    temps = {"N": 27.0, "NE": 26.0, "S": 19.0, "SE": 23.0}
    linhas = []
    for sub in SUBSISTEMAS:
        linhas.append(pd.DataFrame({
            "din_instante": datas,
            "id_subsistema": sub,
            ALVO: cargas[sub],
            "temp_mean": temps[sub],
            "industrial_mwmed": cargas[sub] * 0.3,
            "n_estacoes": 50.0,
        }))
    return pd.concat(linhas, ignore_index=True)


# --- agregacao --------------------------------------------------------------

def test_carga_do_sin_e_a_soma_dos_subsistemas(quatro_subsistemas):
    sin = agregar_sin(quatro_subsistemas, pesos_do_sin(quatro_subsistemas))
    assert sin[ALVO].to_numpy() == pytest.approx(7000 + 12000 + 11000 + 45000)
    assert (sin["id_subsistema"] == SIN).all()


def test_industrial_e_estacoes_tambem_somam(quatro_subsistemas):
    sin = agregar_sin(quatro_subsistemas, pesos_do_sin(quatro_subsistemas))
    assert sin["industrial_mwmed"].iloc[0] == pytest.approx(75000 * 0.3)
    assert sin["n_estacoes"].iloc[0] == pytest.approx(200.0)


def test_clima_do_sin_e_ponderado_pela_carga_nao_media_simples(quatro_subsistemas):
    """Media simples daria 23,75 C; ponderada puxa para o Sudeste."""
    pesos = pesos_do_sin(quatro_subsistemas)
    sin = agregar_sin(quatro_subsistemas, pesos)
    simples = np.mean([27.0, 26.0, 19.0, 23.0])
    ponderada = sum(pesos[s] * t for s, t in
                    zip(["N", "NE", "S", "SE"], [27.0, 26.0, 19.0, 23.0]))
    obtida = sin["temp_mean"].iloc[0]
    assert obtida == pytest.approx(ponderada)
    assert obtida != pytest.approx(simples)


def test_pesos_somam_um_e_sudeste_domina(quatro_subsistemas):
    pesos = pesos_do_sin(quatro_subsistemas)
    assert pesos.sum() == pytest.approx(1.0)
    assert pesos["SE"] > pesos["N"]
    assert pesos.idxmax() == "SE"


def test_pesos_vem_de_um_ano_fixo_e_nao_do_ano_previsto():
    """Ponderar pela carga corrente vazaria o alvo; o peso e do ano de treino."""
    datas = pd.concat([pd.Series(pd.date_range("2024-01-01", periods=5)),
                       pd.Series(pd.date_range("2025-01-01", periods=5))])
    linhas = []
    for sub, c24, c25 in [("N", 1000.0, 90000.0), ("SE", 9000.0, 1000.0),
                          ("NE", 1000.0, 1000.0), ("S", 1000.0, 1000.0)]:
        linhas.append(pd.DataFrame({
            "din_instante": datas.values,
            "id_subsistema": sub,
            ALVO: [c24] * 5 + [c25] * 5,
        }))
    df = pd.concat(linhas, ignore_index=True)
    pesos = pesos_do_sin(df, ano_referencia=2024)
    # Em 2024 o SE domina; em 2025 seria o N. O peso tem de refletir 2024.
    assert pesos.idxmax() == "SE"


def test_dia_incompleto_nao_vira_linha_do_sin(quatro_subsistemas):
    """Sem os quatro subsistemas, o total nacional seria subestimado."""
    faltando = quatro_subsistemas[
        ~((quatro_subsistemas["id_subsistema"] == "SE")
          & (quatro_subsistemas["din_instante"] == pd.Timestamp("2024-01-05")))]
    sin = agregar_sin(faltando, pesos_do_sin(quatro_subsistemas))
    assert pd.Timestamp("2024-01-05") not in set(sin["din_instante"])
    assert len(sin) == 9


# --- SIN como entrada -------------------------------------------------------

@pytest.mark.parametrize("rotulo", ["SIN", "sin", "Brasil", "Nacional",
                                    "Sistema Interligado Nacional", "BR"])
def test_sin_aceito_em_planilha(rotulo):
    d = normalizar(pd.DataFrame({"regiao": [rotulo], "data": ["2026-01-15"]}))
    assert d["id_subsistema"].iloc[0] == SIN


def test_sin_esta_entre_as_regioes_modeladas():
    assert SIN in NOMES_REGIOES
    assert SIN not in SUBSISTEMAS          # nao vem do arquivo do ONS


@precisa_artefatos
def test_preve_o_sin(quatro_subsistemas):
    e = pd.DataFrame({"id_subsistema": [SIN], "din_instante": ["2026-02-10"]})
    r = pv.prever(e)
    assert r["previsao_mwmed"].notna().all()
    assert 50_000 < r["previsao_mwmed"].iloc[0] < 120_000


@precisa_artefatos
def test_sin_e_maior_que_qualquer_subsistema():
    e = pd.DataFrame({"id_subsistema": [SIN] + SUBSISTEMAS,
                      "din_instante": ["2026-02-10"] * 5})
    r = pv.prever(e).set_index("id_subsistema")["previsao_mwmed"]
    assert (r[SIN] > r[SUBSISTEMAS]).all()


# --- climatologia -----------------------------------------------------------

@precisa_artefatos
def test_climatologia_cobre_o_ano_todo_de_cada_regiao():
    c = pv.climatologia()
    for sub in list(NOMES_REGIOES):
        dias = c[c["id_subsistema"] == sub]["dia_ano"]
        assert dias.min() == 1 and dias.max() >= 365
    assert c["temp_mean"].notna().all()


@precisa_artefatos
def test_completar_preenche_os_dias_que_faltam():
    parcial = pd.DataFrame({"id_subsistema": [SIN] * 3,
                            "din_instante": ["2026-01-10", "2026-01-11", "2026-01-12"],
                            "temp_mean": [24.0, 25.0, 26.0]})
    cheio = pv.completar_periodo(parcial, ate="2026-12-31")
    assert len(cheio) == 356                      # 10/jan a 31/dez
    assert (cheio["origem"] == "arquivo").sum() == 3
    assert cheio["temp_mean"].notna().all()


@precisa_artefatos
def test_completar_preserva_o_clima_informado():
    parcial = pd.DataFrame({"id_subsistema": ["SE"], "din_instante": ["2026-03-01"],
                            "temp_mean": [33.3]})
    cheio = pv.completar_periodo(parcial, ate="2026-03-10")
    linha = cheio[cheio["din_instante"] == pd.Timestamp("2026-03-01")].iloc[0]
    assert linha["temp_mean"] == pytest.approx(33.3)
    assert linha["origem"] == "arquivo"
    outra = cheio[cheio["din_instante"] == pd.Timestamp("2026-03-05")].iloc[0]
    assert outra["origem"] == "climatologia"
    assert outra["temp_mean"] != pytest.approx(33.3)


@precisa_artefatos
def test_completar_com_varias_regioes():
    parcial = pd.DataFrame({"id_subsistema": [SIN, "SE", "N"],
                            "din_instante": ["2026-01-05"] * 3})
    cheio = pv.completar_periodo(parcial, ate="2026-01-31")
    assert set(cheio["id_subsistema"]) == {SIN, "SE", "N"}
    assert cheio.groupby("id_subsistema").size().nunique() == 1


@precisa_artefatos
def test_completar_alimenta_a_previsao():
    parcial = pd.DataFrame({"id_subsistema": [SIN], "din_instante": ["2026-01-01"]})
    r = pv.prever(pv.completar_periodo(parcial, ate="2026-12-31"))
    assert len(r) == 365
    assert r["previsao_mwmed"].notna().all()
    # Energia anual do SIN na casa de centenas de TWh
    twh = r["previsao_mwmed"].sum() * 24 / 1e6
    assert 400 < twh < 900


@precisa_artefatos
def test_data_final_anterior_a_inicial_da_erro():
    parcial = pd.DataFrame({"id_subsistema": ["SE"], "din_instante": ["2026-06-01"]})
    with pytest.raises(ValueError, match="anterior"):
        pv.completar_periodo(parcial, ate="2026-01-01")


# --- projecao anual ---------------------------------------------------------

@precisa_artefatos
def test_projetar_ano_cobre_todos_os_dias_e_regioes():
    d = pv.projetar_ano(2026)
    assert set(d["id_subsistema"]) == set(NOMES_REGIOES)
    assert (d.groupby("id_subsistema").size() == 365).all()
    assert d["previsao_mwmed"].notna().all()


@precisa_artefatos
def test_projetar_ano_bissexto_tem_366_dias():
    d = pv.projetar_ano(2028, regioes=["SE"])
    assert len(d) == 366


@precisa_artefatos
def test_resumo_anual_tem_mwmed_e_energia_coerentes():
    d = pv.projetar_ano(2026, regioes=[SIN])
    r = pv.resumo_periodo(d).iloc[0]
    assert r["dias"] == 365
    assert r["mwmed_min"] <= r["mwmed_medio"] <= r["mwmed_max"]
    # energia = MWmed medio x 24 h x dias
    assert r["energia_TWh"] == pytest.approx(
        r["mwmed_medio"] * 24 * 365 / 1e6, rel=1e-6)
    assert 400 < r["energia_TWh"] < 900
    assert 0 < r["fator_carga"] <= 1


@precisa_artefatos
def test_clima_informado_muda_a_media_anual():
    """Se o clima nao alterasse o resultado, nao seria previsao climatica."""
    grade = pd.date_range("2026-01-01", "2026-12-31", freq="D")
    quente = pd.DataFrame({"id_subsistema": "SE", "din_instante": grade,
                           "temp_mean": 29.0, "temp_max": 34.0, "temp_min": 24.0})
    ameno = quente.assign(temp_mean=19.0, temp_max=24.0, temp_min=14.0)

    m_quente = pv.resumo_periodo(pv.projetar_ano(2026, ["SE"], quente))["mwmed_medio"][0]
    m_ameno = pv.resumo_periodo(pv.projetar_ano(2026, ["SE"], ameno))["mwmed_medio"][0]
    assert m_quente > m_ameno, "ano quente deveria ter carga media maior no SE"


@precisa_artefatos
def test_projecao_anual_bate_o_observado_dentro_de_10pct():
    """Backtest: 2025 projetado com o clima real, contra a media observada."""
    h = pv.historico()
    do_ano = h[h["din_instante"].dt.year == 2025]
    cols = [c for c in pv.CAMPOS_CLIMA if c in do_ano.columns]
    clima = do_ano[["id_subsistema", "din_instante"] + cols]

    r = pv.resumo_periodo(pv.projetar_ano(2025, clima=clima)).set_index("id_subsistema")
    real = pv.mwmed_anual_observado(2025)
    for sub in SUBSISTEMAS + [SIN]:
        erro = abs(r.loc[sub, "mwmed_medio"] / real[sub] - 1)
        assert erro < 0.10, f"{sub} errou {erro:.1%} na media anual"


@precisa_artefatos
def test_mwmed_observado_bate_com_o_historico():
    real = pv.mwmed_anual_observado(2025)
    h = pv.historico()
    esperado = h[(h["din_instante"].dt.year == 2025)
                 & (h["id_subsistema"] == SIN)][ALVO].mean()
    assert real[SIN] == pytest.approx(esperado)


# --- resumo de periodo arbitrario e confronto com o observado ---------------

@precisa_artefatos
def test_resumo_funciona_para_periodo_parcial():
    """Arquivo curto tem de render os mesmos indicadores de um ano inteiro."""
    grade = pd.date_range("2026-03-01", "2026-04-30", freq="D")
    e = pd.DataFrame({"id_subsistema": "SE", "din_instante": grade})
    r = pv.resumo_periodo(pv.prever(e)).iloc[0]
    assert r["dias"] == len(grade)
    assert r["mwmed_min"] <= r["mwmed_medio"] <= r["mwmed_max"]
    assert r["energia_TWh"] == pytest.approx(
        r["mwmed_medio"] * 24 * len(grade) / 1e6, rel=1e-6)
    assert pd.Timestamp(r["de"]) == grade[0]
    assert pd.Timestamp(r["ate"]) == grade[-1]


@precisa_artefatos
def test_resumo_conta_dias_do_arquivo_e_estimados():
    parcial = pd.DataFrame({"id_subsistema": ["SE"] * 3,
                            "din_instante": ["2026-01-01", "2026-01-02", "2026-01-03"],
                            "temp_mean": [25.0, 26.0, 27.0]})
    r = pv.resumo_periodo(pv.prever(pv.completar_periodo(parcial, ate="2026-01-31")))
    linha = r.iloc[0]
    assert linha["dias_do_arquivo"] == 3
    assert linha["dias_estimados"] == 28
    assert linha["dias"] == 31


@precisa_artefatos
def test_resumo_traz_a_decomposicao_media():
    e = pd.DataFrame({"id_subsistema": ["SE"] * 5,
                      "din_instante": pd.date_range("2026-05-01", periods=5)})
    r = pv.resumo_periodo(pv.prever(e)).iloc[0]
    assert "nivel_medio" in r and "resposta_climatica_media" in r
    assert r["nivel_medio"] + r["resposta_climatica_media"] == pytest.approx(
        r["mwmed_medio"], rel=1e-6)


@precisa_artefatos
def test_comparacao_com_observado_em_periodo_passado():
    grade = pd.date_range("2025-06-01", "2025-06-30", freq="D")
    e = pd.DataFrame({"id_subsistema": "SE", "din_instante": grade})
    c = pv.comparar_com_observado(pv.prever(e))
    assert len(c) == 1
    assert c["dias_comparados"].iloc[0] == 30
    assert c["mape_diario_%"].iloc[0] < 15
    assert abs(c["erro_medio_%"].iloc[0]) < 15


@precisa_artefatos
def test_comparacao_vazia_para_periodo_futuro():
    """Ano sem observacao nao pode inventar comparacao."""
    e = pd.DataFrame({"id_subsistema": ["SE"] * 3,
                      "din_instante": pd.date_range("2030-01-01", periods=3)})
    assert pv.comparar_com_observado(pv.prever(e)).empty
