"""Testes do processamento de clima e do nivel industrial."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energia import clima as cl
from energia import industrial as ind
from energia.config import (LIMITES_INMET, TEMP_BASE_AQUECIMENTO,
                            TEMP_BASE_REFRIGERACAO)

CABECALHO = """REGIAO:;{regiao}
UF:;PA
ESTACAO:;TESTE
CODIGO (WMO):;{codigo}
LATITUDE:;-1,5
LONGITUDE:;-48,5
ALTITUDE:;10,0
DATA DE FUNDACAO:;01/01/00
"""


def escrever_estacao(pasta: Path, regiao="N", codigo="A001", temp=25.0,
                     horas=24, dias=3) -> Path:
    linhas = ["Data;Hora UTC;TEMPERATURA DO AR - BULBO SECO, HORARIA (°C);"
              "UMIDADE RELATIVA DO AR, HORARIA (%);VENTO, VELOCIDADE HORARIA (m/s)"]
    for d in range(dias):
        for h in range(horas):
            data = f"2024/01/{d + 1:02d}"
            linhas.append(f"{data};{h:02d}00 UTC;"
                          f"{str(temp).replace('.', ',')};70;2,0")
    arq = pasta / f"INMET_{regiao}_PA_{codigo}_TESTE.CSV"
    arq.write_text(CABECALHO.format(regiao=regiao, codigo=codigo)
                   + "\n".join(linhas), encoding="latin-1")
    return arq


def test_normalizacao_de_cabecalho_ignora_acento_e_caixa():
    assert cl._normalizar("TEMPERATURA DO AR - BULBO SECO, HORÁRIA (°C)").startswith(
        "TEMPERATURA DO AR")
    assert cl._mapear_colunas(["RADIACAO GLOBAL (Kj/m²)"])["rad"] == "RADIACAO GLOBAL (Kj/m²)"


def test_estacao_do_centro_oeste_e_lida_como_sudeste(tmp_path):
    escrever_estacao(tmp_path, regiao="CO")
    d = cl._ler_estacao(next(tmp_path.glob("*.CSV")))
    assert d is not None
    assert (d["id_subsistema"] == "SE").all()


def test_graus_dia_de_refrigeracao_e_aquecimento(tmp_path):
    escrever_estacao(tmp_path, temp=30.0, codigo="A100")
    d = cl._ler_estacao(next(tmp_path.glob("*.CSV")))
    assert d["graus_refrig"].iloc[0] == pytest.approx(30.0 - TEMP_BASE_REFRIGERACAO)
    assert d["graus_aquec"].iloc[0] == pytest.approx(0.0)      # nunca negativo
    assert d["horas_acima_26"].iloc[0] == pytest.approx(24.0)


def test_dia_com_poucas_horas_e_descartado(tmp_path):
    escrever_estacao(tmp_path, horas=6)                        # abaixo do minimo de 18
    d = cl._ler_estacao(next(tmp_path.glob("*.CSV")))
    assert d is None or d.empty


def test_leitura_fora_dos_limites_fisicos_vira_nan(tmp_path):
    escrever_estacao(tmp_path, temp=999.0)
    d = cl._ler_estacao(next(tmp_path.glob("*.CSV")))
    lo, hi = LIMITES_INMET["temp"]
    if d is not None and not d.empty:
        assert d["temp_mean"].isna().all() or ((d["temp_mean"] >= lo)
                                               & (d["temp_mean"] <= hi)).all()


def test_regiao_desconhecida_e_ignorada(tmp_path):
    escrever_estacao(tmp_path, regiao="XX")
    assert cl._ler_estacao(next(tmp_path.glob("*.CSV"))) is None


# --- industrial -------------------------------------------------------------

def _mensal_falso():
    linhas = []
    for ano in (2023, 2024, 2025):
        for mes in range(1, 13):
            for sub, base in (("N", 2_000_000.0), ("SE", 9_000_000.0)):
                linhas.append({"ano": ano, "mes": mes, "id_subsistema": sub,
                               "industrial_mwh": base + mes * 1000})
    return pd.DataFrame(linhas)


def test_defasagem_industrial_atrasa_a_serie(monkeypatch):
    monkeypatch.setattr(ind, "ler_industrial_mensal", lambda *a, **k: _mensal_falso())
    datas = pd.Series(pd.date_range("2025-01-01", "2025-12-31", freq="D"))

    sem = ind.nivel_industrial_diario(datas, defasagem_meses=0)
    com = ind.nivel_industrial_diario(datas, defasagem_meses=2)

    a = sem[sem["id_subsistema"] == "N"].set_index("din_instante")["industrial_mwmed"]
    b = com[com["id_subsistema"] == "N"].set_index("din_instante")["industrial_mwmed"]
    # A serie defasada precisa ser diferente -- se for igual, a protecao nao aplicou.
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_conversao_para_mwmed(monkeypatch):
    monkeypatch.setattr(ind, "ler_industrial_mensal", lambda *a, **k: _mensal_falso())
    datas = pd.Series(pd.date_range("2025-06-01", "2025-06-30", freq="D"))
    d = ind.nivel_industrial_diario(datas, defasagem_meses=0)
    # MWh/mes dividido por horas do mes cai na casa de milhares de MWmed
    n = d[d["id_subsistema"] == "N"]["industrial_mwmed"]
    assert (n > 1000).all() and (n < 10000).all()


def test_rotulos_de_subsistema_da_epe():
    assert ind.ROTULO_PARA_SUBSISTEMA["SUDESTE/C. OESTE"] == "SE"
    assert ind.ROTULO_PARA_SUBSISTEMA["NORTE"] == "N"
    assert "SISTEMAS ISOLADOS" not in ind.ROTULO_PARA_SUBSISTEMA
