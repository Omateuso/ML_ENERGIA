"""INMET horario por estacao -> clima diario agregado por subsistema do ONS.

Duas decisoes importantes aqui:

1. As estacoes do Centro-Oeste sao remapeadas para o subsistema SE. O INMET usa
   as 5 macrorregioes do IBGE, o ONS opera 4 subsistemas, e o "Sudeste" do ONS
   e na verdade "Sudeste/Centro-Oeste".

2. Graus-dia sao calculados por estacao e so depois promediados. Como
   max(0, T - 22) e nao-linear, calcula-lo sobre a media regional apagaria a
   contribuicao das estacoes quentes -- exatamente o sinal que move o ar
   condicionado.
"""
from __future__ import annotations

import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (INTERIM, LIMITES_INMET, REGIAO_INMET_PARA_SUBSISTEMA,
                     TEMP_BASE_AQUECIMENTO, TEMP_BASE_REFRIGERACAO)

# Prefixo normalizado do cabecalho -> nome curto usado no projeto.
# O INMET varia acentuacao e caixa entre anos, entao a busca e por prefixo
# do nome normalizado (sem acento, maiusculo).
PREFIXOS = {
    "TEMPERATURA DO AR": "temp",
    "UMIDADE RELATIVA DO AR": "umid",
    "RADIACAO GLOBAL": "rad",
    "VENTO, VELOCIDADE": "vento",
    "PRECIPITACAO TOTAL": "precip",
}


def _normalizar(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.upper().split())


def _mapear_colunas(colunas) -> dict[str, str]:
    achadas = {}
    for col in colunas:
        n = _normalizar(col)
        for prefixo, curto in PREFIXOS.items():
            if n.startswith(prefixo) and curto not in achadas:
                achadas[curto] = col
    return achadas


def _ler_estacao(arquivo: Path) -> pd.DataFrame | None:
    """Le um CSV de estacao e devolve as estatisticas diarias dela."""
    with open(arquivo, "r", encoding="latin-1") as f:
        cabecalho = [f.readline() for _ in range(8)]
    try:
        regiao_inmet = cabecalho[0].split(";")[1].strip().upper()
        codigo = cabecalho[3].split(";")[1].strip()
    except IndexError:
        return None

    subsistema = REGIAO_INMET_PARA_SUBSISTEMA.get(regiao_inmet)
    if subsistema is None:
        return None

    df = pd.read_csv(arquivo, sep=";", encoding="latin-1", skiprows=8,
                     decimal=",", na_values=["", "-9999", "-9999.0"],
                     low_memory=False)
    cols = _mapear_colunas(df.columns)
    if "temp" not in cols or "umid" not in cols:
        return None

    data = pd.to_datetime(df.iloc[:, 0].astype(str).str.replace("/", "-"),
                          errors="coerce")
    dados = {"data": data}
    for curto, original in cols.items():
        v = pd.to_numeric(df[original], errors="coerce")
        if curto in LIMITES_INMET:            # descarta leitura de sensor quebrado
            lo, hi = LIMITES_INMET[curto]
            v = v.where((v >= lo) & (v <= hi))
        dados[curto] = v
    d = pd.DataFrame(dados).dropna(subset=["data"])
    if d.empty:
        return None

    agr = {"temp": ["mean", "max", "min"], "umid": ["mean", "min"]}
    for opc in ("rad", "vento", "precip"):
        if opc in d.columns:
            agr[opc] = ["sum" if opc in ("rad", "precip") else "mean"]

    g = d.groupby("data").agg(agr)
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g = g.reset_index()

    # Graus-dia sobre a media diaria DESTA estacao (antes de agregar a regiao).
    g["graus_refrig"] = (g["temp_mean"] - TEMP_BASE_REFRIGERACAO).clip(lower=0)
    g["graus_aquec"] = (TEMP_BASE_AQUECIMENTO - g["temp_mean"]).clip(lower=0)
    # Horas do dia acima de 26 C: mede duracao do calor, nao so intensidade.
    horas_quentes = (d.assign(q=(d["temp"] > 26).astype(float))
                       .groupby("data")["q"].sum().rename("horas_acima_26"))
    g = g.merge(horas_quentes, on="data", how="left")

    # Exige cobertura horaria minima para o dia valer (evita dia com 2 leituras).
    cobertura = d.groupby("data")["temp"].count().rename("n_horas")
    g = g.merge(cobertura, on="data", how="left")
    g = g[g["n_horas"] >= 18].drop(columns="n_horas")

    g["id_subsistema"] = subsistema
    g["estacao"] = codigo
    return g


COLUNAS_CLIMA = ["temp_mean", "temp_max", "temp_min", "umid_mean", "umid_min",
                 "rad_sum", "vento_mean", "precip_sum", "graus_refrig",
                 "graus_aquec", "horas_acima_26"]


def clima_diario_por_subsistema(pasta: Path, usar_cache: bool = True) -> pd.DataFrame:
    """Agrega todas as estacoes de uma pasta anual em clima diario por subsistema."""
    cache = INTERIM / f"clima_{pasta.name}.parquet"
    if usar_cache and cache.exists():
        print(f"  clima {pasta.name}: cache")
        return pd.read_parquet(cache)

    arquivos = sorted(pasta.glob("*.CSV"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum CSV de estacao em {pasta}")

    print(f"  clima {pasta.name}: lendo {len(arquivos)} estacoes...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        partes = list(ex.map(_ler_estacao, arquivos))

    validas = [p for p in partes if p is not None and not p.empty]
    descartadas = len(arquivos) - len(validas)
    if not validas:
        raise ValueError(f"Nenhuma estacao valida em {pasta}")

    todas = pd.concat(validas, ignore_index=True)
    # Uma estacao nao pode aparecer duas vezes no mesmo dia (zips republicados).
    todas = todas.drop_duplicates(subset=["estacao", "data"], keep="last")

    presentes = [c for c in COLUNAS_CLIMA if c in todas.columns]
    reg = (todas.groupby(["data", "id_subsistema"])[presentes]
                .mean().reset_index())
    reg["n_estacoes"] = (todas.groupby(["data", "id_subsistema"])["estacao"]
                              .nunique().values)

    print(f"  clima {pasta.name}: {len(validas)} estacoes ok, {descartadas} descartadas, "
          f"{len(reg)} linhas (data x subsistema)")
    INTERIM.mkdir(parents=True, exist_ok=True)
    reg.to_parquet(cache, index=False)
    return reg


def carregar_clima(pastas: list[Path], usar_cache: bool = True) -> pd.DataFrame:
    return pd.concat([clima_diario_por_subsistema(p, usar_cache) for p in pastas],
                     ignore_index=True)
