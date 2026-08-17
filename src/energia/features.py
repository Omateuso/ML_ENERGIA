"""Montagem do dataset de modelagem: carga (ONS) + clima (INMET) + industrial (EPE).

As features sao declaradas em grupos porque cada modo de previsao usa um
subconjunto diferente. O grupo AUTOREGRESSIVO e o unico que le a carga passada;
o modo climatico o exclui por completo, o que torna impossivel contaminar a
previsao com a carga do ano que serve de gabarito.
"""
from __future__ import annotations

from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from .clima import carregar_clima
from .config import (ANO_TREINO, BRUTO_INMET, BRUTO_ONS, PROCESSADO, SIN,
                     SUBSISTEMAS)
from .industrial import COLUNAS_INDUSTRIAL, nivel_industrial_diario

CALENDARIO = ["mes", "dia_semana", "trimestre", "is_feriado", "is_fds",
              "is_vespera_feriado", "dia_ano_sen", "dia_ano_cos"]

CLIMA = ["temp_mean", "temp_max", "temp_min", "umid_mean", "umid_min",
         "rad_sum", "vento_mean", "precip_sum", "graus_refrig", "graus_aquec",
         "horas_acima_26", "graus_refrig_med3", "temp_mean_ontem",
         "amplitude_termica"]

INDUSTRIAL = list(COLUNAS_INDUSTRIAL)

# So estes tocam a carga observada. Legitimos para previsao D+1 (ao prever
# amanha voce conhece hoje), proibidos no modo climatico.
AUTOREGRESSIVO = ["carga_ontem", "carga_lag7", "media_7d"]

ALVO = "val_cargaenergiamwmed"


def carregar_carga(anos: list[int]) -> pd.DataFrame:
    partes = []
    for ano in anos:
        arq = BRUTO_ONS / f"CARGA_ENERGIA_{ano}.csv"
        if not arq.exists():
            raise FileNotFoundError(f"{arq} nao existe. Rode scripts/01_coletar.py")
        partes.append(pd.read_csv(arq, sep=";"))
    df = pd.concat(partes, ignore_index=True)
    df["din_instante"] = pd.to_datetime(df["din_instante"])
    df = df[df["id_subsistema"].isin(SUBSISTEMAS)]
    return df.sort_values(["id_subsistema", "din_instante"], ignore_index=True)


# Somam-se: sao grandezas extensivas (MWmed, contagem de estacoes).
SOMAVEIS = ["val_cargaenergiamwmed", "industrial_mwmed", "n_estacoes"]


def pesos_do_sin(df: pd.DataFrame, ano_referencia: int = ANO_TREINO) -> pd.Series:
    """Participacao de cada subsistema na carga nacional, em um ano fixo.

    Usados para ponderar o clima do SIN. Precisam vir de um ano FIXO e passado:
    ponderar pela carga corrente do dia previsto seria vazar o alvo.
    """
    base = df[df["din_instante"].dt.year == ano_referencia]
    if base.empty:
        base = df
    media = base.groupby("id_subsistema")[ALVO].mean()
    return media / media.sum()


def agregar_sin(df: pd.DataFrame, pesos: pd.Series) -> pd.DataFrame:
    """Cria as linhas do SIN a partir dos quatro subsistemas.

    Carga e patamar industrial somam. O clima e media ponderada pela
    participacao na carga -- media simples daria ao Norte (7 GW) o mesmo peso
    do Sudeste (45 GW), e o clima nacional resultante nao explicaria nada.
    """
    sub = df[df["id_subsistema"].isin(SUBSISTEMAS)].copy()
    sub["_peso"] = sub["id_subsistema"].map(pesos).astype(float)

    somaveis = [c for c in SOMAVEIS if c in sub.columns]
    ponderaveis = [c for c in sub.columns
                   if c not in somaveis
                   and c not in ("din_instante", "id_subsistema", "nom_subsistema",
                                 "data_clima", "_peso")
                   and pd.api.types.is_numeric_dtype(sub[c])]

    g = sub.groupby("din_instante")
    saida = g[somaveis].sum()

    # Renormaliza o peso dia a dia, para o caso de faltar algum subsistema.
    peso_total = g["_peso"].transform("sum")
    for col in ponderaveis:
        contrib = (sub[col] * sub["_peso"] / peso_total).groupby(sub["din_instante"])
        saida[col] = contrib.sum()
        # Onde nenhum subsistema tinha o valor, o resultado e ausente, nao zero.
        saida.loc[g[col].count() == 0, col] = np.nan

    saida = saida.reset_index()
    saida["id_subsistema"] = SIN
    saida["nom_subsistema"] = "SIN"
    # Um dia so vale para o SIN se os quatro subsistemas estiverem presentes.
    completos = g["id_subsistema"].nunique()
    saida = saida[saida["din_instante"].map(completos) == len(SUBSISTEMAS)]
    return saida


def _calendario(df: pd.DataFrame) -> pd.DataFrame:
    d = df["din_instante"]
    df["mes"] = d.dt.month
    df["dia_semana"] = d.dt.dayofweek
    df["trimestre"] = d.dt.quarter
    df["ano"] = d.dt.year

    feriados = holidays.Brazil(years=sorted(d.dt.year.unique()))
    datas = d.dt.date
    df["is_feriado"] = datas.isin(feriados).astype(int)
    df["is_vespera_feriado"] = (d + pd.Timedelta(days=1)).dt.date.isin(feriados).astype(int)
    df["is_fds"] = (df["dia_semana"] >= 5).astype(int)

    # Sazonalidade continua: dia 365 e vizinho do dia 1, o que o inteiro
    # "dia do ano" nao expressa.
    ang = 2 * np.pi * d.dt.dayofyear / 365.25
    df["dia_ano_sen"] = np.sin(ang)
    df["dia_ano_cos"] = np.cos(ang)
    return df


def _clima_derivado(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("id_subsistema")
    df["amplitude_termica"] = df["temp_max"] - df["temp_min"]
    # Inercia termica: predio que aqueceu por 3 dias puxa mais ar condicionado
    # do que um dia quente isolado.
    df["graus_refrig_med3"] = g["graus_refrig"].transform(
        lambda x: x.rolling(3, min_periods=1).mean())
    df["temp_mean_ontem"] = g["temp_mean"].shift(1)
    return df


def _autoregressivo(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("id_subsistema")[ALVO]
    df["carga_ontem"] = g.shift(1)
    df["carga_lag7"] = g.shift(7)
    # shift(1) ANTES do rolling: sem ele a media de 7 dias incluiria o proprio
    # dia que estamos prevendo.
    df["media_7d"] = g.transform(lambda x: x.shift(1).rolling(7).mean())
    return df


def construir_dataset(anos: list[int], usar_cache: bool = True,
                      defasagem_industrial: int = 2) -> pd.DataFrame:
    print("Montando dataset de modelagem")
    carga = carregar_carga(anos)
    print(f"  carga: {len(carga)} linhas, {carga['din_instante'].min().date()} "
          f"a {carga['din_instante'].max().date()}")

    clima = carregar_clima([BRUTO_INMET / str(a) for a in anos], usar_cache)
    clima = clima.rename(columns={"data": "din_instante"})

    df = carga.merge(clima, on=["din_instante", "id_subsistema"], how="left")
    faltando = df[["temp_mean"]].isna().sum().item()
    print(f"  clima: {len(clima)} linhas; {faltando} dias-regiao sem clima")

    ind = nivel_industrial_diario(df["din_instante"], defasagem_industrial)
    df = df.merge(ind, on=["din_instante", "id_subsistema"], how="left")
    print(f"  industrial: defasagem de {defasagem_industrial} meses aplicada")

    # O SIN entra aqui, antes das features derivadas, para receber os mesmos
    # lags e graus-dia que os subsistemas.
    pesos = pesos_do_sin(df)
    df = pd.concat([df, agregar_sin(df, pesos)], ignore_index=True)
    print("  SIN: agregado dos 4 subsistemas; clima ponderado por "
          + ", ".join(f"{s} {p:.0%}" for s, p in pesos.items()))

    df = df.sort_values(["id_subsistema", "din_instante"], ignore_index=True)
    df = _calendario(df)
    df = _clima_derivado(df)
    df = _autoregressivo(df)

    PROCESSADO.mkdir(parents=True, exist_ok=True)
    destino = PROCESSADO / "dataset_modelo.parquet"
    df.to_parquet(destino, index=False)
    print(f"  salvo: {destino} ({len(df)} linhas, {df.shape[1]} colunas)")
    return df


def features_do_modo(modo: str, df: pd.DataFrame) -> list[str]:
    """Colunas usadas em cada modo, filtradas pelo que existe no dataset."""
    if modo == "climatico":
        cols = CALENDARIO + CLIMA + INDUSTRIAL
    elif modo in ("delta_d1", "recursivo"):
        cols = CALENDARIO + CLIMA + INDUSTRIAL + AUTOREGRESSIVO
    else:
        raise ValueError(f"modo desconhecido: {modo}")
    return [c for c in cols if c in df.columns]
