"""API de previsao para uso interativo (interface grafica, notebooks, scripts).

O usuario informa poucas variaveis de clima; tudo o que o modelo precisa e
derivado ou preenchido com a mediana historica da regiao. Nenhuma carga
observada entra em nenhum ponto -- estas funcoes usam so o modo `climatico`.
"""
from __future__ import annotations

import functools

import holidays
import joblib
import numpy as np
import pandas as pd

from .config import (MODELOS, NOMES_REGIOES, PROCESSADO,
                     TEMP_BASE_AQUECIMENTO, TEMP_BASE_REFRIGERACAO)
from .features import ALVO
from .industrial import nivel_industrial_diario

# O que a interface pede ao usuario. O resto tem valor padrao.
CAMPOS_CLIMA = ["temp_mean", "temp_max", "temp_min", "umid_mean", "umid_min",
                "rad_sum", "vento_mean", "precip_sum"]
CAMPOS_PEDIDOS = ["temp_mean", "temp_max", "temp_min", "umid_mean"]

ROTULOS = {
    "temp_mean": "Temperatura média (°C)",
    "temp_max": "Temperatura máxima (°C)",
    "temp_min": "Temperatura mínima (°C)",
    "umid_mean": "Umidade relativa média (%)",
    "umid_min": "Umidade relativa mínima (%)",
    "rad_sum": "Radiação global acumulada (kJ/m²)",
    "vento_mean": "Velocidade média do vento (m/s)",
    "precip_sum": "Precipitação acumulada (mm)",
}


class DadosAusentes(Exception):
    """O dataset processado ou os modelos ainda nao foram gerados."""


@functools.lru_cache(maxsize=1)
def historico() -> pd.DataFrame:
    caminho = PROCESSADO / "dataset_modelo.parquet"
    if not caminho.exists():
        raise DadosAusentes(
            "data/processed/dataset_modelo.parquet nao existe. "
            "Rode scripts/01_coletar.py e scripts/02_construir_dataset.py.")
    return pd.read_parquet(caminho)


@functools.lru_cache(maxsize=1)
def padroes_regionais() -> dict[str, dict[str, float]]:
    """Mediana historica de cada variavel de clima, por subsistema."""
    h = historico()
    return {sub: {c: float(g[c].median()) for c in CAMPOS_CLIMA if c in g.columns}
            for sub, g in h.groupby("id_subsistema")}


@functools.lru_cache(maxsize=8)
def carregar_modelo(regiao: str, modo: str = "climatico"):
    caminho = MODELOS / f"{modo}_{regiao}.joblib"
    if not caminho.exists():
        raise DadosAusentes(f"{caminho.name} nao existe. Rode scripts/03_treinar.py.")
    return joblib.load(caminho)


def _horas_acima(limiar: float, t_med: float, t_max: float, t_min: float) -> float:
    """Horas do dia acima de um limiar, supondo ciclo diario senoidal.

    Evita pedir ao usuario uma variavel que ele nao teria como estimar, e e
    fisicamente mais defensavel que um valor fixo.
    """
    amp = max((t_max - t_min) / 2.0, 0.01)
    x = np.clip((limiar - t_med) / amp, -1.0, 1.0)
    return float(24.0 * (np.pi - 2 * np.arcsin(x)) / (2 * np.pi))


def _industrial(datas: pd.Series, defasagem_meses: int = 2) -> pd.DataFrame:
    """Patamar industrial nas datas pedidas.

    A serie e montada sobre uma janela alargada para tras porque a variacao
    anual precisa de 365 dias de historico para existir.
    """
    datas = pd.to_datetime(pd.Series(datas))
    janela = pd.Series(pd.date_range(datas.min() - pd.Timedelta(days=400),
                                     datas.max(), freq="D"))
    return nivel_industrial_diario(janela, defasagem_meses)


def montar_features(entradas: pd.DataFrame, defasagem_meses: int = 2) -> pd.DataFrame:
    """Constroi o quadro completo de features a partir de entradas minimas.

    `entradas` precisa de `id_subsistema` e `din_instante`; qualquer coluna de
    CAMPOS_CLIMA ausente e preenchida com a mediana da regiao.
    """
    df = entradas.copy()
    df["din_instante"] = pd.to_datetime(df["din_instante"])
    if "id_subsistema" not in df.columns:
        raise ValueError("coluna 'id_subsistema' ausente")
    invalidas = set(df["id_subsistema"]) - set(NOMES_REGIOES)
    if invalidas:
        raise ValueError(f"subsistema desconhecido: {sorted(invalidas)}. "
                         f"Use um de {list(NOMES_REGIOES)}")

    padroes = padroes_regionais()
    for col in CAMPOS_CLIMA:
        preenchido = df["id_subsistema"].map(lambda s: padroes.get(s, {}).get(col))
        df[col] = pd.to_numeric(df.get(col), errors="coerce") if col in df.columns else np.nan
        df[col] = df[col].fillna(preenchido)

    # temp_max/min coerentes com a media, caso o usuario informe so a media
    df["temp_max"] = np.maximum(df["temp_max"], df["temp_mean"])
    df["temp_min"] = np.minimum(df["temp_min"], df["temp_mean"])

    d = df["din_instante"]
    df["mes"] = d.dt.month
    df["dia_semana"] = d.dt.dayofweek
    df["trimestre"] = d.dt.quarter
    df["ano"] = d.dt.year
    feriados = holidays.Brazil(years=sorted(d.dt.year.unique()))
    df["is_feriado"] = d.dt.date.isin(feriados).astype(int)
    df["is_vespera_feriado"] = (d + pd.Timedelta(days=1)).dt.date.isin(feriados).astype(int)
    df["is_fds"] = (df["dia_semana"] >= 5).astype(int)
    ang = 2 * np.pi * d.dt.dayofyear / 365.25
    df["dia_ano_sen"] = np.sin(ang)
    df["dia_ano_cos"] = np.cos(ang)

    df["graus_refrig"] = (df["temp_mean"] - TEMP_BASE_REFRIGERACAO).clip(lower=0)
    df["graus_aquec"] = (TEMP_BASE_AQUECIMENTO - df["temp_mean"]).clip(lower=0)
    df["amplitude_termica"] = df["temp_max"] - df["temp_min"]
    df["horas_acima_26"] = [
        _horas_acima(26.0, m, mx, mn)
        for m, mx, mn in zip(df["temp_mean"], df["temp_max"], df["temp_min"])]

    # Sem serie temporal continua no modo interativo, o melhor palpite para a
    # inercia termica e o proprio dia.
    df["graus_refrig_med3"] = df.get("graus_refrig_med3", pd.Series(np.nan, index=df.index))
    df["graus_refrig_med3"] = df["graus_refrig_med3"].fillna(df["graus_refrig"])
    df["temp_mean_ontem"] = df.get("temp_mean_ontem", pd.Series(np.nan, index=df.index))
    df["temp_mean_ontem"] = df["temp_mean_ontem"].fillna(df["temp_mean"])

    ind = _industrial(df["din_instante"], defasagem_meses)
    df = df.merge(ind, on=["din_instante", "id_subsistema"], how="left")
    for c in ("industrial_mwmed", "industrial_var_anual"):
        if df[c].isna().any():                # data fora do alcance da EPE
            ultimo = ind.groupby("id_subsistema")[c].last()
            df[c] = df[c].fillna(df["id_subsistema"].map(ultimo))
    return df


def prever(entradas: pd.DataFrame, defasagem_meses: int = 2) -> pd.DataFrame:
    """Previsao de carga (MWmed) com a decomposicao nivel + resposta climatica."""
    df = montar_features(entradas, defasagem_meses)
    saidas = []
    for sub, g in df.groupby("id_subsistema", sort=False):
        pac = carregar_modelo(sub, "climatico")
        modelo = pac["modelo"]
        nivel, clima = modelo.decompor(g)
        r = g.copy()
        r["nivel_industrial"] = nivel
        r["resposta_climatica"] = clima
        r["previsao_mwmed"] = nivel + clima
        saidas.append(r)
    out = pd.concat(saidas).sort_index()
    out["regiao"] = out["id_subsistema"].map(NOMES_REGIOES)
    return out


@functools.lru_cache(maxsize=1)
def climatologia() -> pd.DataFrame:
    """Clima tipico de cada dia do ano, por subsistema (mediana historica).

    Suavizada com janela de 15 dias centrada: com apenas dois anos de historico,
    a mediana de um unico dia do ano e ruidosa demais para servir de normal.
    """
    h = historico()
    cols = [c for c in CAMPOS_CLIMA if c in h.columns]
    base = (h.assign(dia_ano=h["din_instante"].dt.dayofyear)
              .groupby(["id_subsistema", "dia_ano"])[cols].median().reset_index())

    partes = []
    for sub, g in base.groupby("id_subsistema"):
        g = g.sort_values("dia_ano").reset_index(drop=True)
        numerico = g[cols].astype(float)
        # Triplica a serie para a janela ser circular: 31/dez vizinho de 1/jan.
        suave = (pd.concat([numerico] * 3, ignore_index=True)
                   .rolling(15, center=True, min_periods=1).median())
        g[cols] = suave.iloc[len(g):2 * len(g)].to_numpy()
        partes.append(g.assign(id_subsistema=sub))
    return pd.concat(partes, ignore_index=True)


def completar_periodo(entradas: pd.DataFrame, ate=None,
                      de=None) -> pd.DataFrame:
    """Estende as entradas para todo o periodo, preenchendo com a climatologia.

    Serve para quando o arquivo enviado cobre so parte do periodo: os dias
    informados mantem o clima do usuario, os demais recebem o clima tipico da
    regiao naquele dia do ano. A coluna `origem` diz de onde veio cada linha.
    """
    df = entradas.copy()
    df["din_instante"] = pd.to_datetime(df["din_instante"])
    inicio = pd.to_datetime(de) if de is not None else df["din_instante"].min()
    fim = (pd.to_datetime(ate) if ate is not None
           else pd.Timestamp(year=int(df["din_instante"].max().year), month=12, day=31))
    if fim < inicio:
        raise ValueError("A data final é anterior à inicial.")

    grade = pd.date_range(inicio, fim, freq="D")
    normal = climatologia()
    cols = [c for c in CAMPOS_CLIMA if c in normal.columns]

    saidas = []
    for sub in df["id_subsistema"].unique():
        informado = (df[df["id_subsistema"] == sub]
                     .drop_duplicates("din_instante", keep="last")
                     .set_index("din_instante"))
        base = pd.DataFrame({"din_instante": grade, "id_subsistema": sub})
        base["dia_ano"] = base["din_instante"].dt.dayofyear

        n = normal[normal["id_subsistema"] == sub]
        if n.empty:                                   # regiao sem historico
            n = normal.groupby("dia_ano")[cols].median().reset_index()
        base = base.merge(n[["dia_ano"] + cols], on="dia_ano", how="left")
        base[cols] = base[cols].ffill().bfill()

        # Onde o usuario informou algo, o valor dele prevalece (celula a celula).
        base = base.set_index("din_instante")
        for col in cols:
            if col in informado.columns:
                base[col] = informado[col].reindex(base.index).combine_first(base[col])
        base["origem"] = np.where(base.index.isin(informado.index),
                                  "arquivo", "climatologia")
        saidas.append(base.reset_index().drop(columns="dia_ano"))

    return pd.concat(saidas, ignore_index=True).sort_values(
        ["id_subsistema", "din_instante"], ignore_index=True)


HORAS_DO_DIA = 24


def projetar_ano(ano: int, regioes: list[str] | None = None,
                 clima: pd.DataFrame | None = None) -> pd.DataFrame:
    """Projeta um ano inteiro, dia a dia, para as regioes pedidas.

    Sem `clima`, usa a climatologia historica -- e a projecao de "ano normal".
    Com `clima` (mesmo cobrindo so parte do ano), os dias informados usam o
    clima real e o restante e completado pela climatologia.
    """
    regioes = list(regioes or NOMES_REGIOES)
    inicio = pd.Timestamp(year=ano, month=1, day=1)
    fim = pd.Timestamp(year=ano, month=12, day=31)

    if clima is None or clima.empty:
        semente = pd.DataFrame({"id_subsistema": regioes, "din_instante": inicio})
    else:
        semente = clima[clima["id_subsistema"].isin(regioes)].copy()
        faltantes = [r for r in regioes if r not in set(semente["id_subsistema"])]
        if faltantes:
            semente = pd.concat(
                [semente, pd.DataFrame({"id_subsistema": faltantes,
                                        "din_instante": inicio})],
                ignore_index=True)

    return prever(completar_periodo(semente, de=inicio, ate=fim))


def resumo_periodo(previsoes: pd.DataFrame) -> pd.DataFrame:
    """Agrega uma previsao diaria em indicadores por regiao.

    Serve para qualquer periodo -- um ano fechado ou o intervalo que o arquivo
    enviado cobrir. `mwmed_medio` e a media das cargas diarias (a potencia
    media do periodo); a energia sai de MWmed x 24 h x dias.
    """
    chave = ["regiao", "id_subsistema"]
    g = previsoes.groupby(chave)["previsao_mwmed"]
    r = g.agg(dias="size", mwmed_medio="mean", mwmed_min="min",
              mwmed_max="max").reset_index()

    idx = pd.MultiIndex.from_frame(r[chave])
    r["energia_TWh"] = (g.sum() * HORAS_DO_DIA / 1e6).reindex(idx).to_numpy()
    r["fator_carga"] = r["mwmed_medio"] / r["mwmed_max"]
    r["de"] = previsoes.groupby(chave)["din_instante"].min().reindex(idx).to_numpy()
    r["ate"] = previsoes.groupby(chave)["din_instante"].max().reindex(idx).to_numpy()

    for col, rotulo in (("nivel_industrial", "nivel_medio"),
                        ("resposta_climatica", "resposta_climatica_media")):
        if col in previsoes.columns:
            r[rotulo] = (previsoes.groupby(chave)[col].mean()
                         .reindex(idx).to_numpy())

    if "origem" in previsoes.columns:
        do_arquivo = (previsoes[previsoes["origem"] == "arquivo"]
                      .groupby(chave).size().reindex(idx).fillna(0))
        r["dias_do_arquivo"] = do_arquivo.astype(int).to_numpy()
        r["dias_estimados"] = r["dias"] - r["dias_do_arquivo"]

    return r.sort_values("mwmed_medio", ascending=False, ignore_index=True)


# Nome antigo, mantido porque `06_projetar_ano.py` fala em ano.
resumo_anual = resumo_periodo


def comparar_com_observado(previsoes: pd.DataFrame) -> pd.DataFrame:
    """Confronta a previsao com a carga observada nos dias que existem no historico.

    Devolve quadro vazio quando o periodo previsto e inteiramente futuro --
    nesse caso nao ha o que comparar.
    """
    h = historico()[["din_instante", "id_subsistema", ALVO]]
    junto = previsoes.merge(h, on=["din_instante", "id_subsistema"], how="inner")
    if junto.empty:
        return pd.DataFrame()

    linhas = []
    for (regiao, sub), g in junto.groupby(["regiao", "id_subsistema"]):
        erro = g["previsao_mwmed"] - g[ALVO]
        linhas.append({
            "regiao": regiao, "id_subsistema": sub, "dias_comparados": len(g),
            "mwmed_previsto": g["previsao_mwmed"].mean(),
            "mwmed_observado": g[ALVO].mean(),
            "erro_medio_%": (g["previsao_mwmed"].mean() / g[ALVO].mean() - 1) * 100,
            "mae_diario": erro.abs().mean(),
            "mape_diario_%": (erro.abs() / g[ALVO]).mean() * 100,
        })
    return pd.DataFrame(linhas).sort_values("mwmed_observado", ascending=False,
                                            ignore_index=True)


def mwmed_anual_observado(ano: int) -> pd.Series:
    """MWmed medio observado no ano, por regiao -- referencia para comparar."""
    h = historico()
    do_ano = h[h["din_instante"].dt.year == ano]
    return do_ano.groupby("id_subsistema")[ALVO].mean()


def curva_temperatura(regiao: str, data, faixa: tuple[float, float] | None = None,
                      passos: int = 60, **clima_fixo) -> pd.DataFrame:
    """Varre a temperatura mantendo o resto fixo: a curva de resposta termica."""
    h = historico()
    g = h[h["id_subsistema"] == regiao]
    if faixa is None:
        faixa = (float(g["temp_mean"].min()) - 2, float(g["temp_mean"].max()) + 2)
    temps = np.linspace(faixa[0], faixa[1], passos)

    amplitude = float((g["temp_max"] - g["temp_min"]).median())
    base = pd.DataFrame({
        "id_subsistema": regiao,
        "din_instante": pd.to_datetime(data),
        "temp_mean": temps,
        "temp_max": temps + amplitude / 2,
        "temp_min": temps - amplitude / 2,
    })
    for k, v in clima_fixo.items():
        if v is not None:
            base[k] = v
    return prever(base)


def contexto_historico(regiao: str, data) -> dict:
    """Estatisticas da regiao na mesma epoca do ano, para dar escala ao numero."""
    h = historico()
    g = h[h["id_subsistema"] == regiao]
    dia = pd.to_datetime(data).dayofyear
    janela = g[(g["din_instante"].dt.dayofyear - dia).abs() <= 10]
    if janela.empty:
        janela = g
    return {
        "carga_media": float(janela[ALVO].mean()),
        "carga_min": float(janela[ALVO].min()),
        "carga_max": float(janela[ALVO].max()),
        "temp_media": float(janela["temp_mean"].mean()),
        "n_dias": int(len(janela)),
    }
