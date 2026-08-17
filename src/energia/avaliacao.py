"""Avaliacao honesta: metricas, baselines e teste de contaminacao.

Duas regras que este modulo impoe:

1. Toda metrica vem acompanhada de baselines. R2 alto sobre carga absoluta nao
   prova nada quando `carga_ontem` esta entre as features -- a persistencia pura
   ja entrega R2 na casa de 0,5. O que mede habilidade e o ganho sobre o melhor
   baseline.

2. O modo recursivo nunca le a carga observada do periodo de teste. Se o
   desempenho se sustentar ali, a previsao esta mesmo apoiada em clima.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .features import ALVO


def metricas(y_real, y_prev) -> dict:
    y_real = np.asarray(y_real, dtype=float)
    y_prev = np.asarray(y_prev, dtype=float)
    ok = ~(np.isnan(y_real) | np.isnan(y_prev))
    y_real, y_prev = y_real[ok], y_prev[ok]
    if len(y_real) == 0:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan, "n": 0}
    return {
        "mae": mean_absolute_error(y_real, y_prev),
        "rmse": float(np.sqrt(mean_squared_error(y_real, y_prev))),
        "mape": float(np.mean(np.abs((y_real - y_prev) / y_real)) * 100),
        "r2": r2_score(y_real, y_prev),
        "n": int(len(y_real)),
    }


def baselines(treino: pd.DataFrame, teste: pd.DataFrame) -> dict[str, np.ndarray]:
    """Referencias que qualquer modelo util precisa bater."""
    n = len(teste)
    b = {"media_treino": np.full(n, treino[ALVO].mean())}

    # Climatologia: media da carga no mesmo dia do ano, no periodo de treino.
    # E o baseline honesto para um modelo climatico -- captura a sazonalidade
    # sem olhar nada do ano de teste.
    perfil = treino.groupby(treino["din_instante"].dt.dayofyear)[ALVO].mean()
    b["climatologia"] = (teste["din_instante"].dt.dayofyear
                         .map(perfil).ffill().bfill().to_numpy())

    if "carga_ontem" in teste.columns and teste["carga_ontem"].notna().any():
        b["persistencia"] = teste["carga_ontem"].to_numpy()
    if "carga_lag7" in teste.columns and teste["carga_lag7"].notna().any():
        b["semana_anterior"] = teste["carga_lag7"].to_numpy()
    return b


def prever_recursivo(modelo, teste: pd.DataFrame, dias_semente: int = 7) -> np.ndarray:
    """Roda o periodo de teste realimentando as proprias previsoes.

    So clima, calendario e patamar industrial vem do periodo de teste. As
    features autoregressivas sao reconstruidas a partir do que o modelo ja
    previu, entao nenhuma carga observada do teste entra apos a semente.
    """
    teste = teste.sort_values("din_instante").reset_index(drop=True)
    previstos = list(teste[ALVO].iloc[:dias_semente].to_numpy())

    for i in range(dias_semente, len(teste)):
        linha = teste.iloc[[i]].copy()
        linha["carga_ontem"] = previstos[-1]
        linha["carga_lag7"] = previstos[-7]
        linha["media_7d"] = float(np.mean(previstos[-7:]))
        previstos.append(float(modelo.predict(linha)[0]))
    return np.array(previstos)


def importancia_permutacao(modelo, teste: pd.DataFrame, colunas: list[str],
                           repeticoes: int = 8, semente: int = 0) -> pd.Series:
    """Quanto o MAE piora ao embaralhar cada coluna. Mede uso real da feature."""
    rng = np.random.default_rng(semente)
    base = metricas(teste[ALVO], modelo.predict(teste))["mae"]
    out = {}
    for col in colunas:
        if col not in teste.columns:
            continue
        pioras = []
        for _ in range(repeticoes):
            emb = teste.copy()
            emb[col] = rng.permutation(emb[col].to_numpy())
            pioras.append(metricas(emb[ALVO], modelo.predict(emb))["mae"] - base)
        out[col] = float(np.mean(pioras))
    return pd.Series(out).sort_values(ascending=False)


def teste_contaminacao(modelo, teste: pd.DataFrame, grupos: dict[str, list[str]],
                       semente: int = 0) -> pd.DataFrame:
    """Embaralha grupos inteiros de features e mede o estrago.

    Se embaralhar o clima quase nao mudar o erro, a previsao NAO esta apoiada
    em clima -- esta se sustentando em outra coisa (tipicamente a carga
    observada que tambem serve de gabarito).
    """
    rng = np.random.default_rng(semente)
    base = metricas(teste[ALVO], modelo.predict(teste))["mae"]
    linhas = []
    for nome, grupo in grupos.items():
        cols = [c for c in grupo if c in teste.columns]
        if not cols:
            continue
        emb = teste.copy()
        for c in cols:
            emb[c] = rng.permutation(emb[c].to_numpy())
        mae = metricas(emb[ALVO], modelo.predict(emb))["mae"]
        linhas.append({"grupo": nome, "mae_base": base, "mae_embaralhado": mae,
                       "piora_%": (mae / base - 1) * 100})
    return pd.DataFrame(linhas)
