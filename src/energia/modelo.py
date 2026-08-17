"""Modelos de previsao de carga.

Dois desenhos, para dois usos diferentes:

ModeloClimatico
    Previsao de carga a partir de clima, calendario e patamar industrial, SEM
    nenhuma variavel autoregressiva. E o modelo que responde "quanto de energia
    esta regiao gasta com este clima". Como o XGBoost nao extrapola alem do
    intervalo do alvo visto no treino, a carga e decomposta em

        carga = nivel(industrial) + resposta_climatica

    O nivel e uma regressao linear sobre o patamar industrial -- linear
    extrapola sem problema. O XGBoost so modela o residuo, que e estacionario.
    E a mesma motivacao do delta, sem pagar o preco da deriva acumulada.

ModeloDelta
    Preve a variacao diaria usando a carga observada de ontem. Serve para
    previsao operacional de um dia a frente (D+1), onde conhecer a carga de
    ontem e legitimo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from .features import ALVO

PARAMS_XGB = dict(n_estimators=600, learning_rate=0.05, max_depth=5,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_lambda=1.0, n_jobs=-1, random_state=42)

COLUNA_NIVEL = "industrial_mwmed"


class ModeloClimatico:
    """Nivel industrial (linear, extrapolavel) + resposta climatica (XGBoost)."""

    modo = "climatico"

    def __init__(self, features: list[str], **kw):
        self.features = list(features)
        self.nivel = LinearRegression()
        self.residuo = XGBRegressor(**{**PARAMS_XGB, **kw})
        self.media_treino: float | None = None

    def _x_nivel(self, df: pd.DataFrame) -> np.ndarray:
        if COLUNA_NIVEL in df.columns and df[COLUNA_NIVEL].notna().any():
            return df[[COLUNA_NIVEL]].ffill().bfill().to_numpy()
        return np.zeros((len(df), 1))

    def fit(self, df: pd.DataFrame) -> "ModeloClimatico":
        y = df[ALVO].to_numpy()
        self.media_treino = float(y.mean())
        self.nivel.fit(self._x_nivel(df), y)
        base = self.nivel.predict(self._x_nivel(df))
        self.residuo.fit(df[self.features], y - base)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        base, clima = self.decompor(df)
        return base + clima

    def decompor(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Separa a previsao em (nivel industrial, resposta climatica).

        A soma das duas e a previsao. Serve para mostrar na interface quanto da
        carga e patamar de base e quanto o clima esta somando ou tirando.
        """
        base = self.nivel.predict(self._x_nivel(df))
        return base, self.residuo.predict(df[self.features])

    def importancia(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.residuo.feature_importances_,
                         index=self.features).sort_values(ascending=False)


class ModeloDelta:
    """Preve carga_hoje - carga_ontem. Requer a carga observada de ontem."""

    modo = "delta"

    def __init__(self, features: list[str], **kw):
        self.features = list(features)
        self.modelo = XGBRegressor(**{**PARAMS_XGB, **kw})

    def fit(self, df: pd.DataFrame) -> "ModeloDelta":
        self.modelo.fit(df[self.features], df[ALVO] - df["carga_ontem"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["carga_ontem"].to_numpy() + self.modelo.predict(df[self.features])

    def predict_delta(self, df: pd.DataFrame) -> np.ndarray:
        return self.modelo.predict(df[self.features])

    def importancia(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.modelo.feature_importances_,
                         index=self.features).sort_values(ascending=False)


def construir(modo: str, features: list[str], **kw):
    if modo == "climatico":
        return ModeloClimatico(features, **kw)
    if modo in ("delta_d1", "delta", "recursivo"):
        return ModeloDelta(features, **kw)
    raise ValueError(f"modo desconhecido: {modo}")
