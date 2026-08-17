"""Treina, por subsistema, o modelo climatico e o modelo delta (D+1)."""
import argparse

import joblib
import pandas as pd

import _caminho  # noqa: F401
from energia.config import (ANO_TREINO, MODELOS, NOMES_REGIOES, PROCESSADO,
                            garantir_diretorios)
from energia.features import ALVO, features_do_modo
from energia.modelo import construir

MODOS = ["climatico", "delta_d1"]


def main(ano_treino: int, modos: list[str]) -> None:
    garantir_diretorios()
    df = pd.read_parquet(PROCESSADO / "dataset_modelo.parquet")
    treino = df[df["ano"] == ano_treino]
    if treino.empty:
        raise SystemExit(f"Sem dados de {ano_treino}. Rode scripts/02_construir_dataset.py")

    for modo in modos:
        print(f"\n{'=' * 70}\nMODO: {modo}\n{'=' * 70}")
        feats = features_do_modo(modo, df)
        for sigla, nome in NOMES_REGIOES.items():
            sub = treino[treino["id_subsistema"] == sigla].copy()
            obrigatorias = [ALVO] + (["carga_ontem", "carga_lag7", "media_7d"]
                                     if modo != "climatico" else [])
            sub = sub.dropna(subset=obrigatorias)
            if sub.empty:
                print(f"  {nome}: sem dados, pulando")
                continue

            modelo = construir(modo, feats).fit(sub)
            destino = MODELOS / f"{modo}_{sigla}.joblib"
            joblib.dump({"modelo": modelo, "features": feats, "modo": modo,
                         "regiao": sigla, "ano_treino": ano_treino}, destino)

            top = modelo.importancia(sub).head(5)
            print(f"  {nome:<20} {len(sub):>4} dias  ->  {destino.name}")
            print(f"    top features: " +
                  ", ".join(f"{k} ({v:.0%})" for k, v in top.items()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ano-treino", type=int, default=ANO_TREINO)
    p.add_argument("--modos", nargs="+", default=MODOS, choices=MODOS)
    a = p.parse_args()
    main(a.ano_treino, a.modos)
