"""Junta carga, clima e patamar industrial num unico dataset de modelagem."""
import argparse

import _caminho  # noqa: F401
from energia.config import ANO_TESTE, ANO_TREINO
from energia.features import construir_dataset

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--anos", type=int, nargs="+", default=[ANO_TREINO, ANO_TESTE])
    p.add_argument("--sem-cache", action="store_true",
                   help="reprocessa o INMET do zero em vez de usar o cache")
    p.add_argument("--defasagem-industrial", type=int, default=2,
                   help="meses de atraso da serie da EPE (protecao contra vazamento)")
    a = p.parse_args()

    df = construir_dataset(a.anos, usar_cache=not a.sem_cache,
                           defasagem_industrial=a.defasagem_industrial)

    print("\nCobertura por subsistema:")
    resumo = (df.groupby("id_subsistema")
                .agg(linhas=("din_instante", "size"),
                     de=("din_instante", "min"), ate=("din_instante", "max"),
                     temp_media=("temp_mean", "mean"),
                     estacoes=("n_estacoes", "mean"),
                     industrial=("industrial_mwmed", "mean")))
    print(resumo.to_string(float_format=lambda x: f"{x:,.1f}"))
