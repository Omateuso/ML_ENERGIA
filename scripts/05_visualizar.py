"""Graficos comparativos a partir de output/previsoes_<ano>.csv."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import _caminho  # noqa: F401
from energia.config import ANO_TESTE, PROCESSADO, SAIDA

CORES = {"previsao_climatico": ("#d62728", "Modelo climatico (sem carga observada)"),
         "previsao_delta_recursivo": ("#2ca02c", "Delta recursivo"),
         "baseline_climatologia": ("#999999", "Baseline: climatologia 2024")}


def main(ano: int) -> None:
    arq = SAIDA / f"previsoes_{ano}.csv"
    if not arq.exists():
        raise SystemExit(f"{arq} nao existe. Rode scripts/04_avaliar.py")
    df = pd.read_csv(arq, parse_dates=["din_instante"])
    pasta = SAIDA / "plots"
    pasta.mkdir(parents=True, exist_ok=True)

    for regiao, g in df.groupby("regiao"):
        g = g.sort_values("din_instante")
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})

        ax.plot(g["din_instante"], g["val_cargaenergiamwmed"],
                label="Carga real", color="#1f77b4", linewidth=1.4)
        for col, (cor, rotulo) in CORES.items():
            if col in g.columns and g[col].notna().any():
                ax.plot(g["din_instante"], g[col], label=rotulo, color=cor,
                        linestyle="--", linewidth=1.1, alpha=0.85)
        ax.set_title(f"Carga diaria - {regiao} ({ano}, modelo treinado em {ano - 1})")
        ax.set_ylabel("Carga (MWmed)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)

        if "previsao_climatico" in g.columns:
            erro = g["previsao_climatico"] - g["val_cargaenergiamwmed"]
            ax2.fill_between(g["din_instante"], erro, 0, color="#d62728", alpha=0.4)
            ax2.axhline(0, color="black", linewidth=0.8)
            ax2.set_ylabel("Erro (MWmed)")
            ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.set_xlabel("Data")

        fig.tight_layout()
        sigla = g["id_subsistema"].iloc[0]
        destino = pasta / f"carga_{sigla}_{ano}.png"
        fig.savefig(destino, dpi=120)
        plt.close(fig)
        print(f"  {destino}")

    # Dispersao carga x temperatura: mostra a forma em U da resposta termica.
    fig, eixos = plt.subplots(1, df["regiao"].nunique(), figsize=(16, 4), sharey=False)
    proc = pd.read_parquet(PROCESSADO / "dataset_modelo.parquet")
    for ax, (regiao, g) in zip(eixos, proc.groupby("id_subsistema")):
        ax.scatter(g["temp_mean"], g["val_cargaenergiamwmed"], s=6, alpha=0.35,
                   c=g["is_fds"].map({0: "#1f77b4", 1: "#ff7f0e"}))
        ax.set_title(regiao)
        ax.set_xlabel("Temperatura media (C)")
        ax.grid(True, linestyle=":", alpha=0.5)
    eixos[0].set_ylabel("Carga (MWmed)")
    fig.suptitle("Carga x temperatura (azul = dia util, laranja = fim de semana)")
    fig.tight_layout()
    fig.savefig(pasta / "carga_vs_temperatura.png", dpi=120)
    plt.close(fig)
    print(f"  {pasta / 'carga_vs_temperatura.png'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ano", type=int, default=ANO_TESTE)
    main(p.parse_args().ano)
