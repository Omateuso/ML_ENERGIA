"""Avalia os modelos no ano de teste, contra baselines, com teste de contaminacao.

Tres modos sao reportados lado a lado:

  climatico          nenhuma feature autoregressiva. A carga do ano de teste
                     nunca entra como insumo -- contaminacao e impossivel por
                     construcao.
  delta_d1           previsao operacional de um dia a frente; usa a carga
                     observada de ontem (legitimo, mas nao e "prever o ano").
  delta_recursivo    o mesmo modelo realimentando as proprias previsoes, sem
                     ler carga observada. Mede previsao de horizonte longo.
"""
import argparse

import joblib
import numpy as np
import pandas as pd

import _caminho  # noqa: F401
from energia.avaliacao import (baselines, metricas, prever_recursivo,
                               teste_contaminacao)
from energia.config import (ANO_TESTE, ANO_TREINO, MODELOS, NOMES_REGIOES,
                            PROCESSADO, SAIDA, SIN, SUBSISTEMAS,
                            garantir_diretorios)
from energia.features import (ALVO, AUTOREGRESSIVO, CALENDARIO, CLIMA,
                              INDUSTRIAL)

GRUPOS = {"clima": CLIMA, "industrial": INDUSTRIAL,
          "calendario": CALENDARIO, "autoregressivo": AUTOREGRESSIVO}

# Um modelo que nunca le a carga do ano de teste nao pode ser cobrado contra um
# baseline que le. Cada modo e comparado com a sua propria classe de referencia.
BASELINES_SEM_CARGA = ["media_treino", "climatologia"]
BASELINES_COM_CARGA = ["persistencia", "semana_anterior"]
CLASSE_DO_MODO = {"climatico": "sem carga observada",
                  "climatico (soma dos 4)": "sem carga observada",
                  "delta_recursivo (soma dos 4)": "sem carga observada",
                  "delta_recursivo": "sem carga observada",
                  "delta_d1": "com carga observada"}


def main(ano_treino: int, ano_teste: int) -> None:
    garantir_diretorios()
    df = pd.read_parquet(PROCESSADO / "dataset_modelo.parquet")
    treino_todo = df[df["ano"] == ano_treino]
    teste_todo = df[df["ano"] == ano_teste]

    linhas, contaminacao, series = [], [], []

    for sigla, nome in NOMES_REGIOES.items():
        treino = treino_todo[treino_todo["id_subsistema"] == sigla]
        teste = (teste_todo[teste_todo["id_subsistema"] == sigla]
                 .sort_values("din_instante").reset_index(drop=True))
        if teste.empty:
            continue

        base = baselines(treino, teste)
        for bnome, bpred in base.items():
            linhas.append({"regiao": nome, "modo": f"[baseline] {bnome}",
                           **metricas(teste[ALVO], bpred)})

        previsoes = {}

        # --- modelo climatico: sem nenhuma carga observada de 2025 ---
        cam = MODELOS / f"climatico_{sigla}.joblib"
        if cam.exists():
            pac = joblib.load(cam)
            m = pac["modelo"]
            p = m.predict(teste)
            previsoes["climatico"] = p
            linhas.append({"regiao": nome, "modo": "climatico",
                           **metricas(teste[ALVO], p)})
            c = teste_contaminacao(m, teste, GRUPOS)
            c.insert(0, "regiao", nome)
            c.insert(1, "modo", "climatico")
            contaminacao.append(c)

        # --- modelo delta: D+1 e recursivo ---
        cam = MODELOS / f"delta_d1_{sigla}.joblib"
        if cam.exists():
            pac = joblib.load(cam)
            m = pac["modelo"]
            t = teste.dropna(subset=["carga_ontem", "carga_lag7", "media_7d"])
            p = m.predict(t)
            linhas.append({"regiao": nome, "modo": "delta_d1",
                           **metricas(t[ALVO], p)})
            c = teste_contaminacao(m, t, GRUPOS)
            c.insert(0, "regiao", nome)
            c.insert(1, "modo", "delta_d1")
            contaminacao.append(c)

            p_rec = prever_recursivo(m, teste)
            previsoes["delta_recursivo"] = p_rec
            linhas.append({"regiao": nome, "modo": "delta_recursivo",
                           **metricas(teste[ALVO], p_rec)})

        s = teste[["din_instante", "id_subsistema", ALVO]].copy()
        s["regiao"] = nome
        for k, v in previsoes.items():
            s[f"previsao_{k}"] = v
        s["baseline_climatologia"] = base["climatologia"]
        series.append(s)

    serie = pd.concat(series, ignore_index=True)

    # Modelo dedicado ao SIN vs. somar as previsoes dos quatro subsistemas.
    # Vale medir: no agregado os erros regionais podem se cancelar.
    partes = serie[serie["id_subsistema"].isin(SUBSISTEMAS)]
    real_sin = serie[serie["id_subsistema"] == SIN].set_index("din_instante")[ALVO]
    if not partes.empty and not real_sin.empty:
        dias_completos = (partes.groupby("din_instante")["id_subsistema"].nunique()
                          == len(SUBSISTEMAS))
        for modo in ("climatico", "delta_recursivo"):
            col = f"previsao_{modo}"
            if col not in partes.columns or partes[col].isna().all():
                continue
            soma = partes.groupby("din_instante")[col].sum()[dias_completos]
            linhas.append({"regiao": NOMES_REGIOES[SIN],
                           "modo": f"{modo} (soma dos 4)",
                           **metricas(real_sin.reindex(soma.index), soma)})

    res = pd.DataFrame(linhas)
    cont = pd.concat(contaminacao, ignore_index=True) if contaminacao else pd.DataFrame()

    SAIDA.mkdir(parents=True, exist_ok=True)
    res.to_csv(SAIDA / "metricas.csv", index=False)
    cont.to_csv(SAIDA / "teste_contaminacao.csv", index=False)
    serie.to_csv(SAIDA / f"previsoes_{ano_teste}.csv", index=False)

    fmt = lambda x: f"{x:,.1f}" if isinstance(x, float) else str(x)
    pd.set_option("display.width", 200, "display.max_columns", 30)

    print(f"\n{'=' * 92}")
    print(f"DESEMPENHO EM {ano_teste} (treino em {ano_treino}) - MWmed")
    print("Cada modo e comparado com baselines da MESMA classe de informacao.")
    print("=" * 92)
    for nome in res["regiao"].unique():
        r = res[res["regiao"] == nome].copy()
        mae_de = dict(zip(r["modo"].str.replace("[baseline] ", "", regex=False), r["mae"]))
        ref = {"sem carga observada": min(mae_de[b] for b in BASELINES_SEM_CARGA
                                          if b in mae_de),
               "com carga observada": min(mae_de[b] for b in BASELINES_COM_CARGA
                                          if b in mae_de)}

        def ganho(linha):
            classe = CLASSE_DO_MODO.get(linha["modo"])
            if classe is None:                      # a propria linha e baseline
                return np.nan
            return (1 - linha["mae"] / ref[classe]) * 100

        r["classe"] = r["modo"].map(CLASSE_DO_MODO).fillna("-")
        r["ganho_%"] = r.apply(ganho, axis=1)
        print(f"\n--- {nome} ---")
        print(r[["modo", "classe", "mae", "rmse", "mape", "r2", "ganho_%"]]
              .to_string(index=False, float_format=fmt, na_rep="-"))

    if not cont.empty:
        print(f"\n{'=' * 92}")
        print("TESTE DE CONTAMINACAO - quanto o erro piora ao embaralhar cada grupo")
        print("Clima com piora alta = a previsao esta mesmo apoiada no clima.")
        print("=" * 92)
        piv = cont.pivot_table(index=["regiao", "modo"], columns="grupo",
                               values="piora_%")
        print(piv.to_string(float_format=lambda x: f"{x:+.1f}%"))

    print(f"\nArquivos salvos em {SAIDA}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ano-treino", type=int, default=ANO_TREINO)
    p.add_argument("--ano-teste", type=int, default=ANO_TESTE)
    a = p.parse_args()
    main(a.ano_treino, a.ano_teste)
