"""EPE: consumo industrial mensal por subsistema -> nivel industrial diario.

Por que isto existe
-------------------
A carga do Norte e dominada por eletrointensivos (aluminio, mineracao) que nao
respondem a temperatura. Um modelo so climatico nao tem como saber em que
patamar a regiao esta operando, e por isso o Norte tinha R2 negativo. Esta
feature da ao modelo o nivel de base industrial.

Cuidado com contaminacao
------------------------
Consumo industrial e um COMPONENTE da carga total que estamos prevendo. Usar o
valor do proprio mes entregaria parte da resposta. A EPE publica com cerca de
45 dias de atraso, entao a serie e defasada em `defasagem_meses` (2 por padrao):
o modelo so enxerga o que estaria de fato publicado na data da previsao.
"""
from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

from .config import BRUTO_EPE, SIN

ARQUIVO = BRUTO_EPE / "consumo_mensal_por_classe.xlsx"

MESES = ["JAN", "FEV", "MAR", "ABR", "MAI", "JUN",
         "JUL", "AGO", "SET", "OUT", "NOV", "DEZ"]

# Rotulos da aba, sob o cabecalho "SUBSISTEMA ELETRICO", -> id_subsistema do ONS
ROTULO_PARA_SUBSISTEMA = {
    "NORTE": "N",
    "NORDESTE": "NE",
    "SUDESTE/C. OESTE": "SE",
    "SUDESTE/C.OESTE": "SE",
    "SUL": "S",
}


def _norm(v) -> str:
    s = unicodedata.normalize("NFKD", str(v))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.upper().split())


def ler_industrial_mensal(caminho=ARQUIVO) -> pd.DataFrame:
    """Extrai o bloco SUBSISTEMA ELETRICO de cada ano -> (ano, mes, subsistema, MWh)."""
    if not caminho.exists():
        raise FileNotFoundError(f"Planilha da EPE nao encontrada: {caminho}. "
                                f"Rode scripts/01_coletar.py")
    bruto = pd.read_excel(caminho, sheet_name="INDUSTRIAL", header=None)

    registros = []
    ano_atual = None
    dentro_subsistema = False

    for _, linha in bruto.iterrows():
        rotulo = _norm(linha[0]) if pd.notna(linha[0]) else ""

        # Cabecalho de ano: "2025" repetido nas colunas de mes. O asterisco de
        # "2026*" marca dado preliminar -- mantemos, mas o ano e o mesmo.
        marcador = str(linha[1]).strip() if pd.notna(linha[1]) else ""
        if marcador.rstrip("*").isdigit() and len(marcador.rstrip("*")) == 4:
            ano_atual = int(marcador.rstrip("*"))
            dentro_subsistema = False
            continue

        if rotulo.startswith("SUBSISTEMA EL"):
            dentro_subsistema = True
            continue
        if rotulo.startswith("REGIAO GEOGRAFICA") or rotulo.startswith("TOTAL"):
            dentro_subsistema = False       # bloco por regiao do IBGE: ignorar
            continue

        if not (dentro_subsistema and ano_atual and rotulo):
            continue
        sub = ROTULO_PARA_SUBSISTEMA.get(rotulo)
        if sub is None:                      # "Sistemas isolados" cai aqui
            continue

        for i, mes in enumerate(MESES, start=1):
            valor = pd.to_numeric(linha[i], errors="coerce")
            if pd.notna(valor):
                registros.append({"ano": ano_atual, "mes": i,
                                  "id_subsistema": sub, "industrial_mwh": float(valor)})

    df = pd.DataFrame(registros)
    if df.empty:
        raise ValueError("Nao foi possivel extrair o bloco SUBSISTEMA ELETRICO da EPE")
    return (df.drop_duplicates(subset=["ano", "mes", "id_subsistema"], keep="last")
              .sort_values(["id_subsistema", "ano", "mes"], ignore_index=True))


def nivel_industrial_diario(datas: pd.Series, defasagem_meses: int = 2,
                            caminho=ARQUIVO) -> pd.DataFrame:
    """Serie diaria do nivel industrial por subsistema, em MWmed e defasada.

    MWh do mes / horas do mes = MWmed, que e a mesma unidade da carga do ONS.
    O valor mensal e ancorado no dia 15 e interpolado, evitando degraus
    artificiais na virada do mes.
    """
    mensal = ler_industrial_mensal(caminho)

    mensal["ancora"] = pd.to_datetime(
        dict(year=mensal["ano"], month=mensal["mes"], day=15))
    horas = mensal["ancora"].dt.days_in_month * 24
    mensal["industrial_mwmed"] = mensal["industrial_mwh"] / horas

    # A defasagem e o coracao da protecao contra contaminacao: o valor de
    # janeiro so passa a ser visivel em marco.
    mensal["ancora_visivel"] = (mensal["ancora"]
                                + pd.DateOffset(months=defasagem_meses))

    inicio, fim = pd.to_datetime(datas).min(), pd.to_datetime(datas).max()
    grade = pd.date_range(inicio, fim, freq="D")

    saidas = []
    for sub, g in mensal.groupby("id_subsistema"):
        g = g.sort_values("ancora_visivel")
        ancoras = pd.DatetimeIndex(g["ancora_visivel"])
        s = (pd.Series(g["industrial_mwmed"].to_numpy(), index=ancoras)
               .reindex(ancoras.union(grade))
               .interpolate(method="time")
               .ffill().bfill()
               .reindex(grade))
        saidas.append(pd.DataFrame({"din_instante": grade,
                                    "id_subsistema": sub,
                                    "industrial_mwmed": s.values}))

    diario = pd.concat(saidas, ignore_index=True)

    # Patamar industrial nacional = soma dos subsistemas. Necessario para o
    # modelo do SIN e para prever o SIN pela interface.
    nacional = (diario.groupby("din_instante", as_index=False)["industrial_mwmed"]
                      .sum())
    nacional["id_subsistema"] = SIN
    diario = pd.concat([diario, nacional], ignore_index=True)

    # Variacao anual do patamar industrial: capta expansao/retracao de planta,
    # que e o que faz a carga do Norte saltar de degrau.
    diario = diario.sort_values(["id_subsistema", "din_instante"])
    diario["industrial_var_anual"] = (
        diario.groupby("id_subsistema")["industrial_mwmed"]
              .transform(lambda x: x.pct_change(365).fillna(0.0) * 100))
    return diario.reset_index(drop=True)


COLUNAS_INDUSTRIAL = ["industrial_mwmed", "industrial_var_anual"]
