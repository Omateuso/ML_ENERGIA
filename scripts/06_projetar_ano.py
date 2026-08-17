"""Projeta a carga media anual (MWmed) e a energia (TWh) de um ano inteiro.

Sem argumentos extras usa a climatologia historica -- projecao de "ano normal".
Com --com-clima-observado usa o clima realmente medido no ano (util para
backtest: mede o quanto a media anual acerta quando o clima e conhecido).
"""
import argparse

import pandas as pd

import _caminho  # noqa: F401
from energia.config import ANO_TESTE, NOMES_REGIOES, SAIDA, SUBSISTEMAS, SIN
from energia.previsao import (CAMPOS_CLIMA, historico, mwmed_anual_observado,
                              projetar_ano, resumo_periodo)


def clima_observado(ano: int) -> pd.DataFrame | None:
    h = historico()
    do_ano = h[h["din_instante"].dt.year == ano]
    if do_ano.empty:
        return None
    cols = [c for c in CAMPOS_CLIMA if c in do_ano.columns]
    return do_ano[["id_subsistema", "din_instante"] + cols].copy()


def main(ano: int, usar_observado: bool) -> None:
    clima = clima_observado(ano) if usar_observado else None
    if usar_observado and clima is None:
        raise SystemExit(f"Não há clima observado de {ano} no dataset. "
                         f"Rode sem --com-clima-observado para projetar pela "
                         f"climatologia.")

    diario = projetar_ano(ano, clima=clima)
    resumo = resumo_periodo(diario)

    fonte = "clima observado" if usar_observado else "climatologia historica"
    print(f"\n{'=' * 88}")
    print(f"PROJECAO ANUAL DE {ano} - fonte de clima: {fonte}")
    print("=" * 88)
    print(resumo[["regiao", "dias", "mwmed_medio", "mwmed_min", "mwmed_max",
                  "energia_TWh", "fator_carga"]]
          .to_string(index=False, float_format=lambda x: f"{x:,.1f}"))

    # A soma dos subsistemas costuma bater o modelo dedicado do SIN.
    por_sigla = resumo.set_index("id_subsistema")
    if SIN in por_sigla.index and set(SUBSISTEMAS) <= set(por_sigla.index):
        soma = por_sigla.loc[SUBSISTEMAS, "mwmed_medio"].sum()
        print(f"\nSIN pelo modelo dedicado : {por_sigla.loc[SIN, 'mwmed_medio']:,.1f} MWmed")
        print(f"SIN somando os 4 subsist.: {soma:,.1f} MWmed")

    real = mwmed_anual_observado(ano)
    if not real.empty:
        print(f"\n{'=' * 88}")
        print(f"COMPARACAO COM O OBSERVADO DE {ano}")
        print("=" * 88)
        comp = por_sigla.join(real.rename("real_mwmed"), how="inner")
        comp["erro_%"] = (comp["mwmed_medio"] / comp["real_mwmed"] - 1) * 100
        if SIN in comp.index and set(SUBSISTEMAS) <= set(por_sigla.index):
            comp.loc["SIN (soma dos 4)"] = {
                "regiao": "SIN (soma dos 4)",
                "mwmed_medio": soma, "real_mwmed": real.get(SIN),
                "erro_%": (soma / real.get(SIN) - 1) * 100}
        print(comp[["regiao", "mwmed_medio", "real_mwmed", "erro_%"]]
              .to_string(index=False, float_format=lambda x: f"{x:,.1f}"))

    SAIDA.mkdir(parents=True, exist_ok=True)
    sufixo = "clima_observado" if usar_observado else "climatologia"
    diario.to_csv(SAIDA / f"projecao_{ano}_{sufixo}_diaria.csv", index=False)
    resumo.to_csv(SAIDA / f"projecao_{ano}_{sufixo}_resumo.csv", index=False)
    print(f"\nSalvo em {SAIDA}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ano", type=int, default=ANO_TESTE)
    p.add_argument("--com-clima-observado", action="store_true",
                   help="usa o clima medido do ano em vez da climatologia")
    a = p.parse_args()
    main(a.ano, a.com_clima_observado)
