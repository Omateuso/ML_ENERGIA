"""Interface grafica de previsao de carga a partir do clima.

Executar:  streamlit run app/app.py
"""
from __future__ import annotations

import io
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energia.config import NOMES_REGIOES, SAIDA, SUBSISTEMAS
from energia.entrada import (ALIASES, EXTENSOES, FormatoNaoSuportado,
                             ler_tabela, para_csv, para_xlsx, planilhas)
from energia.previsao import (CAMPOS_CLIMA, CAMPOS_PEDIDOS, ROTULOS,
                              DadosAusentes, comparar_com_observado,
                              completar_periodo,
                              contexto_historico, curva_temperatura,
                              historico, mwmed_anual_observado,
                              padroes_regionais, prever, projetar_ano,
                              resumo_periodo)

st.set_page_config(page_title="Previsão de carga do SIN",
                   page_icon="⚡", layout="wide")

AZUL, VERMELHO, CINZA = "#1f77b4", "#d62728", "#9e9e9e"
XLSX_MIME = ("application/vnd.openxmlformats-officedocument."
             "spreadsheetml.sheet")


@st.cache_data(show_spinner=False)
def _historico() -> pd.DataFrame:
    return historico()


@st.cache_data(show_spinner=False)
def _padroes() -> dict:
    return padroes_regionais()


@st.cache_data(show_spinner=False)
def _prever(entradas: pd.DataFrame) -> pd.DataFrame:
    return prever(entradas)


@st.cache_data(show_spinner=False)
def _projetar(ano: int, com_clima_observado: bool) -> pd.DataFrame:
    clima = None
    if com_clima_observado:
        h = historico()
        do_ano = h[h["din_instante"].dt.year == ano]
        cols = [c for c in CAMPOS_CLIMA if c in do_ano.columns]
        clima = do_ano[["id_subsistema", "din_instante"] + cols].copy()
    return projetar_ano(ano, clima=clima)


@st.cache_data(show_spinner=False)
def _curva(regiao: str, dia: date, umid: float) -> pd.DataFrame:
    return curva_temperatura(regiao, pd.Timestamp(dia), umid_mean=umid)


def _fmt(v: float) -> str:
    return f"{v:,.0f}".replace(",", ".")


def _render_resultado(saida: pd.DataFrame, rotulo: str) -> None:
    """Bloco completo de resultados, usado tanto na projecao anual quanto no lote.

    Mantido em um lugar so para que enviar um arquivo entregue exatamente os
    mesmos indicadores da projecao anual.
    """
    resumo = resumo_periodo(saida)
    por_sigla = resumo.set_index("id_subsistema")
    tem_sin = "SIN" in por_sigla.index
    tem_todos = set(SUBSISTEMAS) <= set(por_sigla.index)
    soma_4 = por_sigla.reindex(SUBSISTEMAS)["mwmed_medio"].sum() if tem_todos else None

    # --- indicadores de destaque ---
    if tem_sin:
        linha = por_sigla.loc["SIN"]
        cols = st.columns(4 if soma_4 is not None else 3)
        cols[0].metric(f"SIN — carga média ({rotulo})",
                       f"{_fmt(linha['mwmed_medio'])} MWmed")
        cols[1].metric("Energia no período",
                       f"{linha['energia_TWh']:,.1f} TWh".replace(",", "."))
        cols[2].metric("Faixa diária",
                       f"{_fmt(linha['mwmed_min'])} – {_fmt(linha['mwmed_max'])}")
        if soma_4 is not None:
            cols[3].metric("Somando os 4 subsistemas", f"{_fmt(soma_4)} MWmed",
                           delta=f"{soma_4 - linha['mwmed_medio']:+,.0f}"
                                 .replace(",", "."),
                           help="Medido em 2025, somar os subsistemas errou "
                                "menos que o modelo dedicado do SIN "
                                "(−1,6% contra −4,6%).")
    else:
        cols = st.columns(3)
        cols[0].metric("Carga média do período",
                       f"{_fmt(resumo['mwmed_medio'].mean())} MWmed")
        cols[1].metric("Energia total",
                       f"{resumo['energia_TWh'].sum():,.1f} TWh".replace(",", "."))
        cols[2].metric("Regiões", len(resumo))

    # --- tabela por regiao ---
    st.subheader("Indicadores por região")
    mostrar = ["regiao", "dias", "mwmed_medio", "mwmed_min", "mwmed_max",
               "energia_TWh", "fator_carga"]
    for extra in ("nivel_medio", "resposta_climatica_media",
                  "dias_do_arquivo", "dias_estimados"):
        if extra in resumo.columns:
            mostrar.append(extra)
    st.dataframe(
        resumo[mostrar].style.format({
            "mwmed_medio": "{:,.0f}", "mwmed_min": "{:,.0f}",
            "mwmed_max": "{:,.0f}", "energia_TWh": "{:,.1f}",
            "fator_carga": "{:.2f}", "nivel_medio": "{:,.0f}",
            "resposta_climatica_media": "{:+,.0f}"}),
        width="stretch", hide_index=True)

    # --- confronto com o observado, quando o periodo ja aconteceu ---
    comp = comparar_com_observado(saida)
    if not comp.empty:
        if tem_sin and tem_todos:
            sin_obs = comp[comp["id_subsistema"] == "SIN"]
            subs = comp[comp["id_subsistema"].isin(SUBSISTEMAS)]
            if not sin_obs.empty and len(subs) == len(SUBSISTEMAS):
                obs = float(sin_obs["mwmed_observado"].iloc[0])
                comp = pd.concat([comp, pd.DataFrame([{
                    "regiao": "SIN (soma dos 4)", "id_subsistema": "—",
                    "dias_comparados": int(sin_obs["dias_comparados"].iloc[0]),
                    "mwmed_previsto": soma_4, "mwmed_observado": obs,
                    "erro_medio_%": (soma_4 / obs - 1) * 100,
                    "mae_diario": np.nan, "mape_diario_%": np.nan}])],
                    ignore_index=True)
        st.subheader("Acerto contra o observado")
        st.caption("Só os dias do período que já existem no histórico "
                   "(2024–2025) entram nesta comparação.")
        st.dataframe(
            comp[["regiao", "dias_comparados", "mwmed_previsto",
                  "mwmed_observado", "erro_medio_%", "mae_diario",
                  "mape_diario_%"]].style.format({
                      "mwmed_previsto": "{:,.0f}", "mwmed_observado": "{:,.0f}",
                      "erro_medio_%": "{:+.1f}%", "mae_diario": "{:,.0f}",
                      "mape_diario_%": "{:.2f}%"}, na_rep="—"),
            width="stretch", hide_index=True)

    # --- serie diaria ---
    if saida["din_instante"].nunique() > 1:
        st.subheader("Série diária")
        fig, ax = plt.subplots(figsize=(11, 4.2))
        for regiao, g in saida.groupby("regiao"):
            g = g.sort_values("din_instante")
            estilo = (dict(linewidth=2.2, zorder=5) if regiao.startswith("SIN")
                      else dict(linewidth=1.3))
            linha_, = ax.plot(g["din_instante"], g["previsao_mwmed"],
                              label=regiao, **estilo)
            if "origem" in g.columns:
                do_arq = g[g["origem"] == "arquivo"]
                if not do_arq.empty and len(do_arq) < len(g):
                    ax.scatter(do_arq["din_instante"], do_arq["previsao_mwmed"],
                               s=28, zorder=6, color=linha_.get_color(),
                               edgecolor="black", linewidth=0.6)
        ax.set_xlabel("Data")
        ax.set_ylabel("Carga prevista (MWmed)")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.4)
        if "origem" in saida.columns and (saida["origem"] == "climatologia").any():
            ax.set_title("Pontos marcados = clima do seu arquivo; "
                         "linha = climatologia histórica", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # --- perfil mensal ---
    if saida["din_instante"].dt.to_period("M").nunique() > 1:
        st.subheader("Perfil mensal")
        mensal = (saida.assign(mes=saida["din_instante"].dt.month)
                  .groupby(["regiao", "mes"])["previsao_mwmed"].mean().reset_index())
        fig, ax = plt.subplots(figsize=(11, 4))
        for regiao, g in mensal.groupby("regiao"):
            estilo = dict(linewidth=2.4, zorder=5) if regiao.startswith("SIN") else {}
            ax.plot(g["mes"], g["previsao_mwmed"], marker="o", markersize=4,
                    label=regiao, **estilo)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                            "Jul", "Ago", "Set", "Out", "Nov", "Dez"])
        ax.set_ylabel("Carga média (MWmed)")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.4)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # --- downloads ---
    st.subheader("Baixar")
    diario = saida.copy()
    diario["din_instante"] = diario["din_instante"].dt.strftime("%Y-%m-%d")
    colunas = [c for c in ["regiao", "id_subsistema", "din_instante", "temp_mean",
                           "nivel_industrial", "resposta_climatica",
                           "previsao_mwmed", "origem"] if c in diario.columns]
    resumo_saida = resumo.copy()
    for c in ("de", "ate"):
        resumo_saida[c] = pd.to_datetime(resumo_saida[c]).dt.strftime("%Y-%m-%d")

    b1, b2, b3, b4 = st.columns(4)
    b1.download_button("Diário (CSV)", para_csv(diario[colunas]),
                       f"previsao_diaria_{rotulo}.csv", "text/csv",
                       width="stretch")
    b2.download_button("Diário (Excel)", para_xlsx(diario[colunas], "Diario"),
                       f"previsao_diaria_{rotulo}.xlsx", XLSX_MIME, width="stretch")
    b3.download_button("Resumo (CSV)", para_csv(resumo_saida),
                       f"resumo_{rotulo}.csv", "text/csv", width="stretch")
    b4.download_button("Resumo (Excel)", para_xlsx(resumo_saida, "Resumo"),
                       f"resumo_{rotulo}.xlsx", XLSX_MIME, width="stretch")




# --------------------------------------------------------------------------
# Guarda: sem dataset/modelos a interface nao tem o que fazer
# --------------------------------------------------------------------------
try:
    hist = _historico()
    padroes = _padroes()
except DadosAusentes as e:
    st.error(str(e))
    st.code("python scripts/01_coletar.py\n"
            "python scripts/02_construir_dataset.py\n"
            "python scripts/03_treinar.py", language="bash")
    st.stop()


st.title("⚡ Previsão de carga de energia do SIN")
st.caption("Modelo climático — prevê a carga diária a partir de clima, "
           "calendário e patamar industrial. Nenhuma carga observada é usada "
           "como entrada.")

# --------------------------------------------------------------------------
# Barra lateral: entradas
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Entradas")
    sigla = st.selectbox("Subsistema", list(NOMES_REGIOES),
                         format_func=lambda s: NOMES_REGIOES[s])
    dia = st.date_input("Data", value=date(2026, 2, 15),
                        min_value=date(2020, 1, 1), max_value=date(2035, 12, 31))

    p = padroes[sigla]
    g_reg = hist[hist["id_subsistema"] == sigla]

    st.subheader("Clima")
    st.caption(f"Padrão = mediana histórica de {NOMES_REGIOES[sigla]}. "
               "O que você não ajustar usa esse valor.")

    entradas_clima = {}
    for campo in CAMPOS_PEDIDOS:
        col_lo = float(g_reg[campo].min())
        col_hi = float(g_reg[campo].max())
        margem = (col_hi - col_lo) * 0.25
        entradas_clima[campo] = st.slider(
            ROTULOS[campo], round(col_lo - margem, 1), round(col_hi + margem, 1),
            round(float(p[campo]), 1), step=0.1)

    with st.expander("Variáveis secundárias"):
        for campo in [c for c in CAMPOS_CLIMA if c not in CAMPOS_PEDIDOS]:
            entradas_clima[campo] = st.number_input(
                ROTULOS[campo], value=round(float(p[campo]), 2), step=1.0,
                format="%.2f")

    st.divider()
    st.caption("O patamar industrial é preenchido automaticamente pela série "
               "da EPE, defasada em 2 meses.")


entrada = pd.DataFrame([{"id_subsistema": sigla, "din_instante": pd.Timestamp(dia),
                         **entradas_clima}])
resultado = _prever(entrada).iloc[0]
ctx = contexto_historico(sigla, pd.Timestamp(dia))

aba_sim, aba_curva, aba_ano, aba_lote, aba_res = st.tabs(
    ["Simulador", "Resposta térmica", "Projeção anual", "Previsão em lote",
     "Desempenho do modelo"])

# --------------------------------------------------------------------------
# Simulador
# --------------------------------------------------------------------------
with aba_sim:
    previsto = float(resultado["previsao_mwmed"])
    dif = previsto - ctx["carga_media"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Carga prevista", f"{_fmt(previsto)} MWmed",
              f"{dif:+,.0f} vs. média da época".replace(",", "."))
    c2.metric("Nível de base", f"{_fmt(resultado['nivel_industrial'])} MWmed",
              help="Patamar estimado a partir do consumo industrial da região.")
    c3.metric("Resposta climática",
              f"{resultado['resposta_climatica']:+,.0f} MWmed".replace(",", "."),
              help="Quanto o clima e o calendário somam ou tiram do patamar.")
    c4.metric("Dia da semana",
              ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"][dia.weekday()]
              + (" · feriado" if resultado["is_feriado"] else ""))

    st.divider()
    esq, dir_ = st.columns([3, 2])

    with esq:
        st.subheader("Onde essa previsão cai no histórico")
        fig, ax = plt.subplots(figsize=(8, 3.4))
        dia_ano = pd.Timestamp(dia).dayofyear
        janela = g_reg[(g_reg["din_instante"].dt.dayofyear - dia_ano).abs() <= 10]
        ax.hist(janela["val_cargaenergiamwmed"], bins=22, color=AZUL, alpha=0.65,
                label=f"Histórico ({ctx['n_dias']} dias na mesma época)")
        ax.axvline(previsto, color=VERMELHO, linewidth=2.5, label="Previsão")
        ax.axvline(ctx["carga_media"], color=CINZA, linestyle="--", linewidth=1.5,
                   label="Média da época")
        ax.set_xlabel("Carga (MWmed)")
        ax.set_ylabel("Dias")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.4)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with dir_:
        st.subheader("Composição")
        fig, ax = plt.subplots(figsize=(5, 3.4))
        nivel = float(resultado["nivel_industrial"])
        clima = float(resultado["resposta_climatica"])
        ax.barh(["Previsão"], [nivel], color=AZUL, label="Nível de base")
        ax.barh(["Previsão"], [clima], left=[nivel],
                color=VERMELHO if clima >= 0 else "#2ca02c",
                label="Resposta climática")
        ax.set_xlabel("MWmed")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, axis="x", linestyle=":", alpha=0.4)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            f"Faixa histórica nessa época: {_fmt(ctx['carga_min'])} a "
            f"{_fmt(ctx['carga_max'])} MWmed · temperatura média "
            f"{ctx['temp_media']:.1f} °C")

# --------------------------------------------------------------------------
# Curva de resposta termica
# --------------------------------------------------------------------------
with aba_curva:
    st.subheader(f"Como a carga responde à temperatura — {NOMES_REGIOES[sigla]}")
    st.caption("Varre a temperatura mantendo data e demais variáveis fixas. "
               "Os pontos são os dias observados em 2024–2025.")

    curva = _curva(sigla, dia, float(entradas_clima["umid_mean"]))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.scatter(g_reg["temp_mean"], g_reg["val_cargaenergiamwmed"], s=9,
               alpha=0.25, color=AZUL, label="Observado (2024–2025)")
    ax.plot(curva["temp_mean"], curva["previsao_mwmed"], color=VERMELHO,
            linewidth=2.5, label=f"Resposta do modelo em {dia:%d/%m/%Y}")
    ax.scatter([entradas_clima["temp_mean"]], [resultado["previsao_mwmed"]],
               s=160, color="black", zorder=5, marker="X",
               label="Ponto simulado")
    ax.set_xlabel("Temperatura média (°C)")
    ax.set_ylabel("Carga (MWmed)")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    variacao = curva["previsao_mwmed"].max() - curva["previsao_mwmed"].min()
    st.info(f"Nessa data, varrer toda a faixa de temperatura da região move a "
            f"carga em **{_fmt(variacao)} MWmed** "
            f"({variacao / curva['previsao_mwmed'].mean() * 100:.1f}% da média). "
            "É a sensibilidade climática do subsistema.")

# --------------------------------------------------------------------------
# Projecao anual
# --------------------------------------------------------------------------
with aba_ano:
    st.subheader("Carga média do ano inteiro (MWmed) e energia (TWh)")

    a1, a2 = st.columns([1, 2])
    ano_alvo = int(a1.number_input("Ano", min_value=2020, max_value=2040,
                                   value=2026, step=1))
    tem_observado = ano_alvo in set(hist["din_instante"].dt.year)
    fonte = a2.radio(
        "Fonte do clima",
        ["Climatologia histórica (ano normal)"]
        + (["Clima observado do ano"] if tem_observado else []),
        horizontal=True,
        help="Sem clima medido para o ano, o modelo usa o clima típico de cada "
             "dia do ano. Com clima observado, mede-se o acerto real.")

    with st.spinner("Projetando o ano..."):
        diario_ano = _projetar(ano_alvo, fonte.startswith("Clima observado"))

    _render_resultado(diario_ano, str(ano_alvo))

    st.info(
        "**Como ler a média anual.** Ela é dominada pelo patamar de carga, não "
        "pelo clima: em 2025, projetar com o clima observado deu praticamente o "
        "mesmo que projetar com a climatologia (−4,6% contra −3,8% no SIN), "
        "porque as anomalias de temperatura se compensam ao longo de 365 dias. "
        "O clima é o que explica a variação **diária** — embaralhá-lo piora o "
        "erro diário em 55–89%.")

# --------------------------------------------------------------------------
# Lote
# --------------------------------------------------------------------------
with aba_lote:
    st.subheader("Previsão a partir de um arquivo")
    st.caption("Aceita **CSV** e **Excel** (.xlsx, .xlsm, .xls). Obrigatórias: "
               "a região e a data — em qualquer grafia usual. Variáveis de "
               "clima ausentes usam a mediana da região.")

    with st.expander("Nomes de coluna aceitos"):
        st.dataframe(
            pd.DataFrame([{"Coluna": c, "Também aceita": ", ".join(a[1:6])}
                          for c, a in ALIASES.items()]),
            width="stretch", hide_index=True)
        st.caption("A região pode vir como sigla (N, NE, S, SE) ou por extenso "
                   "(Norte, Nordeste, Sul, Sudeste). Centro-Oeste é lido como "
                   "Sudeste, que é o subsistema do ONS ao qual pertence.")

    exemplo = pd.DataFrame({
        "id_subsistema": ["SIN", "SE", "S", "N", "NE"],
        "din_instante": ["2026-01-15"] * 5,
        "temp_mean": [24.0, 26.5, 24.0, 27.5, 27.0],
        "temp_max": [29.5, 32.0, 30.0, 32.5, 31.0],
        "temp_min": [19.5, 21.0, 18.0, 24.0, 23.0],
        "umid_mean": [70, 70, 72, 80, 68],
    })
    ex1, ex2 = st.columns(2)
    ex1.download_button("Baixar exemplo em CSV", para_csv(exemplo),
                        "exemplo_entrada.csv", "text/csv", width="stretch")
    ex2.download_button(
        "Baixar exemplo em Excel", para_xlsx(exemplo, "Entrada"),
        "exemplo_entrada.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch")

    enviado = st.file_uploader("Arquivo CSV ou Excel",
                               type=[e.lstrip(".") for e in sorted(EXTENSOES)])
    if enviado is not None:
        aba_escolhida = None
        try:
            abas = planilhas(enviado, enviado.name)
        except Exception:                              # noqa: BLE001
            abas = []
        if len(abas) > 1:
            aba_escolhida = st.selectbox("Planilha", abas)

        try:
            bruto = ler_tabela(enviado, enviado.name, aba_escolhida)
        except FormatoNaoSuportado as e:
            st.error(str(e))
            bruto = None
        except Exception as e:                         # noqa: BLE001
            st.error(f"Não foi possível processar o arquivo: {e}")
            bruto = None

    if enviado is not None and bruto is not None:
        d_ini, d_fim = bruto["din_instante"].min(), bruto["din_instante"].max()
        st.caption(f"Arquivo lido: {len(bruto)} linhas · "
                   f"{d_ini:%d/%m/%Y} a {d_fim:%d/%m/%Y} · "
                   f"regiões: {', '.join(sorted(bruto['id_subsistema'].unique()))}")

        completar = st.checkbox(
            "Completar o período com a climatologia histórica", value=False,
            help="Preenche os dias que faltam no arquivo com o clima típico da "
                 "região naquele dia do ano. Os dias informados por você são "
                 "preservados.")
        fim_periodo = d_fim
        if completar:
            c_a, c_b = st.columns(2)
            fim_periodo = c_a.date_input(
                "Prever até", value=date(int(d_fim.year), 12, 31),
                min_value=d_ini.date(),
                max_value=date(int(d_fim.year) + 2, 12, 31))
            c_b.metric("Dias a projetar",
                       max((pd.Timestamp(fim_periodo) - d_ini).days + 1, 0))

        try:
            entrada_final = (completar_periodo(bruto, ate=fim_periodo)
                             if completar else bruto)
            saida = prever(entrada_final)
        except Exception as e:                         # noqa: BLE001
            st.error(f"Não foi possível gerar a previsão: {e}")
        else:
            if "origem" in saida.columns:
                n_arq = int((saida["origem"] == "arquivo").sum())
                st.success(f"{len(saida)} linhas previstas — {n_arq} com o clima "
                           f"do seu arquivo, {len(saida) - n_arq} completadas "
                           f"pela climatologia.")
            else:
                st.success(f"{len(saida)} linhas previstas.")

            rotulo = (f"{saida['din_instante'].min():%Y-%m-%d}_a_"
                      f"{saida['din_instante'].max():%Y-%m-%d}")
            _render_resultado(saida, rotulo)

            with st.expander("Previsão dia a dia"):
                cols = ["regiao", "din_instante", "temp_mean", "nivel_industrial",
                        "resposta_climatica", "previsao_mwmed"]
                if "origem" in saida.columns:
                    cols.append("origem")
                st.dataframe(saida[cols].style.format({
                    "temp_mean": "{:.1f}", "nivel_industrial": "{:,.0f}",
                    "resposta_climatica": "{:+,.0f}", "previsao_mwmed": "{:,.0f}"}),
                    width="stretch", hide_index=True)

# --------------------------------------------------------------------------
# Desempenho
# --------------------------------------------------------------------------
with aba_res:
    st.subheader("Desempenho medido em 2025 (modelo treinado em 2024)")
    arq = SAIDA / "metricas.csv"
    if not arq.exists():
        st.warning("Rode `python scripts/04_avaliar.py` para gerar as métricas.")
    else:
        m = pd.read_csv(arq)
        modelos = m[~m["modo"].str.startswith("[baseline]")].copy()
        modelos["acuracia_%"] = 100 - modelos["mape"]

        st.dataframe(
            modelos.pivot(index="regiao", columns="modo", values="acuracia_%")
                   .style.format("{:.2f}%"),
            width="stretch")
        st.caption("Acurácia = 100 − MAPE. O modo `climatico` é o usado nesta "
                   "interface: não recebe nenhuma carga observada.")

        with st.expander("Métricas completas (MAE, RMSE, R²)"):
            st.dataframe(m, width="stretch", hide_index=True)

        cont = SAIDA / "teste_contaminacao.csv"
        if cont.exists():
            st.subheader("Teste de contaminação")
            st.caption("Piora do erro ao embaralhar cada grupo de variáveis. "
                       "`autoregressivo` em 0% no modo climático prova que "
                       "nenhuma carga observada entra na previsão.")
            c = pd.read_csv(cont)
            c = c[c["modo"] == "climatico"]
            st.dataframe(
                c.pivot(index="regiao", columns="grupo", values="piora_%")
                 .style.format("{:+.1f}%"),
                width="stretch")

        plots = sorted((SAIDA / "plots").glob("carga_*.png"))
        if plots:
            st.subheader("Séries de 2025")
            escolha = st.selectbox("Gráfico", plots, format_func=lambda p: p.stem)
            st.image(str(escolha), width="stretch")
