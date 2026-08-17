"""Testes da leitura de CSV e Excel enviados pelo usuario."""
import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energia.entrada import (FormatoNaoSuportado, ler_tabela, normalizar,
                             para_csv, para_xlsx, planilhas)

CANONICO = pd.DataFrame({
    "id_subsistema": ["SE", "N"],
    "din_instante": ["2026-01-15", "2026-01-16"],
    "temp_mean": [26.5, 27.0],
    "umid_mean": [70.0, 80.0],
})


def _arquivo(tmp_path, nome, conteudo: bytes):
    p = tmp_path / nome
    p.write_bytes(conteudo)
    return p


# --- CSV: variacoes de separador e codificacao ------------------------------

def test_csv_padrao_virgula(tmp_path):
    p = _arquivo(tmp_path, "a.csv", CANONICO.to_csv(index=False).encode("utf-8"))
    d = ler_tabela(p)
    assert list(d["id_subsistema"]) == ["SE", "N"]
    assert d["temp_mean"].iloc[0] == pytest.approx(26.5)


def test_csv_excel_brasileiro(tmp_path):
    """`;`, virgula decimal e latin-1 -- o que sai do Excel em pt-BR."""
    conteudo = ("id_subsistema;din_instante;temp_mean;umid_mean\n"
                "SE;15/01/2026;26,5;70\n"
                "N;16/01/2026;27,0;80\n").encode("latin-1")
    d = ler_tabela(_arquivo(tmp_path, "b.csv", conteudo))
    assert d["temp_mean"].iloc[0] == pytest.approx(26.5)
    assert d["din_instante"].iloc[0] == pd.Timestamp("2026-01-15")


def test_csv_separado_por_tab(tmp_path):
    conteudo = CANONICO.to_csv(index=False, sep="\t").encode("utf-8")
    d = ler_tabela(_arquivo(tmp_path, "c.tsv", conteudo))
    assert len(d) == 2 and "temp_mean" in d.columns


def test_csv_com_acento_latin1(tmp_path):
    conteudo = ("Regi\xe3o;Data;Temperatura m\xe9dia\n"
                "Nordeste;10/02/2026;28,3\n").encode("latin-1")
    d = ler_tabela(_arquivo(tmp_path, "d.csv", conteudo))
    assert d["id_subsistema"].iloc[0] == "NE"
    assert d["temp_mean"].iloc[0] == pytest.approx(28.3)


# --- Excel ------------------------------------------------------------------

def test_xlsx(tmp_path):
    p = tmp_path / "e.xlsx"
    CANONICO.to_excel(p, index=False)
    d = ler_tabela(p)
    assert list(d["id_subsistema"]) == ["SE", "N"]
    assert d["din_instante"].iloc[1] == pd.Timestamp("2026-01-16")


def test_xlsx_com_varias_abas(tmp_path):
    p = tmp_path / "f.xlsx"
    with pd.ExcelWriter(p, engine="openpyxl") as w:
        CANONICO.to_excel(w, sheet_name="Vazia", index=False)
        CANONICO.assign(temp_mean=[31.0, 32.0]).to_excel(
            w, sheet_name="Cenario", index=False)

    assert planilhas(p) == ["Vazia", "Cenario"]
    d = ler_tabela(p, planilha="Cenario")
    assert d["temp_mean"].iloc[0] == pytest.approx(31.0)
    # sem escolher aba, usa a primeira
    assert ler_tabela(p)["temp_mean"].iloc[0] == pytest.approx(26.5)


def test_planilhas_vazio_para_csv(tmp_path):
    p = _arquivo(tmp_path, "g.csv", CANONICO.to_csv(index=False).encode())
    assert planilhas(p) == []


def test_le_de_buffer_em_memoria():
    """A interface entrega um objeto tipo arquivo, nao um caminho."""
    buf = io.BytesIO(CANONICO.to_csv(index=False).encode("utf-8"))
    d = ler_tabela(buf, nome="upload.csv")
    assert len(d) == 2


# --- normalizacao de nomes e valores ----------------------------------------

@pytest.mark.parametrize("rotulo,esperado", [
    ("Norte", "N"), ("NORDESTE", "NE"), ("sul", "S"), ("Sudeste", "SE"),
    ("se", "SE"), ("Centro-Oeste", "SE"), ("CO", "SE"),
    ("Sudeste/C. Oeste", "SE"),
])
def test_regiao_por_extenso_ou_sigla(rotulo, esperado):
    d = normalizar(pd.DataFrame({"regiao": [rotulo], "data": ["2026-01-15"]}))
    assert d["id_subsistema"].iloc[0] == esperado


@pytest.mark.parametrize("nome", [
    "temperatura", "Temperatura Média", "temp_media", "TEMP", "tmed"])
def test_aliases_de_temperatura(nome):
    d = normalizar(pd.DataFrame({"subsistema": ["SE"], "Data": ["2026-01-15"],
                                 nome: [25.0]}))
    assert d["temp_mean"].iloc[0] == pytest.approx(25.0)


def test_data_brasileira_e_interpretada_com_dia_primeiro():
    d = normalizar(pd.DataFrame({"regiao": ["SE"], "data": ["05/03/2026"]}))
    assert d["din_instante"].iloc[0] == pd.Timestamp("2026-03-05")


# --- erros claros -----------------------------------------------------------

def test_coluna_obrigatoria_ausente():
    with pytest.raises(FormatoNaoSuportado, match="obrigatória"):
        normalizar(pd.DataFrame({"temperatura": [25.0]}))


def test_regiao_desconhecida():
    with pytest.raises(FormatoNaoSuportado, match="não reconhecido"):
        normalizar(pd.DataFrame({"regiao": ["Amazonas"], "data": ["2026-01-15"]}))


def test_data_invalida():
    with pytest.raises(FormatoNaoSuportado, match="data"):
        normalizar(pd.DataFrame({"regiao": ["SE"], "data": ["nao e data"]}))


def test_extensao_nao_suportada(tmp_path):
    p = _arquivo(tmp_path, "h.pdf", b"%PDF-1.4")
    with pytest.raises(FormatoNaoSuportado, match="não suportada"):
        ler_tabela(p)


def test_arquivo_vazio(tmp_path):
    p = _arquivo(tmp_path, "i.csv", b"id_subsistema,din_instante\n")
    with pytest.raises(FormatoNaoSuportado):
        ler_tabela(p)


# --- escrita ----------------------------------------------------------------

def test_xlsx_de_ida_e_volta(tmp_path):
    dados = para_xlsx(CANONICO)
    assert dados[:2] == b"PK"                       # assinatura de zip/xlsx
    lido = pd.read_excel(io.BytesIO(dados))
    assert len(lido) == 2


def test_csv_no_formato_excel_br():
    saida = para_csv(pd.DataFrame({"v": [1.5]}), excel_br=True).decode("utf-8-sig")
    assert ";" in saida or "1,5" in saida
    assert "1,5" in saida


def test_excel_corrompido_da_mensagem_util(tmp_path):
    """Extensao de Excel com conteudo invalido nao pode vazar erro cru."""
    p = _arquivo(tmp_path, "quebrado.xlsx", b"isto nao e uma planilha")
    with pytest.raises(FormatoNaoSuportado, match="planilha"):
        ler_tabela(p)


def test_csv_renomeado_para_xlsx(tmp_path):
    p = _arquivo(tmp_path, "disfarcado.xlsx",
                 CANONICO.to_csv(index=False).encode("utf-8"))
    with pytest.raises(FormatoNaoSuportado):
        ler_tabela(p)
