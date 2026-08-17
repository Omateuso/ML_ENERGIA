"""Leitura tolerante de tabelas enviadas pelo usuario (CSV e Excel).

Planilha de usuario nao chega no formato canonico. Este modulo absorve as
variacoes previsiveis em vez de exigir que a pessoa formate o arquivo:

- CSV exportado do Excel brasileiro vem com `;`, virgula decimal e latin-1
- os nomes das colunas vem em portugues corrente ("Temperatura media", "Data")
- a regiao vem por extenso ("Nordeste") em vez da sigla do ONS ("NE")
"""
from __future__ import annotations

import io
import unicodedata
from pathlib import Path

import pandas as pd

EXTENSOES_CSV = {".csv", ".txt", ".tsv"}
EXTENSOES_EXCEL = {".xlsx", ".xlsm", ".xls"}
EXTENSOES = EXTENSOES_CSV | EXTENSOES_EXCEL

# Coluna canonica -> nomes aceitos (ja normalizados: sem acento, minusculo)
ALIASES = {
    "id_subsistema": ["id_subsistema", "subsistema", "regiao", "sub", "sigla",
                      "id subsistema", "nom_subsistema"],
    "din_instante": ["din_instante", "data", "dia", "date", "data_previsao",
                     "din instante", "datahora"],
    "temp_mean": ["temp_mean", "temp_media", "temperatura", "temperatura media",
                  "temp", "tmed", "temperatura_media", "temp media"],
    "temp_max": ["temp_max", "temperatura maxima", "tmax", "temperatura_maxima",
                 "temp maxima", "maxima"],
    "temp_min": ["temp_min", "temperatura minima", "tmin", "temperatura_minima",
                 "temp minima", "minima"],
    "umid_mean": ["umid_mean", "umidade", "umidade media", "umidade relativa",
                  "ur", "umidade_media", "umid media"],
    "umid_min": ["umid_min", "umidade minima", "umidade_minima", "ur min"],
    "rad_sum": ["rad_sum", "radiacao", "radiacao global", "radiacao_global"],
    "vento_mean": ["vento_mean", "vento", "velocidade do vento",
                   "velocidade_vento", "vento medio"],
    "precip_sum": ["precip_sum", "precipitacao", "chuva", "precipitacao total"],
}

# Como o usuario pode escrever a regiao -> id_subsistema do ONS
REGIOES_ACEITAS = {
    "n": "N", "norte": "N",
    "ne": "NE", "nordeste": "NE",
    "s": "S", "sul": "S",
    "se": "SE", "sudeste": "SE", "sudeste/centro-oeste": "SE",
    "sudeste/c. oeste": "SE", "sudeste/c.oeste": "SE", "sudeste e centro-oeste": "SE",
    # o Centro-Oeste pertence ao subsistema Sudeste no ONS
    "co": "SE", "centro-oeste": "SE", "centro oeste": "SE",
    # agregado nacional
    "sin": "SIN", "brasil": "SIN", "nacional": "SIN", "br": "SIN",
    "sistema interligado nacional": "SIN", "total": "SIN",
}

COLUNAS_NUMERICAS = ["temp_mean", "temp_max", "temp_min", "umid_mean",
                     "umid_min", "rad_sum", "vento_mean", "precip_sum"]


class FormatoNaoSuportado(Exception):
    pass


def _norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.strip().lower().split())


def _mapa_aliases() -> dict[str, str]:
    return {_norm(a): canonica for canonica, lista in ALIASES.items() for a in lista}


def planilhas(fonte, nome: str | None = None) -> list[str]:
    """Nomes das abas de um Excel. Lista vazia se nao for Excel."""
    ext = Path(nome or getattr(fonte, "name", "") or "").suffix.lower()
    if ext not in EXTENSOES_EXCEL:
        return []
    if hasattr(fonte, "seek"):
        fonte.seek(0)
    return pd.ExcelFile(fonte).sheet_names


def _ler_csv(fonte) -> pd.DataFrame:
    """Tenta as combinacoes plausiveis de separador e codificacao."""
    dados = fonte.read() if hasattr(fonte, "read") else Path(fonte).read_bytes()
    if isinstance(dados, str):
        dados = dados.encode("utf-8")

    melhor, melhor_cols = None, 0
    erros = []
    for encoding in ("utf-8-sig", "latin-1"):
        for sep in (";", ",", "\t", "|"):
            try:
                df = pd.read_csv(io.BytesIO(dados), sep=sep, encoding=encoding,
                                 engine="python")
            except Exception as e:                     # noqa: BLE001
                erros.append(f"{encoding}/{sep!r}: {e}")
                continue
            # O separador certo e o que mais divide o cabecalho em colunas.
            if df.shape[1] > melhor_cols:
                melhor, melhor_cols = df, df.shape[1]
    if melhor is None or melhor_cols < 2:
        raise FormatoNaoSuportado(
            "Não foi possível interpretar o CSV. Tentativas:\n  "
            + "\n  ".join(erros[:4]))
    return melhor


def _ler_excel(fonte, planilha: str | int | None) -> pd.DataFrame:
    if hasattr(fonte, "seek"):
        fonte.seek(0)
    try:
        return pd.read_excel(fonte, sheet_name=planilha if planilha is not None else 0)
    except ImportError as e:
        raise FormatoNaoSuportado(
            f"Falta a biblioteca para ler este Excel ({e}). "
            "Para .xlsx instale openpyxl; para .xls antigo, xlrd.") from e
    except Exception as e:                             # noqa: BLE001
        # Arquivo corrompido, protegido por senha, ou extensao que nao bate
        # com o conteudo (um .csv renomeado para .xlsx, por exemplo).
        raise FormatoNaoSuportado(
            f"Não foi possível abrir a planilha: {e}. Verifique se o arquivo "
            "não está corrompido nem protegido por senha, e se a extensão "
            "corresponde ao formato real.") from e


def normalizar(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia colunas para o padrao do projeto e converte tipos."""
    mapa = _mapa_aliases()
    renomear, ja_usadas = {}, set()
    for col in df.columns:
        canonica = mapa.get(_norm(col))
        if canonica and canonica not in ja_usadas:
            renomear[col] = canonica
            ja_usadas.add(canonica)
    df = df.rename(columns=renomear)

    faltando = [c for c in ("id_subsistema", "din_instante") if c not in df.columns]
    if faltando:
        raise FormatoNaoSuportado(
            f"Coluna obrigatória ausente: {', '.join(faltando)}. "
            f"Colunas encontradas: {list(df.columns)}")

    invalidas = []
    def _regiao(v):
        s = REGIOES_ACEITAS.get(_norm(v))
        if s is None:
            invalidas.append(v)
        return s
    df["id_subsistema"] = df["id_subsistema"].map(_regiao)
    if invalidas:
        raise FormatoNaoSuportado(
            f"Subsistema não reconhecido: {sorted(set(map(str, invalidas)))[:5]}. "
            f"Use N, NE, S, SE (ou Norte, Nordeste, Sul, Sudeste).")

    # dayfirst: planilha brasileira escreve 05/03/2026 como 5 de marco
    df["din_instante"] = pd.to_datetime(df["din_instante"], dayfirst=True,
                                        errors="coerce")
    if df["din_instante"].isna().any():
        n = int(df["din_instante"].isna().sum())
        raise FormatoNaoSuportado(f"{n} linha(s) com data inválida na coluna de data.")

    for col in COLUNAS_NUMERICAS:
        if col not in df.columns or pd.api.types.is_numeric_dtype(df[col]):
            continue
        # Coluna veio como texto. Só onde existe virgula tratamos o ponto como
        # separador de milhar -- caso contrario "26.5" viraria 265.
        s = df[col].astype("string")
        virgula = s.str.contains(",", na=False)
        s = s.where(~virgula, s.str.replace(".", "", regex=False)
                               .str.replace(",", ".", regex=False))
        df[col] = pd.to_numeric(s, errors="coerce")
    return df


def ler_tabela(fonte, nome: str | None = None,
               planilha: str | int | None = None) -> pd.DataFrame:
    """Le CSV ou Excel de um caminho ou de um arquivo enviado, e normaliza."""
    ext = Path(nome or getattr(fonte, "name", "") or "").suffix.lower()
    if ext in EXTENSOES_EXCEL:
        bruto = _ler_excel(fonte, planilha)
    elif ext in EXTENSOES_CSV or ext == "":
        bruto = _ler_csv(fonte)
    else:
        raise FormatoNaoSuportado(
            f"Extensão '{ext}' não suportada. Use: "
            + ", ".join(sorted(EXTENSOES)))
    if bruto.empty:
        raise FormatoNaoSuportado("O arquivo não tem nenhuma linha de dados.")
    return normalizar(bruto)


def para_xlsx(df: pd.DataFrame, planilha: str = "Previsoes") -> bytes:
    """Serializa um DataFrame em .xlsx para download."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as escritor:
        df.to_excel(escritor, sheet_name=planilha, index=False)
    return buffer.getvalue()


def para_csv(df: pd.DataFrame, excel_br: bool = False) -> bytes:
    """CSV utf-8. Com excel_br=True usa `;` e vírgula decimal."""
    if excel_br:
        return df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
    return df.to_csv(index=False).encode("utf-8")
