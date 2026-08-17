"""Caminhos, constantes e mapeamentos compartilhados por todo o pipeline."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]

DADOS = RAIZ / "data"
BRUTO = DADOS / "raw"
INTERIM = DADOS / "interim"
PROCESSADO = DADOS / "processed"

BRUTO_ONS = BRUTO / "ons"
BRUTO_INMET = BRUTO / "inmet"
BRUTO_EPE = BRUTO / "epe"

MODELOS = RAIZ / "modelos"
SAIDA = RAIZ / "output"

ANO_TREINO = 2024
ANO_TESTE = 2025

# Subsistemas do ONS -- sao os unicos valores de id_subsistema no CSV de carga.
NOMES_SUBSISTEMAS = {
    "N": "Norte",
    "NE": "Nordeste",
    "S": "Sul",
    "SE": "Sudeste/Centro-Oeste",
}
SUBSISTEMAS = list(NOMES_SUBSISTEMAS)

# O SIN nao existe como linha no arquivo do ONS: e o agregado nacional,
# derivado somando os quatro subsistemas. Recebe modelo proprio, treinado
# sobre a serie agregada.
SIN = "SIN"

# Tudo que o projeto modela e exibe (subsistemas + agregado nacional).
NOMES_REGIOES = {**NOMES_SUBSISTEMAS, SIN: "SIN (nacional)"}
REGIOES = list(NOMES_REGIOES)

# O INMET rotula as estacoes pelas 5 macrorregioes do IBGE; o ONS opera 4
# subsistemas. O Centro-Oeste pertence ao subsistema Sudeste -- sem este
# mapeamento as 96 estacoes do CO ficam orfas e sao descartadas no merge.
REGIAO_INMET_PARA_SUBSISTEMA = {
    "N": "N",
    "NE": "NE",
    "S": "S",
    "SE": "SE",
    "CO": "SE",
}

# Colunas do INMET usadas (nomes exatos do cabecalho, encoding latin-1)
COL_INMET_DATA = "Data"
COL_INMET_TEMP = "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)"
COL_INMET_UMID = "UMIDADE RELATIVA DO AR, HORARIA (%)"
COL_INMET_RAD = "RADIACAO GLOBAL (Kj/m²)"
COL_INMET_VENTO = "VENTO, VELOCIDADE HORARIA (m/s)"

# Limites fisicos plausiveis -- valores fora disso sao sensor com defeito.
# O INMET usa -9999 como codigo de ausencia.
LIMITES_INMET = {
    "temp": (-15.0, 55.0),
    "umid": (1.0, 100.0),
    "rad": (0.0, 6000.0),
    "vento": (0.0, 60.0),
}

# Temperaturas de referencia para graus-dia. O conforto termico brasileiro
# fica em torno de 22 C; abaixo de 18 C liga aquecimento (relevante no Sul).
TEMP_BASE_REFRIGERACAO = 22.0
TEMP_BASE_AQUECIMENTO = 18.0

URL_ONS = ("https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/"
           "carga_energia_di/CARGA_ENERGIA_{ano}.csv")
URL_INMET = "https://portal.inmet.gov.br/uploads/dadoshistoricos/{ano}.zip"
URL_EPE = ("https://www.epe.gov.br/sites-pt/publicacoes-dados-abertos/publicacoes/"
           "Documents/CONSUMO%20MENSAL%20DE%20ENERGIA%20EL%c3%89TRICA%20POR%20CLASSE.xlsx")

# O portal do INMET derruba a conexao sem User-Agent de navegador.
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


def garantir_diretorios() -> None:
    for d in (BRUTO_ONS, BRUTO_INMET, BRUTO_EPE, INTERIM, PROCESSADO, MODELOS, SAIDA):
        d.mkdir(parents=True, exist_ok=True)
