"""Baixa os dados brutos das tres fontes (ONS, INMET, EPE)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energia.coleta import coletar_tudo
from energia.config import ANO_TESTE, ANO_TREINO

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--anos", type=int, nargs="+", default=[ANO_TREINO, ANO_TESTE])
    p.add_argument("--forcar", action="store_true", help="rebaixa mesmo se ja existir")
    a = p.parse_args()
    coletar_tudo(a.anos, a.forcar)
