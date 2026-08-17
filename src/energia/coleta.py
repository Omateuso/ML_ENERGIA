"""Download das tres fontes primarias.

ONS   -- carga diaria por subsistema (alvo do modelo)
INMET -- clima horario por estacao (variaveis explicativas)
EPE   -- consumo mensal por classe, inclusive industrial (nivel de base)
"""
from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

from .config import (BRUTO_EPE, BRUTO_INMET, BRUTO_ONS, URL_EPE, URL_INMET,
                     URL_ONS, USER_AGENT, garantir_diretorios)


def _baixar(url: str, destino: Path, forcar: bool = False) -> Path:
    if destino.exists() and not forcar and destino.stat().st_size > 0:
        print(f"  ja existe, pulando: {destino.name} "
              f"({destino.stat().st_size / 1e6:.1f} MB)")
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"  baixando {destino.name} ...")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    parcial = destino.with_suffix(destino.suffix + ".parcial")
    with urllib.request.urlopen(req, timeout=300) as r, open(parcial, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    parcial.replace(destino)
    print(f"  ok: {destino.name} ({destino.stat().st_size / 1e6:.1f} MB)")
    return destino


def baixar_ons(anos: list[int], forcar: bool = False) -> list[Path]:
    print("ONS - carga de energia diaria por subsistema")
    return [_baixar(URL_ONS.format(ano=a), BRUTO_ONS / f"CARGA_ENERGIA_{a}.csv", forcar)
            for a in anos]


def baixar_inmet(anos: list[int], forcar: bool = False) -> list[Path]:
    """Baixa e extrai os zips anuais.

    Extrai para um diretorio limpo por ano: o INMET republica os arquivos com o
    periodo no nome (..._A_30-11-2025 vs ..._A_31-12-2025), entao acumular
    downloads na mesma pasta duplica estacoes -- foi o que aconteceu com o
    dataset antigo de 2025 (1177 arquivos para 565 estacoes).
    """
    print("INMET - dados horarios por estacao automatica")
    pastas = []
    for ano in anos:
        zip_path = _baixar(URL_INMET.format(ano=ano), BRUTO_INMET / f"{ano}.zip", forcar)
        pasta = BRUTO_INMET / str(ano)
        if pasta.exists() and not forcar:
            n = len(list(pasta.rglob("*.CSV")))
            if n:
                print(f"  ja extraido: {pasta.name}/ ({n} estacoes)")
                pastas.append(pasta)
                continue
        if pasta.exists():
            shutil.rmtree(pasta)
        pasta.mkdir(parents=True)
        print(f"  extraindo {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(pasta)
        # Alguns anos vem com um subdiretorio interno; achata para um nivel so.
        for csv in list(pasta.rglob("*.CSV")):
            if csv.parent != pasta:
                csv.rename(pasta / csv.name)
        for sub in [p for p in pasta.iterdir() if p.is_dir()]:
            shutil.rmtree(sub)
        print(f"  ok: {pasta.name}/ ({len(list(pasta.glob('*.CSV')))} estacoes)")
        pastas.append(pasta)
    return pastas


def baixar_epe(forcar: bool = False) -> Path:
    print("EPE - consumo mensal de energia eletrica por classe")
    return _baixar(URL_EPE, BRUTO_EPE / "consumo_mensal_por_classe.xlsx", forcar)


def coletar_tudo(anos: list[int], forcar: bool = False) -> None:
    garantir_diretorios()
    baixar_ons(anos, forcar)
    baixar_inmet(anos, forcar)
    baixar_epe(forcar)
