"""Genera un CSV de solicitudes nuevas simuladas.   python simular_solicitudes.py 10 --semilla 42 --salida data/ejemplos/nuevos_clientes_10.csv"""
import argparse
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))
from simular import COLUMNAS, simular  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("n", type=int, nargs="?", default=10)
p.add_argument("--semilla", type=int, default=42)
p.add_argument("--salida", default=str(RAIZ / "data" / "ejemplos" / "nuevos_clientes_10.csv"))
a = p.parse_args()
base = pd.read_parquet(RAIZ / "models" / "base_clientes", columns=COLUMNAS)
simular(base, a.n, a.semilla).to_csv(a.salida, index=False)
print("escrito", a.salida)
