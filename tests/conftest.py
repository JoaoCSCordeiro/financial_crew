"""Fixtures partilhadas pelos testes."""

import sys
from pathlib import Path

# Permite `pytest` correr a partir da raiz do projeto sem instalar o pacote,
# adicionando a raiz ao sys.path (mesmo padrao usado em PYTHONPATH=. nos
# scripts manuais -- ver README).
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
