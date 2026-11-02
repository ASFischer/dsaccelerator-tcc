"""Roda o projeto inteiro na ordem: dados -> modelo -> previsões -> testes -> notebooks.

    python run.py                  tudo
    python run.py --sem-notebooks  só src e testes
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
NOTEBOOKS = sorted((ROOT / "notebooks").glob("0*.ipynb"))

PASSOS = [
    ("dataset", ["-m", "src.build_dataset"]),
    ("modelo", ["-m", "src.train"]),
    ("previsões", ["-m", "src.predict"]),
    ("testes", ["-m", "pytest", "-q", "tests"]),
]


def main():
    passos = list(PASSOS)
    if "--sem-notebooks" not in sys.argv:  # os notebooks 05-07 leem o modelo e o CSV gerados acima
        passos += [(nb.stem, ["-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(nb)])
                   for nb in NOTEBOOKS]
    for nome, args in passos:
        print(f"\n=== {nome} ===", flush=True)
        if subprocess.run([sys.executable, *args], cwd=ROOT).returncode:
            sys.exit(f"falhou em: {nome}")
    print("\nok: tudo rodou")


if __name__ == "__main__":
    main()
