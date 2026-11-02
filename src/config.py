from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DADOS = ROOT / "dados"
INTERIM = ROOT / "data" / "interim"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"
SUBMISSAO = ROOT / "submissao"

SEED = 42
JANELA_PARCELAS = 12   # target observado nas N primeiras parcelas
DPD_LIMITE = 30        # atraso (dias) que define inadimplência
TOLERANCIA_PAGO = 0.95 # parcela paga se soma paga >= 95% do previsto
