"""Caminhos, seed e constantes do projeto (valores definidos em docs/PLANO.md)."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados"
OUTPUTS = RAIZ / "outputs"
SUBMISSAO = RAIZ / "submissao"

SEED = 42

N_PARCELAS_JANELA = 6          # performance window
DPD_PRINCIPAL = 5              # target principal: DPD > 5 dias
DPD_ALTERNATIVO = 30           # target de comparação: DPD > 30 dias
TOL_QUITACAO = 0.01            # centavos: pago >= previsto - tol => quitada
RECALL_ALVO = 0.80             # política: recall mínimo da zona "não aprovar direto"
LGD_CENARIOS = (0.45, 0.60, 0.75)
DATA_OBSERVACAO = None         # a definir no nb01
