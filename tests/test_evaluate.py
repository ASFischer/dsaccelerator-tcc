import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluate as ev
import features as ft


def test_limiar_e_metricas_contra_contagem_manual():
    y, p = [1, 1, 0, 0, 0, 1], [.9, .8, .7, .4, .2, .3]
    lim = ev.limiar_max_f1(y, p)
    assert lim == pytest.approx(.8)  # TP=2, FP=0, FN=1: precision 1, recall 2/3, F1 0,8 (o máximo)
    m = ev.metricas(y, p, lim)
    assert m["recall"] == pytest.approx(2 / 3) and m["precision"] == pytest.approx(1) and m["f1"] == pytest.approx(.8)
    assert m["taxa_aprovacao"] == pytest.approx(4 / 6)


def test_delong_confere_com_auc_e_empate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p1, p2 = rng.random(2000) + y * .3, rng.random(2000) + y * .1
    a1, a2, z, pv = ev.delong(y, p1, p2)
    assert a1 == pytest.approx(roc_auc_score(y, p1)) and a2 == pytest.approx(roc_auc_score(y, p2))
    assert z > 3 and pv < 0.01
    assert ev.delong(y, p1, p1)[2:] == (0.0, 1.0)  # mesmos scores: sem diferença, não NaN


def test_separacao_oot_exige_data_de_decisao():
    datas = pd.Series(pd.to_datetime(["2022-06-01", "2023-06-01", "2024-06-01"]))
    assert ft.separar_oot(datas).tolist() == ["treino", "valid", "teste"]
    with pytest.raises(AssertionError, match="data_decisao nula"):
        ft.separar_oot(pd.Series(pd.to_datetime(["2022-06-01", None])))
